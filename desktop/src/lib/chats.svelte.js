import { load, save } from './storage.js';
import { describeError } from './errors.js';

const CHATS_KEY = 'chats';
const LAST_MODEL_KEY = 'lastModel';
const TITLE_CHARS = 48;

function newId() {
  return crypto.randomUUID();
}

function titleFrom(text) {
  const line = text.trim().split('\n')[0];
  return line.length > TITLE_CHARS ? `${line.slice(0, TITLE_CHARS - 1)}…` : line || 'New chat';
}

/**
 * The conversations, persisted as one list. A chat is
 * `{ id, title, model, system, messages, updatedAt }`; an assistant message carries the
 * response's `telemetry` and `usage`, or `error`, and `stopped` when the user cut it short.
 */
export class ChatStore {
  chats = $state([]);
  currentId = $state(null);
  lastModel = $state(null);
  /** The telemetry of the most recent response, for the stats panel. */
  lastTelemetry = $state(null);
  /** Chat id -> AbortController of the response being generated. Not persisted. */
  running = $state({});

  current = $derived(this.chats.find((c) => c.id === this.currentId) ?? null);

  #client;
  #saveTimer;

  constructor(client) {
    this.#client = client;
  }

  /** The client for new requests, after the connection settings changed. */
  setClient(client) {
    this.#client = client;
  }

  async init() {
    const chats = await load(CHATS_KEY, []);
    // A response cut off by quitting the app has no telemetry and no error: mark it.
    for (const chat of chats) {
      for (const m of chat.messages) {
        if (!m.pending) continue;
        delete m.pending;
        m.stopped = true;
      }
    }
    this.chats = chats.sort((a, b) => b.updatedAt - a.updatedAt);
    // The stats panel starts from the newest answer that has telemetry.
    const last = this.chats.map((c) => c.messages.findLast((m) => m.telemetry)).find(Boolean);
    if (last) this.lastTelemetry = { model: last.model, telemetry: last.telemetry, usage: last.usage };
    this.lastModel = await load(LAST_MODEL_KEY, null);
    this.currentId = this.chats[0]?.id ?? null;
  }

  isRunning(id) {
    return id in this.running;
  }

  newChat(model = this.lastModel) {
    const empty = this.chats.find((c) => c.messages.length === 0);
    if (empty) {
      empty.model = model;
      this.currentId = empty.id;
      return empty;
    }
    const chat = { id: newId(), title: 'New chat', model, system: '', messages: [], updatedAt: Date.now() };
    this.chats.unshift(chat);
    this.currentId = chat.id;
    this.#persist();
    return chat;
  }

  select(id) {
    this.currentId = id;
  }

  remove(id) {
    this.stop(id);
    this.chats = this.chats.filter((c) => c.id !== id);
    if (this.currentId === id) this.currentId = this.chats[0]?.id ?? null;
    this.#persist();
  }

  setModel(chat, model) {
    chat.model = model;
    this.lastModel = model;
    save(LAST_MODEL_KEY, model);
    this.#persist();
  }

  setSystem(chat, system) {
    chat.system = system;
    this.#persist();
  }

  async send(chat, text) {
    if (!text.trim() || this.isRunning(chat.id)) return;
    if (chat.messages.length === 0) chat.title = titleFrom(text);
    chat.messages.push({ role: 'user', content: text });
    await this.#generate(chat);
  }

  /** Replaces the last assistant answer with a new one. */
  async regenerate(chat) {
    if (this.isRunning(chat.id)) return;
    while (chat.messages.length && chat.messages.at(-1).role === 'assistant') chat.messages.pop();
    if (chat.messages.length) await this.#generate(chat);
  }

  stop(id) {
    this.running[id]?.abort();
  }

  async #generate(chat) {
    const controller = new AbortController();
    this.running[chat.id] = controller;
    chat.messages.push({ role: 'assistant', content: '', model: chat.model, pending: true });
    // Mutate through the proxy so the UI follows the stream.
    const answer = chat.messages.at(-1);
    chat.updatedAt = Date.now();
    this.#bumpToTop(chat);

    const history = chat.messages
      .slice(0, -1)
      .filter((m) => !m.error && m.content)
      .map(({ role, content }) => ({ role, content }));
    if (chat.system.trim()) history.unshift({ role: 'system', content: chat.system });

    let stream;
    controller.signal.addEventListener('abort', () => stream?.close(), { once: true });
    try {
      stream = await this.#client.chatStream(chat.model, history, { signal: controller.signal });
      for await (const piece of stream.text()) answer.content += piece;
      answer.telemetry = stream.telemetry;
      answer.usage = stream.usage;
      if (stream.telemetry) this.lastTelemetry = { model: chat.model, telemetry: stream.telemetry, usage: stream.usage };
    } catch (e) {
      if (!controller.signal.aborted) answer.error = describeError(e);
    } finally {
      if (controller.signal.aborted) answer.stopped = true;
      delete answer.pending;
      delete this.running[chat.id];
      chat.updatedAt = Date.now();
      this.#persist();
    }
  }

  #bumpToTop(chat) {
    const i = this.chats.indexOf(chat);
    if (i > 0) this.chats.unshift(...this.chats.splice(i, 1));
  }

  #persist() {
    clearTimeout(this.#saveTimer);
    this.#saveTimer = setTimeout(() => save(CHATS_KEY, $state.snapshot(this.chats)), 300);
  }
}
