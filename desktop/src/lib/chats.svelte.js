import { load, save } from './storage.js';
import * as api from './hypnos.js';
import { notifyIfAway } from './appPrefs.js';

const CHATS_KEY = 'chats';
const LAST_MODEL_KEY = 'lastModel';
const THINKING_KEY = 'thinking';
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
 * With thinking on it also keeps the model's thoughts (`reasoning`) and how long they took
 * (`thoughtSeconds`, set when the answer starts); they are shown, never sent back.
 */
export class ChatStore {
  chats = $state([]);
  currentId = $state(null);
  lastModel = $state(null);
  /** Thinking level for new answers (`none`, `low`, `medium`, `high`). */
  thinking = $state('none');
  /** The telemetry of the most recent response, for the stats panel. */
  lastTelemetry = $state(null);
  /** Chat id -> request id of the answer being generated. Not persisted. */
  running = $state({});

  current = $derived(this.chats.find((c) => c.id === this.currentId) ?? null);

  #saveTimer;

  async init() {
    const chats = await load(CHATS_KEY, []);
    // A response cut off by quitting the app has no telemetry and no error: mark it.
    for (const chat of chats) {
      for (const m of chat.messages) {
        if (!m.pending) continue;
        delete m.pending;
        delete m.tool;
        delete m.thinkingSince;
        m.stopped = true;
      }
    }
    this.chats = chats.sort((a, b) => b.updatedAt - a.updatedAt);
    // The stats panel starts from the newest answer that has telemetry.
    const last = this.chats.map((c) => c.messages.findLast((m) => m.telemetry)).find(Boolean);
    if (last) this.lastTelemetry = { model: last.model, telemetry: last.telemetry, usage: last.usage };
    this.lastModel = await load(LAST_MODEL_KEY, null);
    this.thinking = await load(THINKING_KEY, 'none');
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

  /** The model new chats start with; the open chat takes it too while it has no messages. */
  setDefaultModel(model) {
    this.lastModel = model;
    save(LAST_MODEL_KEY, model);
    const chat = this.current;
    if (chat && chat.messages.length === 0) {
      chat.model = model;
      this.#persist();
    }
  }

  setThinking(level) {
    this.thinking = level;
    save(THINKING_KEY, level);
  }

  setSystem(chat, system) {
    chat.system = system;
    this.#persist();
  }

  /** `attachments`: prepared by `attachFromPath` / `attachFromBytes` (they are saved with the chat). */
  async send(chat, text, attachments = []) {
    if ((!text.trim() && attachments.length === 0) || this.isRunning(chat.id)) return;
    const command = text.match(/^\/image\s+([\s\S]+)/);
    if (command && attachments.length === 0) return this.generateImage(chat, command[1].trim());
    if (chat.messages.length === 0) chat.title = titleFrom(text || attachments[0].name);
    chat.messages.push({ role: 'user', content: text, ...(attachments.length ? { attachments } : {}) });
    await this.#generate(chat);
  }

  /**
   * The user request at `index` (or the one an answer at `index` replies to) is asked again:
   * everything after it is dropped and a new answer streams in. `/image` requests make a new image.
   */
  async regenerateFrom(chat, index) {
    if (this.isRunning(chat.id)) return;
    let at = Math.min(index, chat.messages.length - 1);
    while (at >= 0 && chat.messages[at].role !== 'user') at--;
    if (at < 0) return;
    const request = chat.messages[at];
    const command = !request.attachments?.length && request.content?.match(/^\/image\s+([\s\S]+)/);
    if (command) {
      chat.messages.splice(at);
      return this.generateImage(chat, command[1].trim());
    }
    chat.messages.splice(at + 1);
    await this.#generate(chat);
  }

  /** How many later turns `regenerateFrom(chat, index)` would drop (0: only the answer itself). */
  laterTurns(chat, index) {
    let at = index;
    while (at >= 0 && chat.messages[at].role !== 'user') at--;
    return chat.messages.slice(at + 1).filter((m) => m.role === 'user').length;
  }

  stop(id) {
    const requestId = this.running[id];
    // A direct image request cannot be cancelled half way; a chat (with or without the tool) can.
    if (requestId && requestId !== 'image') api.cancelChat(requestId);
  }

  async #generate(chat) {
    const requestId = crypto.randomUUID();
    this.running[chat.id] = requestId;
    chat.messages.push({ role: 'assistant', content: '', model: chat.model, pending: true, at: Date.now() });
    // Mutate through the proxy so the UI follows the stream.
    const answer = chat.messages.at(-1);
    chat.updatedAt = Date.now();
    this.#bumpToTop(chat);

    // Only what was said goes back to the server: never an answer's `reasoning`.
    // Each earlier image goes back with its settings (the chat model reads them, DESK-22); only
    // the latest one carries its pixels, as the reference an edit ("make it darker") can use.
    const sent = chat.messages.slice(0, -1).filter((m) => !m.error && (m.content || m.attachments?.length || m.image));
    const latestImage = sent.findLast((m) => m.image);
    const history = sent.map((m) => ({
      role: m.role,
      content: m.content ?? '',
      attachments: m.attachments ?? [],
      image_prompt: m.image?.prompt ?? null,
      image: m.image ? imageNote(m.image, m === latestImage) : null,
    }));
    if (chat.system.trim()) history.unshift({ role: 'system', content: chat.system });

    // The thoughts end where the answer (or a tool call) starts.
    const endThinking = () => {
      if (answer.thinkingSince === undefined) return;
      answer.thoughtSeconds = (Date.now() - answer.thinkingSince) / 1000;
      delete answer.thinkingSince;
    };

    try {
      const { cancelled } = await api.chat(requestId, chat.model, history, {
        tools: api.imageToolOn(),
        assistant: api.assistantToolsOn(),
        thinking: this.thinking,
        onReasoning: (text) => {
          if (!answer.reasoning) {
            answer.reasoning = '';
            answer.thinkingSince = Date.now();
          }
          answer.reasoning += text;
        },
        onToolCall: ({ event, ...call }) => {
          endThinking();
          answer.tool = { ...call, startedAt: Date.now() };
        },
        onImage: ({ event, ...image }) => {
          // A turn may make more than one (DESK-23): the first is the answer's image.
          if (answer.image) (answer.moreImages ??= []).push(image);
          else answer.image = image;
          delete answer.tool;
        },
        onStep: ({ event, ...step }) => {
          endThinking();
          const steps = (answer.steps ??= []);
          const i = steps.findIndex((s) => s.id === step.id);
          if (i >= 0) steps[i] = step;
          else steps.push(step);
        },
        onSpeak: (text) => {
          api.speak(`tool-${requestId}`, text, true, api.voicePrefs()).catch((e) => (answer.note = api.errorMessage(e)));
        },
        onToolError: (message) => {
          delete answer.tool;
          answer.note = `The image could not be made: ${message}`;
        },
        onModel: (id, reason) => {
          answer.model = id;
          answer.note = `answered by ${id}: ${reason}`;
        },
        onDelta: (text) => {
          endThinking();
          answer.content += text;
        },
        onDone: ({ telemetry, usage, requestId }) => {
          answer.telemetry = telemetry ?? undefined;
          answer.usage = usage ?? undefined;
          answer.requestId = requestId ?? undefined;
          if (telemetry) this.lastTelemetry = { model: answer.model, telemetry, usage };
        },
      });
      if (cancelled) answer.stopped = true;
      else notifyIfAway(answer.image ? 'Image ready' : `Answer from ${answer.model ?? chat.model}`, answer.content || answer.image?.prompt || chat.title);
    } catch (e) {
      answer.error = api.errorMessage(e);
    } finally {
      endThinking();
      delete answer.pending;
      delete this.running[chat.id];
      chat.updatedAt = Date.now();
      this.#persist();
    }
  }

  /** The Image button and `/image <prompt>`: an image straight from the image model. */
  async generateImage(chat, prompt) {
    if (!prompt || this.isRunning(chat.id)) return;
    if (chat.messages.length === 0) chat.title = titleFrom(prompt);
    chat.messages.push({ role: 'user', content: `/image ${prompt}` });
    chat.messages.push({ role: 'assistant', content: '', pending: true, at: Date.now(), tool: { prompt, startedAt: Date.now() } });
    await this.#makeImage(chat, chat.messages.at(-1), prompt);
  }

  /** Another image from the same settings with a new seed, in place. */
  async regenerateImage(chat, answer) {
    if (!answer.image || this.isRunning(chat.id)) return;
    const { prompt, size, negativePrompt, steps } = answer.image;
    const settings = { size, negativePrompt, steps };
    answer.pending = true;
    answer.tool = { prompt, ...settings, startedAt: Date.now() };
    delete answer.error;
    await this.#makeImage(chat, answer, prompt, settings);
  }

  async #makeImage(chat, answer, prompt, settings = {}) {
    this.running[chat.id] = 'image';
    this.#bumpToTop(chat);
    try {
      answer.image = await api.generateImage(prompt, settings);
      answer.model = answer.image.model;
      notifyIfAway('Image ready', prompt);
      if (answer.image.telemetry) this.lastTelemetry = { model: answer.image.model, telemetry: answer.image.telemetry };
    } catch (e) {
      answer.error = api.errorMessage(e);
    } finally {
      delete answer.pending;
      delete answer.tool;
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

/** An image's settings as the history sends them; `withPixels` adds its data URL. */
function imageNote(image, withPixels) {
  const { prompt, size, seed, steps, negativePrompt, model, dataUrl } = image;
  return { prompt, size, seed, steps, negativePrompt, model, dataUrl: withPixels ? dataUrl : null };
}
