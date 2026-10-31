<script>
  import { onMount, tick } from 'svelte';
  import { open as openDialog, save as saveDialog } from '@tauri-apps/plugin-dialog';
  import { getCurrentWebview } from '@tauri-apps/api/webview';
  import { listen } from '@tauri-apps/api/event';
  import Markdown from './Markdown.svelte';
  import Attachments from './Attachments.svelte';
  import * as api from './hypnos.js';
  import { sampleFrames } from './video.js';
  import logo from '../assets/logo.png';
  import { telemetrySummary, tierName } from './format.js';

  let { store, models, serverState, onsettings, onlocalserver, onlog } = $props();

  let input = $state('');
  let showSystem = $state(false);
  let scroller = $state();
  let textarea = $state();
  let stickToBottom = true;
  /** Files being prepared or ready for the next message: `{ id, name, status, attachment?, error? }`. */
  let pending = $state([]);
  let dragging = $state(false);
  /** Microphone (DESK-9): `idle`, `recording` or `transcribing`. */
  let mic = $state('idle');
  let micLevel = $state(0);
  let micSeconds = $state(0);
  let micError = $state(null);
  let micAutoSend = $state(readFlag('hypnos.desktop.micAutoSend'));
  let micPressedAt = 0;
  /** Speaker (DESK-10): read new answers aloud; which answer is playing now. */
  let autoRead = $state(readFlag('hypnos.desktop.autoRead'));
  let speakingId = $state(null);
  let speechError = $state(null);
  let speakTimer;
  /** Ticks while an image is being made, for the elapsed time. */
  let now = $state(Date.now());
  let imageNotice = $state(null);
  const streaming = new Set();

  let chat = $derived(store.current);
  let running = $derived(chat ? store.isRunning(chat.id) : false);
  let modelKnown = $derived(!chat?.model || models.some((m) => m.id === chat.model));
  let preparing = $derived(pending.some((p) => p.status === 'preparing'));
  let ready = $derived(pending.filter((p) => p.status === 'ready').map((p) => p.attachment));
  let canSend = $derived(
    (!!input.trim() || ready.length > 0) && !preparing && !!chat?.model && serverState === 'ready' && !running,
  );

  const STEPS = { image: 'resizing', document: 'converting', audio: 'transcribing', video: 'reading' };

  function stepFor(name) {
    const ext = name.split('.').pop().toLowerCase();
    if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tif', 'tiff'].includes(ext)) return STEPS.image;
    if (['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'oga'].includes(ext)) return STEPS.audio;
    if (['mp4', 'mov', 'm4v', 'webm', 'mkv'].includes(ext)) return 'transcribing the soundtrack';
    return STEPS.document;
  }

  /** Prepares a file (path) or pasted bytes in the background, as a chip above the composer. */
  async function attach(name, prepare, extra = {}) {
    const item = { id: crypto.randomUUID(), name, status: 'preparing', step: extra.url ? 'reading the page' : stepFor(name), ...extra };
    pending.push(item);
    const entry = () => pending.find((p) => p.id === item.id);
    try {
      let attachment = await prepare();
      if (attachment.kind === 'video' && attachment.path) {
        if (entry()) entry().step = 'sampling frames';
        try {
          attachment = { ...attachment, frames: await sampleFrames(attachment.path) };
        } catch (e) {
          console.warn(e);
        }
        if (!attachment.frames.length && !attachment.transcript) throw new Error(`${name}: neither its frames nor its sound could be read here.`);
      }
      if (entry()) Object.assign(entry(), { status: 'ready', attachment });
    } catch (e) {
      if (entry()) Object.assign(entry(), { status: 'error', error: api.errorMessage(e) });
    }
  }

  function attachPaths(paths) {
    for (const path of paths) attach(path.split(/[\\/]/).pop(), () => api.attachFromPath(path, chat?.model));
  }

  async function pickFiles() {
    const picked = await openDialog({
      multiple: true,
      filters: [
        { name: 'Images, documents, audio, video', extensions: ['png', 'jpg', 'jpeg', 'webp', 'gif', 'pdf', 'docx', 'pptx', 'xlsx', 'html', 'md', 'txt', 'csv', 'json', 'wav', 'mp3', 'm4a', 'flac', 'ogg', 'mp4', 'mov', 'm4v', 'webm', 'mkv'] },
        { name: 'All files', extensions: ['*'] },
      ],
    });
    if (picked) attachPaths(Array.isArray(picked) ? picked : [picked]);
  }

  async function onPaste(e) {
    const files = [...(e.clipboardData?.files ?? [])];
    if (!files.length) return;
    e.preventDefault();
    for (const file of files) {
      const name = file.name && file.name !== 'image.png' ? file.name : `pasted-${Date.now()}.${(file.type.split('/')[1] || 'png').replace('jpeg', 'jpg')}`;
      attach(name, async () => api.attachFromBytes(name, new Uint8Array(await file.arrayBuffer()), chat?.model));
    }
  }

  // Links typed in the composer can be read into the message (DESK-8).
  const URL_RE = /https?:\/\/[^\s<>"'`)\]]+/g;
  let links = $derived(
    [...new Set((input.match(URL_RE) ?? []).map((u) => u.replace(/[.,;:!?]+$/, '')))].filter(
      (u) => !pending.some((p) => p.url === u),
    ),
  );

  function readLink(url) {
    let label = url;
    try {
      const u = new URL(url);
      label = u.host + (u.pathname.length > 1 ? u.pathname : '');
    } catch {}
    const item = { url };
    attach(label, () => api.readUrl(url), item);
  }

  function readFlag(key) {
    try {
      return localStorage.getItem(key) === 'true';
    } catch {
      return false;
    }
  }

  let imaging = $derived(chat?.messages.some((m) => m.tool) ?? false);
  $effect(() => {
    if (!imaging) return;
    const t = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(t);
  });

  /** Where the image model sits now, to say whether it has to load (and may unload others). */
  let imageModel = $state(null);
  $effect(() => {
    if (!imaging) return;
    api.route('image_generation').then(async (id) => {
      const all = await api.models();
      imageModel = all.find((m) => m.id === id) ?? null;
    }).catch(() => (imageModel = null));
  });

  function imageProgress(tool) {
    const secs = Math.max(0, Math.round((now - tool.startedAt) / 1000));
    let where = '';
    if (imageModel && imageModel.current_tier !== 'tier1_gpu_vram') {
      where = ` · loading ${imageModel.id} from ${tierName(imageModel.current_tier)}, which may unload the chat model`;
    } else if (imageModel) {
      where = ` · ${imageModel.id}`;
    }
    return `Generating the image… ${secs} s${where}`;
  }

  function imageButton() {
    const prompt = input.trim();
    if (!prompt || !chat) return;
    input = '';
    tick().then(autosize);
    store.generateImage(chat, prompt);
  }

  async function saveImage(image) {
    const name = image.prompt.replace(/[^\w\s-]+/g, '').trim().split(/\s+/).slice(0, 6).join('-') || 'image';
    const path = await saveDialog({ defaultPath: `${name}.jpg`, filters: [{ name: 'JPEG image', extensions: ['jpg', 'jpeg'] }] });
    if (!path) return;
    try {
      await api.saveDataUrl(path, image.dataUrl);
      imageNotice = `Saved to ${path}.`;
    } catch (e) {
      imageNotice = api.errorMessage(e);
    }
  }

  async function copyImage(image) {
    try {
      const img = new Image();
      img.src = image.dataUrl;
      await img.decode();
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      imageNotice = 'Image copied.';
    } catch (e) {
      imageNotice = `Could not copy the image: ${e?.message ?? e}`;
    }
  }

  function answerId(i) {
    return `${chat.id}:${i}`;
  }

  function toggleAutoRead() {
    autoRead = !autoRead;
    try {
      localStorage.setItem('hypnos.desktop.autoRead', String(autoRead));
    } catch {}
    if (!autoRead) api.speakStop();
  }

  function readAloud(i) {
    const id = answerId(i);
    speechError = null;
    if (speakingId === id) return api.speakStop();
    api.speak(id, chat.messages[i].content, true, api.voicePrefs()).catch((e) => (speechError = api.errorMessage(e)));
  }

  // With the speaker on, the newest answer is read while it streams (a few calls a second) and
  // once more when it is complete.
  $effect(() => {
    if (!autoRead || !chat) return;
    const i = chat.messages.length - 1;
    const last = chat.messages[i];
    if (!last || last.role !== 'assistant' || !last.content) return;
    const text = last.content;
    const done = !last.pending;
    const id = answerId(i);
    // Only answers seen streaming in this window are read, never older ones of a chat reopened.
    if (!done) streaming.add(id);
    else if (!streaming.has(id)) return;
    clearTimeout(speakTimer);
    speakTimer = setTimeout(
      () => {
        api.speak(id, text, done, api.voicePrefs()).catch((e) => (speechError = api.errorMessage(e)));
        if (done) streaming.delete(id);
      },
      done ? 0 : 250,
    );
  });

  // A different chat stops the reading.
  $effect(() => {
    chat?.id;
    return () => api.speakStop();
  });

  function setAutoSend(on) {
    micAutoSend = on;
    try {
      localStorage.setItem('hypnos.desktop.micAutoSend', String(on));
    } catch {}
  }

  async function micStartRecording() {
    micError = null;
    micLevel = 0;
    micSeconds = 0;
    try {
      await api.micStart();
      mic = 'recording';
    } catch (e) {
      micError = api.errorMessage(e);
    }
  }

  async function micFinish() {
    if (mic !== 'recording') return;
    mic = 'transcribing';
    try {
      const { text } = await api.micStop(chat?.model);
      if (text) {
        input = input.trim() ? `${input.trimEnd()} ${text}` : text;
        await tick();
        autosize();
        if (micAutoSend) submit();
        else textarea?.focus();
      } else {
        micError = 'Nothing was heard.';
      }
    } catch (e) {
      micError = api.errorMessage(e);
    } finally {
      mic = 'idle';
    }
  }

  async function micDiscard() {
    if (mic !== 'recording') return;
    mic = 'idle';
    await api.micCancel();
  }

  // Click to start and click to stop, or hold and release (push to talk).
  function onMicDown(e) {
    e.preventDefault();
    if (mic === 'idle') {
      micPressedAt = Date.now();
      micStartRecording();
    } else if (mic === 'recording') {
      micPressedAt = 0;
      micFinish();
    }
  }

  function onMicUp() {
    if (mic === 'recording' && micPressedAt && Date.now() - micPressedAt > 400) micFinish();
    micPressedAt = 0;
  }

  function micClock(seconds) {
    const s = Math.floor(seconds);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }

  function removePending(i) {
    pending.splice(i, 1);
  }

  onMount(() => {
    const offLevel = listen('mic-level', (e) => {
      micLevel = Math.min(1, e.payload.rms * 6);
      micSeconds = e.payload.seconds;
    });
    const offLimit = listen('mic-limit', () => micFinish());
    const offSpeech = listen('speech-state', (e) => {
      const { id, playing, error } = e.payload;
      if (playing) speakingId = id;
      else if (speakingId === id) speakingId = null;
      if (error) speechError = error;
    });
    const onEscape = (e) => e.key === 'Escape' && mic === 'recording' && micDiscard();
    window.addEventListener('keydown', onEscape);
    let unlisten;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const { type } = event.payload;
        if (type === 'enter' || type === 'over') dragging = true;
        else if (type === 'leave') dragging = false;
        else if (type === 'drop') {
          dragging = false;
          attachPaths(event.payload.paths);
        }
      })
      .then((u) => (unlisten = u));
    return () => {
      unlisten?.();
      offLevel.then((u) => u());
      offLimit.then((u) => u());
      offSpeech.then((u) => u());
      window.removeEventListener('keydown', onEscape);
      if (mic === 'recording') api.micCancel();
    };
  });

  // A chat without a model (or with one that left the server) takes the first available.
  $effect(() => {
    if (chat && models.length && (!chat.model || !modelKnown) && chat.messages.length === 0) {
      store.setModel(chat, models.find((m) => m.id === store.lastModel)?.id ?? models[0].id);
    }
  });

  // Follow the stream while the user is at the bottom.
  $effect(() => {
    const last = chat?.messages.at(-1);
    last?.content;
    chat?.messages.length;
    if (stickToBottom && scroller) tick().then(() => scroller.scrollTo({ top: scroller.scrollHeight }));
  });

  $effect(() => {
    chat?.id;
    stickToBottom = true;
    textarea?.focus();
  });

  function onScroll() {
    stickToBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 40;
  }

  async function submit() {
    if (!canSend) return;
    const text = input;
    const attachments = ready.map((a) => $state.snapshot(a));
    if (speakingId) api.speakStop();
    input = '';
    pending = [];
    stickToBottom = true;
    await tick();
    autosize();
    store.send(chat, text, attachments);
  }

  function onKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  function autosize() {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 240)}px`;
  }

  function modelTier(id) {
    return models.find((m) => m.id === id)?.current_tier;
  }

  function copy(text) {
    navigator.clipboard.writeText(text);
  }
</script>

<div class="relative h-full flex flex-col">
  {#if dragging}
    <div class="absolute inset-2 z-20 rounded-2xl border-2 border-dashed border-indigo-400/70 bg-slate-950/80 flex items-center justify-center pointer-events-none">
      <p class="text-sm text-indigo-200">Drop images, documents, audio or video</p>
    </div>
  {/if}
  <div class="h-11 shrink-0 px-4 flex items-center gap-2 border-b border-slate-800/60">
    {#if chat}
      <select
        class="bg-transparent text-sm font-medium text-slate-200 rounded-md px-2 py-1 hover:bg-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        value={chat.model ?? ''}
        onchange={(e) => store.setModel(chat, e.currentTarget.value)}
        disabled={running}
        title="Model"
      >
        {#if !modelKnown}
          <option value={chat.model}>{chat.model} (unavailable)</option>
        {/if}
        {#if models.length === 0}
          <option value="" disabled>No chat models</option>
        {/if}
        {#each models as m (m.id)}
          <option value={m.id}>{m.id}</option>
        {/each}
      </select>
      {#if chat.model && modelTier(chat.model)}
        <span class="text-[11px] text-slate-500">{tierName(modelTier(chat.model))}</span>
      {/if}
      <button
        class="ml-auto p-1.5 rounded-md hover:bg-slate-800 {autoRead ? 'text-indigo-300' : 'text-slate-400'}"
        onclick={toggleAutoRead}
        title={autoRead ? 'Stop reading answers aloud' : 'Read answers aloud'}
        aria-label="Read answers aloud"
        aria-pressed={autoRead}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M11 5L6 9H3v6h3l5 4V5z" />
          {#if autoRead}<path d="M15.5 8.5a5 5 0 010 7M18.5 5.5a9 9 0 010 13" />{:else}<path d="M17 9l4 6M21 9l-4 6" />{/if}
        </svg>
      </button>
      <button
        class="text-xs px-2 py-1 rounded-md hover:bg-slate-800 {chat.system ? 'text-indigo-300' : 'text-slate-400'}"
        onclick={() => (showSystem = !showSystem)}
        aria-expanded={showSystem}
      >
        System prompt{chat.system ? ' •' : ''}
      </button>
    {/if}
  </div>

  {#if chat && showSystem}
    <div class="px-4 py-3 border-b border-slate-800/60 bg-slate-900/40">
      <textarea
        class="w-full max-w-3xl mx-auto block bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm resize-y min-h-16 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        placeholder="Instructions sent before every message of this chat"
        value={chat.system}
        oninput={(e) => store.setSystem(chat, e.currentTarget.value)}
      ></textarea>
    </div>
  {/if}

  <div class="flex-1 min-h-0 overflow-y-auto" bind:this={scroller} onscroll={onScroll}>
    {#if !chat || chat.messages.length === 0}
      <div class="h-full flex flex-col items-center justify-center gap-3 text-center px-6">
        <div class="w-14 h-14 rounded-2xl bg-white flex items-center justify-center"><img src={logo} alt="Hypnos" class="w-10 h-10" /></div>
        <p class="text-lg text-slate-300">How can I help?</p>
        {#if serverState === 'starting'}
          <p class="text-sm text-slate-500">Hypnos is loading its models…</p>
        {:else if serverState === 'unreachable' || serverState === 'unauthorized'}
          <p class="text-sm text-slate-500">
            {serverState === 'unauthorized' ? 'The server needs a valid API key.' : 'Cannot reach the Hypnos server. Retrying…'}
            <button class="text-indigo-300 hover:text-indigo-200" onclick={onsettings}>Connection settings</button>
            {#if onlocalserver && serverState === 'unreachable'}
              · <button class="text-indigo-300 hover:text-indigo-200" onclick={onlocalserver}>Start a local server</button>
            {/if}
          </p>
        {:else if serverState !== 'ready'}
          <p class="text-sm text-slate-500">Connecting to Hypnos…</p>
        {:else if models.length === 0}
          <p class="text-sm text-slate-500">The server has no chat model enabled.</p>
        {/if}
      </div>
    {:else}
      <div class="max-w-3xl mx-auto px-4 py-6 space-y-6">
        {#each chat.messages as message, i (i)}
          {#if message.role === 'user'}
            <div class="flex flex-col items-end gap-2">
              {#if message.attachments?.length}
                <div class="max-w-[80%]"><Attachments items={message.attachments} /></div>
              {/if}
              {#if message.content}
                <div class="max-w-[80%] rounded-2xl bg-slate-800 px-4 py-2 text-sm whitespace-pre-wrap break-words">{message.content}</div>
              {/if}
            </div>
          {:else}
            <div class="group">
              {#if message.image}
                <figure class="mb-3 space-y-1.5">
                  <img src={message.image.dataUrl} alt={message.image.prompt} class="max-w-md w-full rounded-xl border border-slate-800" />
                  <figcaption class="flex flex-wrap items-center gap-3 text-[11px] text-slate-500">
                    <span class="max-w-md truncate" title={message.image.prompt}>{message.image.prompt}</span>
                    <span>{message.image.model} · {Math.round(message.image.seconds)} s</span>
                    <button class="hover:text-slate-200" onclick={() => saveImage(message.image)}>Save</button>
                    <button class="hover:text-slate-200" onclick={() => copyImage(message.image)}>Copy</button>
                    <button class="hover:text-slate-200 disabled:opacity-40" onclick={() => store.regenerateImage(chat, message)} disabled={running || serverState !== 'ready'}>Regenerate</button>
                  </figcaption>
                </figure>
                {#if imageNotice}<p class="-mt-2 mb-3 text-[11px] text-slate-400">{imageNotice}</p>{/if}
              {/if}
              {#if message.tool}
                <div class="mb-2 flex items-center gap-2 text-sm text-slate-400">
                  <span class="w-3.5 h-3.5 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin"></span>
                  <span>{imageProgress(message.tool)}</span>
                </div>
                <p class="mb-2 text-[11px] text-slate-500 max-w-md">“{message.tool.prompt}”</p>
              {/if}
              {#if message.content}
                <Markdown text={message.content} />
              {:else if message.pending && !message.tool}
                <div class="flex items-center gap-2 text-sm text-slate-500">
                  <span class="flex gap-1">
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce"></span>
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce [animation-delay:150ms]"></span>
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce [animation-delay:300ms]"></span>
                  </span>
                  {#if modelTier(message.model) && modelTier(message.model) !== 'tier1_gpu_vram'}
                    Loading {message.model} from {tierName(modelTier(message.model))}…
                  {/if}
                </div>
              {/if}
              {#if message.error}
                <div class="mt-2 text-sm text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg px-3 py-2">{message.error}</div>
              {/if}
              {#if !message.pending}
                <div class="mt-1.5 flex items-center gap-3 text-[11px] text-slate-500">
                  {#if message.stopped}<span>Stopped</span>{/if}
                  {#if message.note}<span class="text-amber-300/80">{message.note}</span>{/if}
                  {#if speakingId === answerId(i)}<span class="text-indigo-300">Reading aloud…</span>{/if}
                  {#if message.telemetry}<span>{telemetrySummary(message.telemetry, message.usage)}</span>{/if}
                  <span class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    {#if message.content}
                      <button class="hover:text-slate-200" onclick={() => copy(message.content)}>Copy</button>
                      {#if message.requestId && onlog}
                        <button class="hover:text-slate-200" onclick={() => onlog(message.requestId)} title="This answer in the request log">Log</button>
                      {/if}
                      <button class="hover:text-slate-200" onclick={() => readAloud(i)}>
                        {speakingId === answerId(i) ? 'Stop reading' : 'Read aloud'}
                      </button>
                    {/if}
                    {#if i === chat.messages.length - 1}
                      <button class="hover:text-slate-200 disabled:opacity-40" onclick={() => store.regenerate(chat)} disabled={serverState !== 'ready'}>
                        Regenerate
                      </button>
                    {/if}
                  </span>
                </div>
              {/if}
            </div>
          {/if}
        {/each}
      </div>
    {/if}
  </div>

  <div class="shrink-0 px-4 pb-4">
    {#if pending.length}
      <div class="max-w-3xl mx-auto mb-2"><Attachments items={pending} onremove={removePending} /></div>
    {/if}
    {#if speechError}
      <div class="max-w-3xl mx-auto mb-2 flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
        <span class="flex-1">{speechError}</span>
        <button class="text-slate-400 hover:text-slate-200" onclick={() => (speechError = null)} aria-label="Dismiss">×</button>
      </div>
    {/if}
    {#if mic !== 'idle' || micError}
      <div class="max-w-3xl mx-auto mb-2 flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-xs">
        {#if mic === 'recording'}
          <span class="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse"></span>
          <span class="font-mono text-slate-300">{micClock(micSeconds)}</span>
          <span class="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <span class="block h-full bg-rose-400 transition-[width] duration-100" style="width: {Math.round(micLevel * 100)}%"></span>
          </span>
          <label class="flex items-center gap-1.5 text-slate-400">
            <input type="checkbox" checked={micAutoSend} onchange={(e) => setAutoSend(e.currentTarget.checked)} class="accent-indigo-500" />
            send when done
          </label>
          <button class="text-slate-400 hover:text-slate-100" onclick={micDiscard} title="Discard (Esc)">Cancel</button>
          <button class="text-rose-300 hover:text-rose-200" onclick={micFinish}>Done</button>
        {:else if mic === 'transcribing'}
          <span class="w-3 h-3 rounded-full border-2 border-slate-500 border-t-transparent animate-spin"></span>
          <span class="text-slate-400">Transcribing…</span>
        {:else}
          <span class="flex-1 text-rose-300">{micError}</span>
          <button class="text-slate-500 hover:text-slate-200" onclick={() => (micError = null)} aria-label="Dismiss">×</button>
        {/if}
      </div>
    {/if}
    {#if links.length}
      <div class="max-w-3xl mx-auto mb-2 flex flex-wrap gap-2">
        {#each links as url (url)}
          <button
            class="max-w-80 truncate text-[11px] px-2 py-1 rounded-full border border-slate-700 text-slate-300 hover:border-indigo-400 hover:text-indigo-200"
            onclick={() => readLink(url)}
            title="Fetch this page and add its text to the message"
          >
            Read {url.replace(/^https?:\/\//, '')}
          </button>
        {/each}
      </div>
    {/if}
    <div class="max-w-3xl mx-auto flex items-end gap-2 rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 focus-within:border-slate-500">
      <button
        class="w-8 h-8 shrink-0 rounded-full text-slate-400 hover:text-slate-100 hover:bg-slate-800 flex items-center justify-center"
        onclick={pickFiles}
        title="Attach images, documents, audio or video (or drop / paste them)"
        aria-label="Attach files"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5l-8.6 8.6a5 5 0 01-7.1-7.1l8.6-8.6a3.5 3.5 0 015 5l-8.6 8.6a2 2 0 01-2.8-2.8l7.9-7.9" /></svg>
      </button>
      <button
        class="w-8 h-8 shrink-0 rounded-full text-slate-400 hover:text-slate-100 hover:bg-slate-800 flex items-center justify-center disabled:opacity-30"
        onclick={imageButton}
        disabled={!input.trim() || running || serverState !== 'ready'}
        title="Generate an image from this text (or type /image …)"
        aria-label="Generate an image"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <circle cx="9" cy="10" r="2" />
          <path d="M21 16l-5-5-9 9" />
        </svg>
      </button>
      <textarea
        onpaste={onPaste}
        bind:this={textarea}
        bind:value={input}
        oninput={autosize}
        onkeydown={onKeydown}
        rows="1"
        class="flex-1 bg-transparent resize-none text-sm py-1.5 focus:outline-none placeholder:text-slate-500"
        placeholder={chat?.model ? `Message ${chat.model}` : 'Send a message'}
      ></textarea>
      <button
        class="w-8 h-8 shrink-0 rounded-full flex items-center justify-center disabled:opacity-30
          {mic === 'recording' ? 'bg-rose-500 text-white' : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'}"
        onpointerdown={onMicDown}
        onpointerup={onMicUp}
        disabled={mic === 'transcribing' || serverState !== 'ready'}
        title={mic === 'recording' ? 'Stop and transcribe' : 'Speak (click, or hold to talk)'}
        aria-label={mic === 'recording' ? 'Stop recording' : 'Record a voice message'}
        aria-pressed={mic === 'recording'}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="3" width="6" height="11" rx="3" />
          <path d="M5 11a7 7 0 0014 0M12 18v3" />
        </svg>
      </button>
      {#if running}
        <button
          class="w-8 h-8 shrink-0 rounded-full bg-slate-200 text-slate-900 flex items-center justify-center hover:bg-white"
          onclick={() => store.stop(chat.id)}
          title="Stop"
          aria-label="Stop"
        >
          <span class="w-3 h-3 rounded-sm bg-slate-900"></span>
        </button>
      {:else}
        <button
          class="w-8 h-8 shrink-0 rounded-full bg-slate-200 text-slate-900 flex items-center justify-center hover:bg-white disabled:opacity-30 disabled:hover:bg-slate-200"
          onclick={submit}
          disabled={!canSend}
          title="Send (Enter)"
          aria-label="Send"
        >
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
        </button>
      {/if}
    </div>
  </div>
</div>
