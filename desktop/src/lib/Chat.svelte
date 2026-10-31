<script>
  import { onMount, tick } from 'svelte';
  import { open as openDialog } from '@tauri-apps/plugin-dialog';
  import { getCurrentWebview } from '@tauri-apps/api/webview';
  import Markdown from './Markdown.svelte';
  import Attachments from './Attachments.svelte';
  import * as api from './hypnos.js';
  import { sampleFrames } from './video.js';
  import logo from '../assets/logo.png';
  import { telemetrySummary, tierName } from './format.js';

  let { store, models, serverState, onsettings, onlocalserver } = $props();

  let input = $state('');
  let showSystem = $state(false);
  let scroller = $state();
  let textarea = $state();
  let stickToBottom = true;
  /** Files being prepared or ready for the next message: `{ id, name, status, attachment?, error? }`. */
  let pending = $state([]);
  let dragging = $state(false);

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
  async function attach(name, prepare) {
    const item = { id: crypto.randomUUID(), name, status: 'preparing', step: stepFor(name) };
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

  function removePending(i) {
    pending.splice(i, 1);
  }

  onMount(() => {
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
    return () => unlisten?.();
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
        class="ml-auto text-xs px-2 py-1 rounded-md hover:bg-slate-800 {chat.system ? 'text-indigo-300' : 'text-slate-400'}"
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
              {#if message.content}
                <Markdown text={message.content} />
              {:else if message.pending}
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
                  {#if message.telemetry}<span>{telemetrySummary(message.telemetry, message.usage)}</span>{/if}
                  <span class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    {#if message.content}
                      <button class="hover:text-slate-200" onclick={() => copy(message.content)}>Copy</button>
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
    <div class="max-w-3xl mx-auto flex items-end gap-2 rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 focus-within:border-slate-500">
      <button
        class="w-8 h-8 shrink-0 rounded-full text-slate-400 hover:text-slate-100 hover:bg-slate-800 flex items-center justify-center"
        onclick={pickFiles}
        title="Attach images, documents, audio or video (or drop / paste them)"
        aria-label="Attach files"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5l-8.6 8.6a5 5 0 01-7.1-7.1l8.6-8.6a3.5 3.5 0 015 5l-8.6 8.6a2 2 0 01-2.8-2.8l7.9-7.9" /></svg>
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
