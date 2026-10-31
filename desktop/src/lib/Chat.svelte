<script>
  import { tick } from 'svelte';
  import Markdown from './Markdown.svelte';
  import { telemetrySummary, tierName } from './format.js';

  let { store, models, serverState } = $props();

  let input = $state('');
  let showSystem = $state(false);
  let scroller = $state();
  let textarea = $state();
  let stickToBottom = true;

  let chat = $derived(store.current);
  let running = $derived(chat ? store.isRunning(chat.id) : false);
  let modelKnown = $derived(!chat?.model || models.some((m) => m.id === chat.model));
  let canSend = $derived(!!input.trim() && !!chat?.model && serverState === 'ready' && !running);

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
    input = '';
    stickToBottom = true;
    await tick();
    autosize();
    store.send(chat, text);
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

<div class="h-full flex flex-col">
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
        <div class="w-12 h-12 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-xl shadow-lg shadow-indigo-600/30">H</div>
        <p class="text-lg text-slate-300">How can I help?</p>
        {#if serverState !== 'ready'}
          <p class="text-sm text-slate-500">
            {serverState === 'starting' ? 'Hypnos is loading its models…' : 'Waiting for the Hypnos server…'}
          </p>
        {:else if models.length === 0}
          <p class="text-sm text-slate-500">The server has no chat model enabled.</p>
        {/if}
      </div>
    {:else}
      <div class="max-w-3xl mx-auto px-4 py-6 space-y-6">
        {#each chat.messages as message, i (i)}
          {#if message.role === 'user'}
            <div class="flex justify-end">
              <div class="max-w-[80%] rounded-2xl bg-slate-800 px-4 py-2 text-sm whitespace-pre-wrap break-words">{message.content}</div>
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
    <div class="max-w-3xl mx-auto flex items-end gap-2 rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 focus-within:border-slate-500">
      <textarea
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
