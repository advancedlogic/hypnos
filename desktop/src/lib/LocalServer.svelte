<script>
  import { onMount, tick, untrack } from 'svelte';
  import * as server from './localServer.js';

  /** `settings`: `{ binary, config, port, autoStart }`; `running`: the process status. */
  let { settings, running, onstart, onstop, onsettings, onclose } = $props();

  // The form starts from the saved settings and is saved back on Start.
  const initial = untrack(() => ({ ...settings }));
  let binary = $state(initial.binary);
  let config = $state(initial.config);
  let port = $state(initial.port);
  let autoStart = $state(initial.autoStart);
  let busy = $state(false);
  let error = $state(null);
  let lines = $state([]);
  let logBox = $state();
  let follow = true;

  let isRunning = $derived(running?.running ?? false);

  function current() {
    return { binary: binary.trim(), config: config.trim(), port: Number(port), autoStart };
  }

  async function redetect() {
    let found;
    try {
      found = await server.detect();
    } catch (e) {
      error = String(e?.message ?? e);
      return;
    }
    if (found.binary) binary = found.binary;
    config = found.config ?? '';
    error = found.binary ? null : 'No hypnos binary found: build it (./run.sh build backend) or enter its path.';
  }

  async function start() {
    error = null;
    busy = true;
    try {
      await onsettings(current());
      await onstart(current());
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = false;
    }
  }

  async function stop() {
    busy = true;
    try {
      await onstop();
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = false;
    }
  }

  function onAutoStart() {
    onsettings(current());
  }

  function onScroll() {
    follow = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 30;
  }

  async function scrollDown() {
    await tick();
    if (follow && logBox) logBox.scrollTop = logBox.scrollHeight;
  }

  onMount(() => {
    let unsubscribe;
    server.logs().then((l) => {
      lines = l;
      scrollDown();
    });
    server
      .subscribe(
        (line) => {
          lines.push(line);
          if (lines.length > 2000) lines.splice(0, lines.length - 2000);
          scrollDown();
        },
        () => {},
      )
      .then((u) => (unsubscribe = u));
    return () => unsubscribe?.();
  });

  function onKeydown(e) {
    if (e.key === 'Escape') onclose();
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="fixed inset-0 z-30 bg-black/60 flex items-center justify-center p-4" role="presentation" onclick={(e) => e.target === e.currentTarget && onclose()}>
  <div role="dialog" aria-modal="true" aria-labelledby="local-title" class="w-full max-w-3xl max-h-full flex flex-col rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
    <div class="p-5 space-y-4 shrink-0">
      <div class="flex items-center gap-3">
        <h2 id="local-title" class="text-sm font-semibold">Local server</h2>
        <span class="flex items-center gap-1.5 text-xs text-slate-400">
          <span class="w-2 h-2 rounded-full {isRunning ? 'bg-emerald-400' : running?.exited ? 'bg-rose-500' : 'bg-slate-600'}"></span>
          {#if isRunning}
            running (pid {running.pid}, port {running.options?.port})
          {:else if running?.exited}
            exited{running.exitCode != null ? ` with code ${running.exitCode}` : ''}
          {:else}
            stopped
          {/if}
        </span>
        <button class="ml-auto text-slate-500 hover:text-slate-200" onclick={onclose} aria-label="Close">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18" /></svg>
        </button>
      </div>
      <p class="text-xs text-slate-500">
        Runs <code>hypnos</code> on this machine and connects the app to it: the server that comes with the app
        (macOS), or an installed one. The app stops the server it started when you quit. The bundled server's
        config and models live in <code>~/Library/Application Support/Hypnos</code>; models whose files are not
        there yet are disabled until they are.
      </p>

      <div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 items-center text-sm">
        <label for="ls-binary" class="text-xs text-slate-400">Binary</label>
        <div class="flex gap-2">
          <input id="ls-binary" bind:value={binary} disabled={isRunning} spellcheck="false" placeholder="/path/to/hypnos"
            class="flex-1 min-w-0 rounded-lg bg-slate-950 border border-slate-700 px-3 py-1.5 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60" />
          <button class="px-2 text-xs text-slate-400 hover:text-slate-200 disabled:opacity-40" onclick={redetect} disabled={isRunning}>Detect</button>
        </div>
        <label for="ls-config" class="text-xs text-slate-400">Config</label>
        <input id="ls-config" bind:value={config} disabled={isRunning} spellcheck="false" placeholder="hypnos.toml beside the binary"
          class="rounded-lg bg-slate-950 border border-slate-700 px-3 py-1.5 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60" />
        <label for="ls-port" class="text-xs text-slate-400">Port</label>
        <div class="flex items-center gap-4">
          <input id="ls-port" type="number" min="1" max="65535" bind:value={port} disabled={isRunning}
            class="w-24 rounded-lg bg-slate-950 border border-slate-700 px-3 py-1.5 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60" />
          <label class="flex items-center gap-2 text-xs text-slate-400">
            <input type="checkbox" bind:checked={autoStart} onchange={onAutoStart} class="accent-indigo-500" />
            Start with the app
          </label>
        </div>
      </div>

      {#if error}
        <p class="text-xs text-rose-300">{error}</p>
      {/if}

      <div class="flex gap-2">
        {#if isRunning}
          <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800 disabled:opacity-50" onclick={stop} disabled={busy}>
            {busy ? 'Stopping…' : 'Stop'}
          </button>
        {:else}
          <button class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50" onclick={start} disabled={busy || !binary.trim()}>
            {busy ? 'Starting…' : 'Start'}
          </button>
        {/if}
      </div>
    </div>

    <div
      bind:this={logBox}
      onscroll={onScroll}
      class="min-h-48 h-80 overflow-auto border-t border-slate-800 bg-slate-950 px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-400 whitespace-pre"
    >
      {#if lines.length === 0}
        <span class="text-slate-600">The server's output appears here.</span>
      {:else}
        {lines.join('\n')}
      {/if}
    </div>
  </div>
</div>
