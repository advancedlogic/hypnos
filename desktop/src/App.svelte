<script>
  import { onMount } from 'svelte';
  import { createClient, probe, DEFAULT_SERVER_URL } from './lib/hypnos.js';
  import { ChatStore } from './lib/chats.svelte.js';
  import Sidebar from './lib/Sidebar.svelte';
  import Chat from './lib/Chat.svelte';
  import StatsPanel from './lib/StatsPanel.svelte';

  const POLL_MS = 3000;
  const MODELS_MS = 10000;
  const PANEL_KEY = 'hypnos.desktop.panelOpen';

  const serverUrl = DEFAULT_SERVER_URL;
  const client = createClient({ baseUrl: serverUrl });

  const store = new ChatStore(client);

  let status = $state({ state: 'connecting', health: null });
  let models = $state([]);
  let panelOpen = $state(readPanelOpen());

  const STATES = {
    connecting: { label: 'Connecting', dot: 'bg-slate-500' },
    unreachable: { label: 'Unreachable', dot: 'bg-rose-500' },
    starting: { label: 'Warming up', dot: 'bg-amber-400 animate-pulse' },
    ready: { label: 'Ready', dot: 'bg-emerald-400' },
  };
  // Derived so the effects below run on a state change, not on every poll.
  let serverState = $derived(status.state);
  let shown = $derived(STATES[serverState]);

  function readPanelOpen() {
    try {
      return localStorage.getItem(PANEL_KEY) !== 'false';
    } catch {
      return true;
    }
  }

  function togglePanel() {
    panelOpen = !panelOpen;
    try {
      localStorage.setItem(PANEL_KEY, String(panelOpen));
    } catch {}
  }

  function onKeydown(e) {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (e.key === '.') {
      e.preventDefault();
      togglePanel();
    } else if (e.key === 'n') {
      e.preventDefault();
      store.newChat();
    }
  }

  /** The enabled models that answer `/v1/chat/completions`, with their current tier. */
  async function loadModels(signal) {
    const [list, memory] = await Promise.all([client.models({ signal }), client.memoryStatus({ signal })]);
    const modalities = new Map(memory.models.map((m) => [m.id, m.modalities ?? [m.modality]]));
    models = list.data.filter((m) => (modalities.get(m.id) ?? [m.modality]).includes('text_generation'));
  }

  // The model list follows the server: loaded once it is ready, then refreshed for the tiers.
  $effect(() => {
    if (serverState !== 'ready') return;
    const controller = new AbortController();
    let timer;
    const refresh = async () => {
      try {
        await loadModels(controller.signal);
      } catch {
        if (controller.signal.aborted) return;
      }
      timer = setTimeout(refresh, MODELS_MS);
    };
    refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  });

  onMount(() => {
    store.init().then(() => store.current ?? store.newChat());
    const controller = new AbortController();
    let timer;
    const tick = async () => {
      try {
        status = await probe(client, controller.signal);
      } catch {
        return;
      }
      timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  });
</script>

<svelte:window onkeydown={onKeydown} />

<div class="h-full flex flex-col">
  <header class="h-12 shrink-0 border-b border-slate-800 bg-slate-900/70 px-4 flex items-center justify-between gap-4">
    <div class="flex items-center gap-3">
      <div class="w-7 h-7 rounded-md bg-indigo-600 flex items-center justify-center font-bold text-sm shadow-lg shadow-indigo-600/30">
        H
      </div>
      <h1 class="font-semibold text-sm tracking-tight">Hypnos</h1>
    </div>

    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2 text-xs text-slate-400" title={serverUrl}>
        <span class="w-2 h-2 rounded-full {shown.dot}"></span>
        <span>{shown.label}</span>
        {#if status.health}
          <span class="text-slate-600">v{status.health.version}</span>
        {/if}
      </div>
      <button
        class="p-1.5 rounded-md text-slate-400 hover:text-slate-100 hover:bg-slate-800"
        onclick={togglePanel}
        title="{panelOpen ? 'Hide' : 'Show'} stats (⌘/Ctrl + .)"
        aria-label="Toggle stats panel"
        aria-pressed={panelOpen}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M15 4v16" />
        </svg>
      </button>
    </div>
  </header>

  <div class="flex-1 min-h-0 flex">
    <Sidebar {store} />

    <main class="flex-1 min-w-0">
      <Chat {store} {models} {serverState} />
    </main>

    {#if panelOpen}
      <aside class="w-80 shrink-0 border-l border-slate-800 bg-slate-900/40 overflow-y-auto">
        <StatsPanel {client} {status} lastResponse={store.lastTelemetry} />
      </aside>
    {/if}
  </div>
</div>
