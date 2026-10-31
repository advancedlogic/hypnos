<script>
  import { onMount } from 'svelte';
  import * as api from './lib/hypnos.js';
  import { load, save } from './lib/storage.js';
  import { connectionFor, loadConnection, saveConnection } from './lib/connection.js';
  import * as localServer from './lib/localServer.js';
  import LocalServer from './lib/LocalServer.svelte';
  import Models from './lib/Models.svelte';
  import * as modelFiles from './lib/models.js';
  import { ChatStore } from './lib/chats.svelte.js';
  import Sidebar from './lib/Sidebar.svelte';
  import Chat from './lib/Chat.svelte';
  import StatsPanel from './lib/StatsPanel.svelte';
  import Settings from './lib/Settings.svelte';
  import Logs from './lib/Logs.svelte';
  import logo from './assets/logo.png';
  import { listen } from '@tauri-apps/api/event';
  import { invoke } from '@tauri-apps/api/core';
  import { applyAppPrefs, updateTray } from './lib/appPrefs.js';

  const POLL_MS = 3000;
  const MODELS_MS = 10000;
  const PANEL_KEY = 'hypnos.desktop.panelOpen';
  const ROUTES_KEY = 'routes';

  /** `{ url, apiKey }`, loaded from the store and the keychain at start. */
  let connection = $state(null);
  /** Bumped each time the Rust client points at a (new) server; the loops below restart on it. */
  let connected = $state(0);
  /** Modality -> model id chosen in the settings (absent = automatic). */
  let routes = $state({});

  const store = new ChatStore();

  let status = $state({ state: 'connecting', health: null });
  let panelOpen = $state(readPanelOpen());
  let settingsOpen = $state(false);
  /** The request log view (DESK-13) replaces the chat; `logFocus` is a record to open. */
  let logsOpen = $state(false);
  let logFocus = $state(null);

  function showLog(id = null) {
    logFocus = id;
    logsOpen = true;
  }
  let localOpen = $state(false);
  /** `{ binary, config, port, autoStart }` and the process status (DESK-5). */
  let localSettings = $state(null);
  let localStatus = $state(null);
  /** The Models dialog (DESK-17); `modelsFirstRun` adds the welcome line. */
  let modelsOpen = $state(false);
  /** Which tab the Models dialog opens on: `lineup` or `catalog`. */
  let modelsTab = $state('lineup');
  let modelsFirstRun = $state(false);
  const MODELS_PROMPTED_KEY = 'modelsPrompted';

  const STATES = {
    connecting: { label: 'Connecting', dot: 'bg-slate-500' },
    unreachable: { label: 'Unreachable', dot: 'bg-rose-500' },
    unauthorized: { label: 'API key refused', dot: 'bg-rose-500' },
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
    if (!(e.metaKey || e.ctrlKey) || settingsOpen || localOpen) return;
    if (e.key === '.') {
      e.preventDefault();
      togglePanel();
    } else if (e.key === 'n') {
      e.preventDefault();
      store.newChat();
    } else if (e.key === ',') {
      e.preventDefault();
      settingsOpen = true;
    } else if (e.key === 'l') {
      e.preventDefault();
      if (logsOpen) logsOpen = false;
      else showLog();
    }
  }

  async function useServer(url) {
    if (connection?.url === url) return;
    const next = await connectionFor(url);
    await saveConnection(next);
    connection = next;
    connected++;
  }

  async function saveLocalSettings(next) {
    localSettings = next;
    await localServer.saveLocalSettings(next);
  }

  /** Starts the local server and points the app at it. */
  async function startLocal(opts) {
    localStatus = await localServer.start(opts);
    await useServer(localServer.localUrl(opts.port));
  }

  async function stopLocal() {
    localStatus = await localServer.stop();
  }

  /**
   * After model files arrived or their folder moved: restart the local server so it picks them
   * up (its start enables the models whose files are there), or start it if it is not running.
   */
  async function restartLocal() {
    if (!localSettings?.binary) return;
    try {
      if (localStatus?.running) await stopLocal();
      await startLocal(localSettings);
    } catch (e) {
      serverError = `Local server: ${e?.message ?? e}`;
    }
  }

  /** First launch with the bundled server and no models: offer the download, once. */
  async function promptForModels() {
    if (!localSettings?.config || (await load(MODELS_PROMPTED_KEY, false))) return;
    try {
      const cat = await modelFiles.catalog(localSettings.config);
      if (!cat.managed || cat.models.some((m) => m.installed)) return;
      await save(MODELS_PROMPTED_KEY, true);
      modelsFirstRun = true;
      modelsTab = 'lineup';
      modelsOpen = true;
    } catch (e) {
      console.warn('models catalog', e);
    }
  }

  async function initLocalServer() {
    localSettings = await localServer.loadLocalSettings();
    localStatus = await localServer.status();
    await localServer.subscribe(
      () => {},
      (s) => (localStatus = s),
    );
    promptForModels();
    const { binary, port, autoStart } = localSettings;
    if (!autoStart || !binary) return;
    const [up] = await localServer.probe(Number(port));
    if (up) return useServer(localServer.localUrl(port));
    try {
      await startLocal(localSettings);
    } catch (e) {
      console.warn('could not start the local server', e);
    }
  }

  async function onSaveSettings(next, nextRoutes) {
    await saveConnection(next);
    connection = next;
    connected++;
    routes = nextRoutes;
    await save(ROUTES_KEY, nextRoutes);
    await api.setRoutes(nextRoutes);
    settingsOpen = false;
  }

  /** Every enabled model (with `modalities` and tier); the chat picker shows the chat ones. */
  let allModels = $state([]);

  // A status loop per connection; it keeps going while the server is down, so the app recovers
  // by itself when it comes back. A cancelled loop drops the answer of its last probe.
  $effect(() => {
    if (!connected) return;
    status = { state: 'connecting', health: null };
    allModels = [];
    let live = true;
    let timer;
    const tick = async () => {
      try {
        const next = await api.probe();
        if (live) status = next;
      } catch (e) {
        console.warn(api.errorMessage(e));
      }
      if (live) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  });

  // The model list follows the server: loaded once it is ready, then refreshed for the tiers.
  $effect(() => {
    if (!connected || serverState !== 'ready') return;
    let live = true;
    let timer;
    const refresh = async () => {
      try {
        const next = await api.models();
        if (live) allModels = next;
      } catch (e) {
        console.warn(api.errorMessage(e));
      }
      if (live) timer = setTimeout(refresh, MODELS_MS);
    };
    refresh();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  });

  let models = $derived(allModels.filter((m) => m.modalities.includes('text_generation')));

  /** The port of the server the app uses when it runs on this machine, else null. */
  let localPort = $derived.by(() => {
    try {
      const u = new URL(connection?.url ?? '');
      if (!['localhost', '127.0.0.1', '[::1]'].includes(u.hostname)) return null;
      return Number(u.port || (u.protocol === 'https:' ? 443 : 80));
    } catch {
      return null;
    }
  });

  // The tray's server item follows the server the app is connected to, whoever started it.
  let serverItem = $derived.by(() => {
    if (localPort === null) return { text: 'Server runs on another machine', enabled: false };
    if (serverBusy) return { text: serverBusy, enabled: false };
    if (localStatus?.running || ['ready', 'starting', 'unauthorized'].includes(serverState)) {
      return { text: 'Stop the server', enabled: true, action: 'stop' };
    }
    if (serverState === 'connecting') return { text: 'Checking the server…', enabled: false };
    return { text: 'Start the server', enabled: !!localSettings?.binary, action: 'start' };
  });
  let serverBusy = $state(null);
  let serverError = $state(null);

  // The tray icon follows the server state (DESK-12).
  $effect(() => {
    const version = status.health ? ` · v${status.health.version}` : '';
    updateTray({
      state: serverState,
      label: `Hypnos: ${shown.label}${version}`,
      serverItem: serverItem.text,
      serverItemEnabled: serverItem.enabled,
    });
  });

  async function serverFromTray() {
    const { action } = serverItem;
    if (!action) return;
    serverError = null;
    serverBusy = action === 'stop' ? 'Stopping the server…' : 'Starting the server…';
    try {
      if (action === 'stop') {
        if (localStatus?.running && localStatus.options?.port === localPort) await stopLocal();
        else await invoke('local_server_stop_port', { port: localPort });
      } else {
        await startLocal({ ...localSettings, port: localPort });
      }
    } catch (e) {
      serverError = `Could not ${action} the server: ${e?.message ?? e}`;
      if (action === 'start') localOpen = true; // its log shows why
    } finally {
      serverBusy = null;
    }
  }

  onMount(async () => {
    listen('tray-open-settings', () => (settingsOpen = true));
    listen('tray-server', serverFromTray);
    applyAppPrefs().catch((e) => console.warn(e));
    connection = await loadConnection();
    routes = await load(ROUTES_KEY, {});
    await api.setRoutes(routes);
    await api.connect(connection.url);
    connected++;
    initLocalServer();
    await store.init();
    if (!store.current) store.newChat();
  });
</script>

<svelte:window onkeydown={onKeydown} />

<div class="h-full flex flex-col">
  <header class="h-12 shrink-0 border-b border-slate-800 bg-slate-900/70 px-4 flex items-center justify-between gap-4">
    <div class="flex items-center gap-3">
      <div class="w-7 h-7 rounded-md bg-white flex items-center justify-center">
        <img src={logo} alt="" class="w-5 h-5" />
      </div>
      <h1 class="font-semibold text-sm tracking-tight">Hypnos</h1>
    </div>

    <div class="flex items-center gap-3">
      <button
        class="flex items-center gap-2 text-xs text-slate-400 hover:text-slate-200 rounded-md px-2 py-1 hover:bg-slate-800"
        title="{connection?.url ?? ''}: connection settings (⌘/Ctrl + ,)"
        onclick={() => (settingsOpen = true)}
      >
        <span class="w-2 h-2 rounded-full {shown.dot}"></span>
        <span>{shown.label}</span>
        {#if status.health}
          <span class="text-slate-600">v{status.health.version}</span>
        {/if}
        {#if connection && !/^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(connection.url)}
          <span class="text-slate-500 font-mono">{new URL(connection.url).host}</span>
        {/if}
      </button>
      <button
        class="p-1.5 rounded-md hover:bg-slate-800 {logsOpen ? 'text-indigo-300' : 'text-slate-400 hover:text-slate-100'}"
        onclick={() => (logsOpen ? (logsOpen = false) : showLog())}
        title="Request log (⌘/Ctrl + L)"
        aria-label="Request log"
        aria-pressed={logsOpen}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
        </svg>
      </button>
      {#if localSettings?.config}
        <button
          class="flex items-center gap-1.5 text-xs rounded-md px-2 py-1 text-slate-400 hover:text-slate-100 hover:bg-slate-800"
          onclick={() => {
            modelsFirstRun = false;
            modelsTab = 'catalog';
            modelsOpen = true;
          }}
          title="Models: the lineup and the catalog of tested models to add"
          aria-label="Models"
        >
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" />
            <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
          </svg>
          <span>Models</span>
        </button>
      {/if}
      {#if localSettings}
        <button
          class="relative p-1.5 rounded-md text-slate-400 hover:text-slate-100 hover:bg-slate-800"
          onclick={() => (localOpen = true)}
          title="Local server{localStatus?.running ? ' (running)' : ''}"
          aria-label="Local server"
        >
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="4" width="18" height="7" rx="1.5" />
            <rect x="3" y="13" width="18" height="7" rx="1.5" />
            <path d="M7 7.5h.01M7 16.5h.01" />
          </svg>
          {#if localStatus?.running}
            <span class="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          {:else if localStatus?.exited}
            <span class="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          {/if}
        </button>
      {/if}
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

  {#if serverError}
    <div class="shrink-0 px-4 py-1.5 flex items-center gap-3 text-xs text-rose-300 bg-rose-500/10 border-b border-rose-500/30">
      <span class="flex-1">{serverError}</span>
      <button class="text-slate-400 hover:text-slate-200" onclick={() => (serverError = null)} aria-label="Dismiss">×</button>
    </div>
  {/if}

  <div class="flex-1 min-h-0 flex">
    <Sidebar {store} />

    <main class="flex-1 min-w-0">
      {#if logsOpen}
        {#key logFocus}
          <Logs focus={logFocus} onclose={() => (logsOpen = false)} />
        {/key}
      {:else}
      <Chat {store} {models} {serverState} onsettings={() => (settingsOpen = true)} onlog={showLog}
        onlocalserver={localSettings ? () => (localOpen = true) : null} />
      {/if}
    </main>

    {#if panelOpen && connected}
      <aside class="w-80 shrink-0 border-l border-slate-800 bg-slate-900/40 overflow-y-auto">
        {#key connected}
          <StatsPanel {status} lastResponse={store.lastTelemetry} />
        {/key}
      </aside>
    {/if}
  </div>
</div>

{#if localOpen && localSettings}
  <LocalServer
    settings={localSettings}
    running={localStatus}
    onstart={startLocal}
    onstop={stopLocal}
    onsettings={saveLocalSettings}
    onmodels={() => {
      localOpen = false;
      modelsFirstRun = false;
      modelsTab = 'lineup';
      modelsOpen = true;
    }}
    onclose={() => (localOpen = false)}
  />
{/if}

{#if modelsOpen && localSettings?.config}
  <Models config={localSettings.config} binary={localSettings.binary} firstRun={modelsFirstRun} tab={modelsTab} onchanged={restartLocal}
    onclose={() => (modelsOpen = false)} />
{/if}

{#if settingsOpen && connection}
  <Settings {connection} {routes} models={allModels} onsave={onSaveSettings} onclose={() => (settingsOpen = false)} />
{/if}
