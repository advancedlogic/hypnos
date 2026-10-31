<script>
  import { onMount, untrack } from 'svelte';
  import { open as openDialog } from '@tauri-apps/plugin-dialog';
  import * as models from './models.js';

  /**
   * The local server's model files (DESK-17): pick from the lineup, download with progress,
   * change the folder; add, replace and remove models of the curated catalog (API-10, API-11).
   * `config`: the local server's config path; `binary`: its server binary (renders a catalog
   * model's config block); `onchanged` restarts the server after files arrived, a model was
   * added to the config or the folder moved; `onpool` after the running server's pool changed
   * (no restart: it serves the change right away); `firstRun` shows the welcome line; `tab` opens
   * on `lineup` or `catalog`.
   */
  let { config, binary = null, firstRun = false, tab = 'lineup', onchanged, onpool, onclose } = $props();
  let view = $state(untrack(() => tab)); // 'lineup' | 'catalog'

  let cat = $state(null);
  let loadError = $state(null);
  let selected = $state(new Set());
  let busy = $state(null); // 'download' | 'move' | 'pool' | null
  let progress = $state(null); // { done, total, file, fileBytes, fileSize, verifying }
  let error = $state(null);
  let note = $state(null);
  let moveFiles = $state(true);
  let entries = $state(null); // the catalog's models, `inPool` or not
  let entriesError = $state(null);
  let live = $state(false); // the running server's pool, changed live
  let offline = $state(null); // why not
  let chosen = $state({}); // id -> quant
  let adding = $state(null); // { id, label } of the model being changed
  let confirming = $state(null); // id whose Remove waits for a second click
  let shownBlock = $state(null); // { id, quant, text } for a config the app does not write
  let copied = $state(false);

  const MODALITY = {
    text_generation: 'Chat',
    embeddings: 'Embeddings',
    speech_to_text: 'Speech to text',
    vision: 'Vision',
    text_to_speech: 'Speech',
    image_generation: 'Images',
    document_conversion: 'Documents',
    entity_extraction: 'Entities',
  };

  async function refresh() {
    loadError = null;
    entriesError = null;
    try {
      cat = await models.catalog(config);
    } catch (e) {
      loadError = String(e?.message ?? e);
    }
    try {
      const view = await models.catalogEntries(config);
      entries = view.entries;
      live = view.live;
      offline = view.offline;
    } catch (e) {
      entries = [];
      live = false;
      entriesError = String(e?.message ?? e);
    }
  }

  onMount(() => {
    untrack(refresh);
  });

  let missing = $derived(cat ? cat.models.filter((m) => !m.installed) : []);
  let selectedBytes = $derived(
    cat ? cat.models.filter((m) => selected.has(m.id)).reduce((n, m) => n + Math.max(0, (m.size ?? 0) - m.have), 0) : 0,
  );
  let tooBig = $derived(cat?.freeBytes != null && selectedBytes > cat.freeBytes);

  function toggle(id) {
    const next = new Set(selected);
    next.has(id) ? next.delete(id) : next.add(id);
    selected = next;
  }

  function selectAll() {
    selected = new Set(missing.map((m) => m.id));
  }

  /** Follows a download's events in `progress`; true once it finished. */
  function track(e, stoppedNote) {
    if (e.event === 'start') progress.total = e.totalBytes;
    else if (e.event === 'file') {
      progress.file = e.local;
      progress.fileSize = e.size;
      progress.fileBytes = e.from;
      progress.verifying = false;
    } else if (e.event === 'progress') {
      progress.done = e.doneBytes;
      progress.fileBytes = e.fileBytes;
    } else if (e.event === 'verifying') progress.verifying = true;
    else if (e.event === 'cancelled') note = stoppedNote;
    return e.event === 'done';
  }

  function startProgress() {
    progress = { done: 0, total: null, file: null, fileBytes: 0, fileSize: null, verifying: false };
  }

  async function download() {
    error = null;
    note = null;
    busy = 'download';
    startProgress();
    let finished = false;
    try {
      await models.download(config, [...selected], (e) => {
        if (track(e, 'Stopped. What was downloaded is kept; Download resumes it.')) finished = true;
      });
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = null;
      progress = null;
    }
    if (finished) {
      selected = new Set();
      note = 'Downloaded. The local server restarted with them.';
      await onchanged?.();
    }
    // After the restart: its start enables the models whose files arrived.
    await refresh();
  }

  async function chooseFolder() {
    error = null;
    note = null;
    const dir = await openDialog({ directory: true, multiple: false, defaultPath: cat?.modelsDir });
    if (!dir) return;
    busy = 'move';
    progress = { done: 0, total: null };
    try {
      await models.setDir(config, dir, moveFiles, (e) => {
        if (e.event === 'start') progress.total = e.totalBytes;
        else if (e.event === 'progress') progress.done = e.doneBytes;
      });
      note = `Models folder: ${dir}. The local server restarted with it.`;
      await onchanged?.();
      await refresh();
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = null;
      progress = null;
    }
  }

  const CAPABILITY = { thinking: 'Thinking', tools: 'Tools' };

  let available = $derived(entries ? entries.filter((m) => !m.inPool) : []);
  let pooled = $derived(entries ? entries.filter((m) => m.inPool) : []);

  /** The variant picked in the row: the user's choice, else the pool's, else the default. */
  function quantOf(entry) {
    return (
      chosen[entry.id] ?? entry.currentQuant ?? entry.variants.find((v) => v.default)?.quant ?? entry.variants[0]?.quant
    );
  }

  function variantOf(entry) {
    const q = quantOf(entry);
    return entry.variants.find((v) => v.quant === q) ?? entry.variants[0];
  }

  function missingBytes(v) {
    return Math.max(0, v.size - v.have);
  }

  /**
   * Add: with a live server, downloads the chosen variant's missing files and adds it to the
   * server's pool (no restart). Without one, on the app's own config: downloads, appends its
   * block to the config and starts the server.
   */
  async function add(entry) {
    if (live) return change(entry, 'add');
    const v = variantOf(entry);
    error = null;
    note = null;
    shownBlock = null;
    confirming = null;
    busy = 'download';
    adding = { id: entry.id, label: 'Adding…' };
    startProgress();
    let added = false;
    try {
      added = await models.addFromCatalog(config, binary, entry.id, v.quant, (e) =>
        track(e, `Stopped. What was downloaded is kept; Add resumes ${entry.name}.`),
      );
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = null;
      adding = null;
      progress = null;
    }
    if (added) {
      note = `Added ${entry.name} (${v.quant}) to the config. The local server started with it.`;
      await onchanged?.();
    }
    await refresh();
  }

  const DONE = {
    add: (name, q) => `Added ${name} (${q}). The server serves it now; its first request loads it.`,
    replace: (name, q) => `Replaced ${name} with ${q}, under the same id. Its next request loads the new one.`,
    remove: (name) => `Removed ${name} from the server. Its files stay in the models folder.`,
  };

  /** Add, Replace or Remove on the running server's pool. */
  async function change(entry, action) {
    const v = variantOf(entry);
    error = null;
    note = null;
    shownBlock = null;
    confirming = null;
    adding = { id: entry.id, label: { add: 'Adding…', replace: 'Replacing…', remove: 'Removing…' }[action] };
    const download = action !== 'remove' && !v.installed;
    busy = download ? 'download' : 'pool';
    if (download) startProgress();
    let result = null;
    try {
      if (action === 'remove') result = await models.poolRemove(entry.id);
      else
        result = await models.poolPut(config, entry.id, v.quant, action === 'replace', (e) => {
          // Once the files are there, the server call is not a download Stop can end.
          if (track(e, `Stopped. What was downloaded is kept; ${action === 'add' ? 'Add' : 'Replace'} resumes it.`)) {
            busy = 'pool';
            progress = null;
          }
        });
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = null;
      adding = null;
      progress = null;
    }
    if (result) {
      note = DONE[action](entry.name, v.quant);
      if (!result.persisted) note += ' The server could not save it, so it lasts until the server restarts.';
      const next = { ...chosen };
      delete next[entry.id];
      chosen = next;
      await onpool?.();
    }
    await refresh();
  }

  /** For a config the app does not write: the block to paste into it. */
  async function showBlock(entry) {
    const v = variantOf(entry);
    error = null;
    copied = false;
    try {
      shownBlock = { id: entry.id, quant: v.quant, text: await models.catalogBlock(binary, entry.id, v.quant) };
    } catch (e) {
      shownBlock = null;
      error = String(e?.message ?? e);
    }
  }

  async function copyBlock() {
    await navigator.clipboard.writeText(shownBlock.text);
    copied = true;
  }

  function pct(done, total) {
    return total ? Math.min(100, (100 * done) / total) : 0;
  }

  function onKeydown(e) {
    if (e.key === 'Escape' && !busy) onclose();
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#snippet row(m)}
  {@const v = variantOf(m)}
  {@const short = cat?.freeBytes != null && missingBytes(v) > cat.freeBytes}
  {@const same = m.inPool && m.currentQuant === v.quant}
  <li class="px-5 py-3 space-y-1.5">
    <div class="flex items-center gap-2 flex-wrap text-sm">
      <span class="text-slate-200 font-medium">{m.name}</span>
      <span class="text-xs text-slate-500">{m.publisher}</span>
      <span class="text-[10px] px-1.5 rounded bg-slate-800 text-slate-400" title="License">{m.license}</span>
      {#each [m.modality, ...m.additionalModalities] as mod}
        <span class="text-[10px] px-1.5 rounded bg-slate-800 text-slate-300">{MODALITY[mod] ?? mod}</span>
      {/each}
      {#each m.capabilities as c}
        <span class="text-[10px] px-1.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-900">{CAPABILITY[c] ?? c}</span>
      {/each}
      {#if m.inPool}
        <span class="text-[10px] px-1.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-900"
          title={m.currentQuant ? `Served by the server as ${m.id}, variant ${m.currentQuant}` : `Served by the server as ${m.id}`}>
          In the pool{m.currentQuant ? `: ${m.currentQuant}` : ''}
        </span>
        {#if m.enabled === false}
          <span class="text-[10px] px-1.5 rounded bg-slate-800 text-slate-400" title="Disabled">off</span>
        {/if}
      {/if}
      <span class="ml-auto font-mono text-[11px] text-slate-500">{m.id}</span>
    </div>
    <p class="text-xs text-slate-400">{m.summary}</p>
    <div class="flex items-center gap-2 text-xs">
      <select class="bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-slate-200 disabled:opacity-40"
        value={v.quant} onchange={(e) => (chosen = { ...chosen, [m.id]: e.currentTarget.value })}
        disabled={!!busy || (m.inPool && !live)} aria-label="Variant of {m.name}">
        {#each m.variants as o (o.quant)}
          <option value={o.quant}>{o.quant}, {models.gb(o.size)}{o.default ? ' (default)' : ''}{m.inPool && o.quant === m.currentQuant ? ' (in use)' : ''}</option>
        {/each}
      </select>
      <span class="tabular-nums {short ? 'text-rose-300' : 'text-slate-500'}">
        {#if v.installed}
          <span class="text-emerald-400">Downloaded</span>
        {:else if v.have > 0}
          {models.gb(v.have)} of {models.gb(v.size)} downloaded
        {:else}
          Not downloaded
        {/if}
        {#if short && !v.installed}: more than the free space{/if}
      </span>
      <span class="ml-auto"></span>
      {#if adding?.id === m.id}
        <span class="text-slate-400">{adding.label}</span>
      {:else if m.inPool && !live}
        <span class="text-slate-500" title="Replace and Remove change the running server's pool">Start the server to change it</span>
      {:else if m.inPool && confirming === m.id}
        <span class="text-rose-300">Remove {m.id} from the server?</span>
        <button class="px-2.5 py-1 rounded-lg bg-rose-600 hover:bg-rose-500 text-white" onclick={() => change(m, 'remove')}>Remove</button>
        <button class="px-2.5 py-1 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800" onclick={() => (confirming = null)}>Cancel</button>
      {:else if m.inPool}
        <button class="px-2.5 py-1 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-40"
          onclick={() => change(m, 'replace')} disabled={!!busy || same || short || !cat?.dirExists}
          title={same ? 'Pick another variant to replace it with' : `Serve ${m.id} with ${v.quant} from now on`}>
          Replace
        </button>
        <button class="px-2.5 py-1 rounded-lg border border-rose-900 text-rose-300 hover:bg-rose-950 disabled:opacity-40"
          onclick={() => (confirming = m.id)} disabled={!!busy}>
          Remove
        </button>
      {:else if live || cat?.managed}
        <button class="px-2.5 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
          onclick={() => add(m)} disabled={!!busy || short || !cat?.dirExists || (!live && !binary)}
          title={live || binary ? '' : 'No server binary: choose one in the local server settings'}>
          Add
        </button>
      {:else}
        <button class="px-2.5 py-1 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-40"
          onclick={() => showBlock(m)} disabled={!!busy || !binary}>
          Config block
        </button>
      {/if}
    </div>
    {#if v.note}
      <p class="text-[11px] text-slate-500">{v.note}</p>
    {/if}
    {#if shownBlock && shownBlock.id === m.id}
      <div class="relative">
        <pre class="text-[11px] font-mono text-slate-300 bg-slate-950 border border-slate-800 rounded p-2 overflow-x-auto">{shownBlock.text}</pre>
        <button class="absolute top-1.5 right-1.5 text-[11px] px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800"
          onclick={copyBlock}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    {/if}
  </li>
{/snippet}


<div class="fixed inset-0 z-30 bg-black/60 flex items-center justify-center p-4" role="presentation" onclick={(e) => e.target === e.currentTarget && !busy && onclose()}>
  <div role="dialog" aria-modal="true" aria-labelledby="models-title" class="w-full max-w-2xl max-h-full flex flex-col rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
    <div class="p-5 space-y-3 shrink-0">
      <div class="flex items-center gap-3">
        <h2 id="models-title" class="text-sm font-semibold">Models</h2>
        <button class="ml-auto text-slate-500 hover:text-slate-200 disabled:opacity-30" onclick={onclose} disabled={!!busy} aria-label="Close">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18" /></svg>
        </button>
      </div>
      {#if firstRun}
        <p class="text-xs text-slate-300">
          Welcome. The server is installed, but it has no models yet. Pick what you want to use: each one is downloaded
          once into the models folder below and enabled automatically.
        </p>
      {/if}

      {#if cat}
        <div class="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-xs space-y-1.5">
          <div class="flex items-center gap-2">
            <span class="text-slate-400 shrink-0">Folder</span>
            <span class="font-mono text-slate-300 truncate" title={cat.modelsDir}>{cat.modelsDir}</span>
            {#if cat.managed}
              <button class="ml-auto shrink-0 px-2 py-0.5 rounded border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-40" onclick={chooseFolder} disabled={!!busy}>
                Change…
              </button>
            {/if}
          </div>
          <div class="flex items-center gap-3 text-slate-500">
            {#if !cat.dirExists}
              <span class="text-amber-300">Not found: an unplugged drive? Connect it or choose another folder.</span>
            {:else if cat.freeBytes != null}
              <span>{models.gb(cat.freeBytes)} free</span>
            {/if}
            {#if cat.managed}
              <label class="ml-auto flex items-center gap-1.5">
                <input type="checkbox" bind:checked={moveFiles} class="accent-indigo-500" disabled={!!busy} />
                Move the files to the new folder
              </label>
            {:else}
              <span class="ml-auto">Set by the config ({config.split('/').pop()}): <code>[server] models_dir</code></span>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <div class="flex-1 min-h-0 overflow-y-auto border-t border-slate-800">
      <div class="sticky top-0 z-10 flex gap-1 px-5 pt-2 bg-slate-900 border-b border-slate-800" role="tablist">
        {#each [['lineup', 'Lineup'], ['catalog', `Catalog${entries ? ` (${available.length})` : ''}`]] as [id, label] (id)}
          <button role="tab" aria-selected={view === id}
            class="px-3 py-1.5 text-xs -mb-px border-b-2 {view === id ? 'border-indigo-400 text-slate-100' : 'border-transparent text-slate-400 hover:text-slate-200'}"
            onclick={() => (view = id)}>{label}</button>
        {/each}
      </div>
      {#if view === 'catalog'}
        <!-- the catalog, below -->
      {:else if loadError}
        <p class="p-5 text-xs text-rose-300">{loadError}</p>
      {:else if !cat}
        <p class="p-5 text-xs text-slate-500">Reading the lineup…</p>
      {:else}
        {#if cat.errors.length}
          <p class="px-5 pt-3 text-xs text-amber-300">
            Could not reach every host (offline?): some sizes are unknown and those models cannot be downloaded now.
          </p>
        {/if}
        <ul class="divide-y divide-slate-800">
          {#each cat.models as m (m.id)}
            <li class="px-5 py-2.5 flex items-center gap-3 text-sm">
              {#if m.installed}
                <span class="w-4 h-4 text-emerald-400" title="Downloaded">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12l5 5L20 7" /></svg>
                </span>
              {:else}
                <input type="checkbox" class="accent-indigo-500" checked={selected.has(m.id)} onchange={() => toggle(m.id)}
                  disabled={!!busy} aria-label="Download {m.id}" />
              {/if}
              <span class="font-mono text-xs text-slate-200">{m.id}</span>
              <span class="text-xs text-slate-500">{MODALITY[m.modality] ?? m.modality ?? ''}</span>
              {#if m.inConfig && m.enabled === false && m.installed}
                <span class="text-[10px] px-1.5 rounded bg-slate-800 text-slate-400" title="Disabled in the config">off</span>
              {/if}
              {#if !m.inConfig}
                <span class="text-[10px] px-1.5 rounded bg-slate-800 text-slate-500" title="Not in this config: downloading it changes nothing until a [[models]] block uses it">not in config</span>
              {/if}
              <span class="ml-auto text-xs tabular-nums {m.installed ? 'text-slate-500' : 'text-slate-300'}">
                {#if m.installed}
                  {models.gb(m.have)}
                {:else if m.have > 0}
                  {models.gb(m.have)} of {models.gb(m.size)}
                {:else}
                  {models.gb(m.size)}
                {/if}
              </span>
            </li>
          {/each}
        </ul>
      {/if}

      {#if view === 'catalog' && !entries}
        <p class="p-5 text-xs text-slate-500">Reading the catalog…</p>
      {:else if view === 'catalog'}
        <div>
          <div class="px-5 pt-3 pb-1">
            <p class="text-xs text-slate-500">
              {#if live}
                Tested models, added to the running server and taken out of it without a restart. Add downloads the chosen
                variant if needed; the server keeps the change next to {config.split('/').pop()}.
              {:else if cat?.managed}
                Working from the config: {offline}. Add downloads the chosen variant, adds it to the config and
                (re)starts the local server; with the server running, models are added, replaced and removed live.
              {:else}
                Working from the config: {offline}. Download the files with ./run.sh models and paste the block into
                {config.split('/').pop()}, or start the server to add models to it directly.
              {/if}
            </p>
          </div>
          {#if entriesError}
            <p class="px-5 py-2 text-xs text-amber-300">{entriesError}</p>
          {/if}
          {#if available.length}
            <h3 class="px-5 pt-3 text-[11px] uppercase tracking-wide text-slate-500">Not in the pool</h3>
          {:else if !entriesError}
            <p class="px-5 py-2 text-xs text-slate-500">The server has every model of the catalog.</p>
          {/if}
          <ul class="divide-y divide-slate-800">
            {#each available as m (m.id)}
              {@render row(m)}
            {/each}
          </ul>
          {#if pooled.length}
            <h3 class="px-5 pt-4 text-[11px] uppercase tracking-wide text-slate-500 border-t border-slate-800">In the pool</h3>
            <ul class="divide-y divide-slate-800">
              {#each pooled as m (m.id)}
                {@render row(m)}
              {/each}
            </ul>
          {/if}
        </div>
      {/if}
    </div>

    <div class="p-5 space-y-3 border-t border-slate-800 shrink-0">
      {#if progress}
        <div class="space-y-1">
          <div class="h-1.5 rounded bg-slate-800 overflow-hidden">
            <div class="h-full bg-indigo-500 transition-all" style="width: {pct(progress.done, progress.total)}%"></div>
          </div>
          <p class="text-xs text-slate-400 tabular-nums flex gap-2">
            {#if busy === 'move'}
              <span>Moving the files… {models.gb(progress.done)}{progress.total ? ` of ${models.gb(progress.total)}` : ''}</span>
            {:else if progress.verifying}
              <span class="truncate">Checking {progress.file}…</span>
            {:else}
              <span class="truncate">{progress.file ?? 'Starting…'}</span>
              <span class="ml-auto shrink-0">{models.gb(progress.done)}{progress.total ? ` of ${models.gb(progress.total)}` : ''}</span>
            {/if}
          </p>
        </div>
      {/if}
      {#if error}
        <p class="text-xs text-rose-300">{error}</p>
      {/if}
      {#if note}
        <p class="text-xs text-emerald-300">{note}</p>
      {/if}
      <div class="flex items-center gap-3">
        {#if missing.length && !busy}
          <button class="text-xs text-slate-400 hover:text-slate-200" onclick={selectAll}>Select all</button>
        {/if}
        <span class="text-xs {tooBig ? 'text-rose-300' : 'text-slate-500'}">
          {#if selected.size}
            {selected.size} selected, {models.gb(selectedBytes)}{tooBig ? ': more than the free space' : ''}
          {/if}
        </span>
        <span class="ml-auto"></span>
        {#if busy === 'download'}
          <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800" onclick={models.cancel}>Stop</button>
        {:else}
          <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800 disabled:opacity-40" onclick={onclose} disabled={!!busy}>Close</button>
          <button class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
            onclick={download} disabled={!!busy || !selected.size || tooBig || !cat?.dirExists}>
            Download
          </button>
        {/if}
      </div>
    </div>
  </div>
</div>
