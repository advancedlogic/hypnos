<script>
  import { onMount, untrack } from 'svelte';
  import { open as openDialog } from '@tauri-apps/plugin-dialog';
  import * as models from './models.js';

  /**
   * The local server's model files (DESK-17): pick from the lineup, download with progress,
   * change the folder. `config`: the local server's config path; `onchanged` restarts the server
   * after files arrived or the folder moved; `firstRun` shows the welcome line.
   */
  let { config, firstRun = false, onchanged, onclose } = $props();

  let cat = $state(null);
  let loadError = $state(null);
  let selected = $state(new Set());
  let busy = $state(null); // 'download' | 'move' | null
  let progress = $state(null); // { done, total, file, fileBytes, fileSize, verifying }
  let error = $state(null);
  let note = $state(null);
  let moveFiles = $state(true);

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
    try {
      cat = await models.catalog(config);
    } catch (e) {
      loadError = String(e?.message ?? e);
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

  async function download() {
    error = null;
    note = null;
    busy = 'download';
    progress = { done: 0, total: null, file: null, fileBytes: 0, fileSize: null, verifying: false };
    let finished = false;
    try {
      await models.download(config, [...selected], (e) => {
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
        else if (e.event === 'done') finished = true;
        else if (e.event === 'cancelled') note = 'Stopped. What was downloaded is kept; Download resumes it.';
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

  function pct(done, total) {
    return total ? Math.min(100, (100 * done) / total) : 0;
  }

  function onKeydown(e) {
    if (e.key === 'Escape' && !busy) onclose();
  }
</script>

<svelte:window onkeydown={onKeydown} />

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
      {#if loadError}
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
