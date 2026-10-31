<script>
  import { onMount } from 'svelte';
  import * as api from './hypnos.js';
  import { formatMs, tierName } from './format.js';

  /** `focus`: a request id to select (an answer's "Log" link); `onclose` returns to the chat. */
  let { focus = null, onclose } = $props();

  const POLL_MS = 3000;
  const LEVELS = ['off', 'error', 'warn', 'info', 'debug', 'trace'];
  const STATUSES = [
    ['', 'Any status'],
    ['ok', 'OK'],
    ['error', 'Errors'],
    ['4xx', '4xx'],
    ['5xx', '5xx'],
  ];

  let list = $state(null);
  let error = $state(null);
  let model = $state('');
  let route = $state('');
  let status = $state('');
  let paused = $state(false);
  let selectedId = $state(null);
  let record = $state(null);
  let recordError = $state(null);
  let notice = $state(null);
  let seenModels = $state(new Set());
  let seenRoutes = $state(new Set());

  let records = $derived(list?.records ?? []);

  async function refresh() {
    try {
      const next = await api.logs({ model, route, status, limit: 200 });
      list = next;
      error = null;
      for (const r of next.records) {
        if (r.model) seenModels.add(r.model);
        if (r.route) seenRoutes.add(r.route);
      }
      seenModels = new Set(seenModels);
      seenRoutes = new Set(seenRoutes);
    } catch (e) {
      error = api.errorMessage(e);
    }
  }

  async function open(id) {
    selectedId = id;
    record = null;
    recordError = null;
    notice = null;
    try {
      record = await api.log(id);
    } catch (e) {
      recordError = api.errorMessage(e);
    }
  }

  async function setLevel(level) {
    try {
      await api.setLogLevel(level);
      await refresh();
    } catch (e) {
      error = api.errorMessage(e);
    }
  }

  async function clear() {
    if (!confirm('Clear the server’s request log? This removes every record for everyone.')) return;
    try {
      await api.clearLogs();
      selectedId = null;
      record = null;
      await refresh();
    } catch (e) {
      error = api.errorMessage(e);
    }
  }

  async function copyRecord() {
    await navigator.clipboard.writeText(JSON.stringify(record, null, 2));
    notice = 'Copied as JSON.';
  }

  function time(iso) {
    if (!iso) return '–';
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function statusClass(code) {
    if (code >= 500) return 'text-rose-300';
    if (code >= 400) return 'text-amber-300';
    return 'text-emerald-300';
  }

  function onKeydown(e) {
    if (e.key === 'Escape') {
      if (selectedId) selectedId = null;
      else onclose();
    }
  }

  // Filters apply at once.
  $effect(() => {
    model;
    route;
    status;
    refresh();
  });

  onMount(() => {
    let timer;
    const tick = async () => {
      if (!paused && !document.hidden) await refresh();
      timer = setTimeout(tick, POLL_MS);
    };
    timer = setTimeout(tick, POLL_MS);
    if (focus) open(focus);
    return () => clearTimeout(timer);
  });
</script>

<svelte:window onkeydown={onKeydown} />

<div class="h-full flex flex-col">
  <div class="shrink-0 px-4 py-2 border-b border-slate-800/60 flex flex-wrap items-center gap-2 text-xs">
    <h2 class="text-sm font-semibold mr-2">Request log</h2>
    <select bind:value={model} class="rounded-md bg-slate-950 border border-slate-700 px-2 py-1">
      <option value="">Any model</option>
      {#each [...seenModels].sort() as m (m)}<option value={m}>{m}</option>{/each}
    </select>
    <select bind:value={route} class="rounded-md bg-slate-950 border border-slate-700 px-2 py-1">
      <option value="">Any route</option>
      {#each [...seenRoutes].sort() as r (r)}<option value={r}>{r}</option>{/each}
    </select>
    <select bind:value={status} class="rounded-md bg-slate-950 border border-slate-700 px-2 py-1">
      {#each STATUSES as [v, label] (v)}<option value={v}>{label}</option>{/each}
    </select>
    <label class="flex items-center gap-1.5 text-slate-400 ml-1">
      <input type="checkbox" bind:checked={paused} class="accent-indigo-500" /> pause
    </label>
    <span class="flex-1"></span>
    {#if list}
      <span class="text-slate-500">{list.stored} / {list.capacity} kept</span>
      <label class="flex items-center gap-1.5 text-slate-400" title="What the next requests capture: debug adds the text, trace the payloads">
        level
        <select value={list.level} onchange={(e) => setLevel(e.currentTarget.value)} class="rounded-md bg-slate-950 border border-slate-700 px-2 py-1">
          {#each LEVELS as l (l)}<option value={l}>{l}</option>{/each}
        </select>
      </label>
    {/if}
    <button class="px-2 py-1 rounded-md border border-slate-700 text-slate-300 hover:bg-slate-800" onclick={clear}>Clear</button>
    <button class="px-2 py-1 rounded-md text-slate-400 hover:text-slate-100" onclick={onclose} title="Back to the chat (Esc)">Close</button>
  </div>

  {#if error}
    <p class="shrink-0 px-4 py-1.5 text-xs text-rose-300 bg-rose-500/10 border-b border-rose-500/30">{error}</p>
  {/if}

  <div class="flex-1 min-h-0 flex">
    <div class="flex-1 min-w-0 overflow-auto">
      <table class="w-full text-xs">
        <thead class="sticky top-0 bg-slate-950 text-slate-500 text-left">
          <tr>
            <th class="px-3 py-1.5 font-medium">Time</th>
            <th class="px-2 py-1.5 font-medium">Route</th>
            <th class="px-2 py-1.5 font-medium">Model</th>
            <th class="px-2 py-1.5 font-medium">Status</th>
            <th class="px-2 py-1.5 font-medium text-right">Duration</th>
            <th class="px-2 py-1.5 font-medium text-right">Tokens</th>
            <th class="px-2 py-1.5 font-medium text-right">tok/s</th>
            <th class="px-2 py-1.5 font-medium">Tier</th>
            <th class="px-2 py-1.5 font-medium">Key</th>
          </tr>
        </thead>
        <tbody>
          {#each records as r (r.id)}
            <tr
              class="border-t border-slate-800/60 cursor-pointer hover:bg-slate-800/40
                {r.id === selectedId ? 'bg-indigo-500/15' : r.id === focus ? 'bg-indigo-500/5' : ''}"
              onclick={() => open(r.id)}
            >
              <td class="px-3 py-1.5 font-mono text-slate-400 whitespace-nowrap">{time(r.started_at)}</td>
              <td class="px-2 py-1.5 text-slate-300 whitespace-nowrap">{r.method} {r.route}{r.stream ? ' · stream' : ''}</td>
              <td class="px-2 py-1.5 text-slate-300 whitespace-nowrap">{r.model ?? '–'}</td>
              <td class="px-2 py-1.5 font-mono {statusClass(r.status)}" title={r.error ?? ''}>{r.status}{r.error ? ' !' : ''}</td>
              <td class="px-2 py-1.5 font-mono text-right text-slate-400">{formatMs(r.duration_ms)}</td>
              <td class="px-2 py-1.5 font-mono text-right text-slate-400 whitespace-nowrap">
                {r.tokens_in != null || r.tokens_out != null ? `${r.tokens_in ?? 0} / ${r.tokens_out ?? 0}` : '–'}
              </td>
              <td class="px-2 py-1.5 font-mono text-right text-slate-400">{r.tokens_per_s != null ? r.tokens_per_s.toFixed(1) : '–'}</td>
              <td class="px-2 py-1.5 text-slate-400 whitespace-nowrap">
                {r.previous_tier ? tierName(r.previous_tier) : '–'}{r.tier_switch_ms ? ` · ${formatMs(r.tier_switch_ms)}` : ''}
              </td>
              <td class="px-2 py-1.5 text-slate-500">{r.key ?? ''}</td>
            </tr>
          {:else}
            <tr><td colspan="9" class="px-3 py-6 text-center text-slate-500">{list ? 'No requests recorded yet.' : 'Loading…'}</td></tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if selectedId}
      <aside class="w-[28rem] max-w-[50%] shrink-0 border-l border-slate-800 bg-slate-900/40 overflow-y-auto p-4 space-y-3 text-xs">
        <div class="flex items-center gap-2">
          <h3 class="text-sm font-semibold truncate flex-1 font-mono" title={selectedId}>{selectedId}</h3>
          {#if record}<button class="text-slate-400 hover:text-slate-100" onclick={copyRecord}>Copy JSON</button>{/if}
          <button class="text-slate-500 hover:text-slate-200" onclick={() => (selectedId = null)} aria-label="Close">×</button>
        </div>
        {#if notice}<p class="text-slate-400">{notice}</p>{/if}
        {#if recordError}
          <p class="text-rose-300">{recordError}</p>
        {:else if !record}
          <p class="text-slate-500">Loading…</p>
        {:else}
          <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            {#each [
              ['Route', `${record.method} ${record.route}`],
              ['Status', record.status],
              ['Model', record.model],
              ['Started', record.started_at],
              ['Duration', record.duration_ms != null ? formatMs(record.duration_ms) : null],
              ['Inference', record.inference_ms != null ? formatMs(record.inference_ms) : null],
              ['Tier switch', record.tier_switch_ms != null ? formatMs(record.tier_switch_ms) : null],
              ['Previous tier', record.previous_tier ? tierName(record.previous_tier) : null],
              ['Tokens in / out', record.tokens_in != null ? `${record.tokens_in} / ${record.tokens_out ?? 0}` : null],
              ['Cached tokens', record.cached_tokens],
              ['Speed', record.tokens_per_s != null ? `${record.tokens_per_s.toFixed(1)} tok/s` : null],
              ['Key', record.key],
              ['Captured at', record.level],
            ].filter(([, v]) => v != null && v !== '') as [label, value] (label)}
              <dt class="text-slate-500">{label}</dt>
              <dd class="font-mono text-slate-300 break-all">{value}</dd>
            {/each}
          </dl>
          {#if record.error}
            <div class="rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-1.5 text-rose-300">{record.error}</div>
          {/if}
          {#each [['Input', record.input], ['Output', record.output]] as [label, text] (label)}
            {#if text}
              <div>
                <h4 class="text-slate-400 mb-1">{label}{record.truncated ? ' (truncated)' : ''}</h4>
                <pre class="max-h-60 overflow-auto rounded-md border border-slate-800 bg-slate-950 p-2 text-slate-300 whitespace-pre-wrap">{text}</pre>
              </div>
            {/if}
          {/each}
          {#each [['Request payload', record.request_payload], ['Response payload', record.response_payload]] as [label, payload] (label)}
            {#if payload}
              <div>
                <h4 class="text-slate-400 mb-1">{label}</h4>
                <pre class="max-h-72 overflow-auto rounded-md border border-slate-800 bg-slate-950 p-2 text-slate-400">{JSON.stringify(payload, null, 2)}</pre>
              </div>
            {/if}
          {/each}
          {#if !record.input && !record.output && !record.request_payload}
            <p class="text-slate-500">The text is captured from the <code>debug</code> level on, the payloads at <code>trace</code>; raise the level for the next requests.</p>
          {/if}
        {/if}
      </aside>
    {/if}
  </div>
</div>
