<script>
  import { onMount } from 'svelte';
  import { formatBytes, formatMs, tierName } from './format.js';
  import * as api from './hypnos.js';

  let { status, lastResponse } = $props();

  const POLL_MS = 1500;
  const TIERS = ['tier1_gpu_vram', 'tier2_pinned_ram', 'tier3_nvme_ssd'];
  const TIER_STYLE = {
    tier1_gpu_vram: { text: 'text-emerald-300', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
    tier2_pinned_ram: { text: 'text-amber-300', badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
    tier3_nvme_ssd: { text: 'text-slate-300', badge: 'bg-slate-700/60 text-slate-300 border-slate-600' },
  };
  const EVICTION_REASONS = {
    memory_budget: 'did not fit in the memory budget',
    full_gpu_placement: 'to fit all layers on the GPU',
    process_memory_cap: 'process memory cap',
    device_out_of_memory: 'engine ran out of device memory',
  };

  let memory = $state(null);
  let error = $state(null);
  let now = $state(Date.now());

  let caps = $derived(memory?.capabilities ?? {});
  let unified = $derived(caps.topology === 'unified_memory' && caps.physical_memory_bytes > 0);
  let t1Used = $derived(memory?.tier1_vram_used_bytes ?? 0);
  let t1Total = $derived(caps.usable_vram_budget_bytes ?? 0);
  let t2Used = $derived(memory?.tier2_pinned_ram_used_bytes ?? 0);
  let t2Total = $derived(caps.pinned_ram_budget_bytes ?? 0);
  let physical = $derived(caps.physical_memory_bytes || 1);
  let models = $derived((memory?.models ?? []).filter((m) => m.enabled));
  let byTier = $derived(Object.fromEntries(TIERS.map((t) => [t, models.filter((m) => m.current_tier === t)])));
  let t3Bytes = $derived(byTier.tier3_nvme_ssd.reduce((sum, m) => sum + m.total_weight_bytes, 0));
  let evictions = $derived((memory?.recent_evictions ?? []).slice(0, 5));
  let processCap = $derived(memory?.process_memory_cap_bytes ?? null);

  function pct(used, total) {
    return total > 0 ? Math.min(100, (used / total) * 100) : 0;
  }

  function timeAgo(unixMs) {
    const secs = Math.max(0, Math.round((now - unixMs) / 1000));
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    return `${Math.round(secs / 3600)}h ago`;
  }

  function duration(secs) {
    if (secs == null) return '–';
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.round(secs / 60)}m`;
    if (secs < 86400) return `${(secs / 3600).toFixed(1)}h`;
    return `${(secs / 86400).toFixed(1)}d`;
  }

  function placementNote(m) {
    const p = m.placement;
    if (m.current_tier !== 'tier1_gpu_vram' || !p) return '';
    if (p.gpu_layers === 0) return 'on CPU';
    const total = p.gpu_layers + p.pinned_ram_layers + p.ssd_layers;
    return p.pinned_ram_layers + p.ssd_layers > 0 ? `${p.gpu_layers}/${total} layers on GPU` : '';
  }

  // Polls while the panel is mounted (it unmounts when collapsed) and the window is visible.
  onMount(() => {
    let live = true;
    let timer;
    const poll = async () => {
      clearTimeout(timer);
      if (document.hidden) return;
      try {
        const next = await api.memoryStatus();
        if (!live) return;
        memory = next;
        error = null;
      } catch (e) {
        if (!live) return;
        error = api.errorMessage(e);
      }
      now = Date.now();
      timer = setTimeout(poll, POLL_MS);
    };
    const onVisibility = () => !document.hidden && poll();
    document.addEventListener('visibilitychange', onVisibility);
    poll();
    return () => {
      live = false;
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  });
</script>

{#snippet bar(label, labelClass, used, total, fill)}
  <div class="space-y-1">
    <div class="flex justify-between text-[11px]">
      <span class="font-medium {labelClass}">{label}</span>
      <span class="font-mono text-slate-400">{formatBytes(used)} / {formatBytes(total)}</span>
    </div>
    <div class="h-2 rounded-full bg-slate-950 border border-slate-800 overflow-hidden">
      <div class="h-full transition-all duration-300 {fill}" style="width: {pct(used, total)}%"></div>
    </div>
  </div>
{/snippet}

{#snippet row(label, value)}
  <div class="flex justify-between gap-2">
    <dt class="text-slate-500">{label}</dt>
    <dd class="font-mono text-slate-300 text-right">{value}</dd>
  </div>
{/snippet}

<div class="p-4 space-y-5 text-xs">
  <section class="space-y-2">
    <h2 class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Server</h2>
    <dl class="space-y-1">
      {@render row('Status', { connecting: 'connecting', unreachable: 'unreachable', starting: 'warming up', ready: 'ready' }[status.state])}
      {#if status.health}
        {@render row('Version', status.health.version)}
        {@render row('Uptime', duration(status.health.uptime_secs))}
      {/if}
      {#if caps.device_name}
        {@render row('Device', caps.device_name)}
      {/if}
    </dl>
    {#if error && status.state !== 'unreachable'}
      <p class="text-rose-300">{error}</p>
    {/if}
  </section>

  {#if memory}
    <section class="space-y-3">
      <h2 class="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
        Memory
        {#if caps.simulated}
          <span class="normal-case font-medium px-1.5 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-300 border border-fuchsia-500/40">simulated</span>
        {/if}
      </h2>

      {#if unified}
        <div class="space-y-1">
          <div class="flex justify-between text-[11px]">
            <span class="font-medium text-slate-200">Unified memory</span>
            <span class="font-mono text-slate-400">{formatBytes(t1Used + t2Used)} / {formatBytes(physical)}</span>
          </div>
          <div class="relative h-2 rounded-full bg-slate-950 border border-slate-800 overflow-hidden flex">
            <div class="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300" style="width: {pct(t1Used, physical)}%"></div>
            <div class="h-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-300" style="width: {Math.min(100 - pct(t1Used, physical), pct(t2Used, physical))}%"></div>
            <div class="absolute top-0 h-full w-0.5 bg-slate-300/70" style="left: {pct(t1Total, physical)}%" title="GPU working-set limit"></div>
          </div>
          <div class="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
            <span><span class="text-emerald-300">■</span> GPU {formatBytes(t1Used)} / {formatBytes(t1Total)}</span>
            <span><span class="text-amber-300">■</span> Warm {formatBytes(t2Used)} / {formatBytes(t2Total)}</span>
          </div>
        </div>
      {:else}
        {@render bar('Tier 1 · GPU VRAM', 'text-emerald-300', t1Used, t1Total, 'bg-gradient-to-r from-emerald-500 to-teal-400')}
        {@render bar('Tier 2 · Pinned RAM', 'text-amber-300', t2Used, t2Total, 'bg-gradient-to-r from-amber-500 to-orange-400')}
      {/if}

      <div class="flex justify-between text-[11px]">
        <span class="font-medium text-slate-300">Tier 3 · NVMe</span>
        <span class="font-mono text-slate-400">{formatBytes(t3Bytes)} mapped</span>
      </div>

      {#if memory.process_memory_bytes != null}
        {#if processCap}
          {@render bar('Hypnos process (cap)', 'text-sky-300', memory.process_memory_bytes, processCap,
            pct(memory.process_memory_bytes, processCap) > 95 ? 'bg-rose-500' : pct(memory.process_memory_bytes, processCap) > 80 ? 'bg-amber-400' : 'bg-gradient-to-r from-sky-500 to-cyan-400')}
        {:else}
          <div class="flex justify-between text-[11px]">
            <span class="font-medium text-slate-300">Hypnos process</span>
            <span class="font-mono text-slate-400">{formatBytes(memory.process_memory_bytes)}</span>
          </div>
        {/if}
      {/if}

      {#if memory.gpu}
        <div class="text-[11px] {memory.gpu.throttled ? 'text-rose-300' : 'text-slate-400'}">
          GPU {memory.gpu.utilization_percent ?? '–'}% · {memory.gpu.temperature_celsius ?? '–'} °C ·
          {memory.gpu.power_watts != null ? memory.gpu.power_watts.toFixed(0) : '–'} W{memory.gpu.throttled ? ' · throttled' : ''}
        </div>
      {/if}
    </section>

    <section class="space-y-2">
      <h2 class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Models</h2>
      {#each TIERS as tier (tier)}
        {#if byTier[tier].length}
          <div class="space-y-1">
            <h3 class="text-[11px] font-medium {TIER_STYLE[tier].text}">{tierName(tier)}</h3>
            <ul class="space-y-1">
              {#each byTier[tier] as m (m.id)}
                <li class="rounded-md border border-slate-800 bg-slate-900/60 px-2 py-1.5">
                  <div class="flex items-center gap-1.5">
                    {#if m.active_leases > 0}
                      <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" title="{m.active_leases} active lease(s)"></span>
                    {/if}
                    <span class="flex-1 truncate text-slate-200" title={m.id}>{m.id}</span>
                    <span class="font-mono text-slate-500">{formatBytes(m.total_weight_bytes)}</span>
                  </div>
                  <div class="mt-0.5 flex gap-2 text-[11px] text-slate-500">
                    <span>{m.modality.replaceAll('_', ' ')}</span>
                    {#if placementNote(m)}<span class="text-amber-300">{placementNote(m)}</span>{/if}
                    <span class="ml-auto" title="requests served · idle">{m.total_requests_served} req · idle {duration(m.idle_seconds)}</span>
                  </div>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      {/each}
    </section>
  {/if}

  <section class="space-y-2">
    <h2 class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Last response</h2>
    {#if lastResponse}
      {@const t = lastResponse.telemetry}
      {@const e = t.engine ?? {}}
      <dl class="space-y-1">
        {@render row('Model', lastResponse.model)}
        {#if e.ttft_ms != null}{@render row('Time to first token', formatMs(e.ttft_ms))}{/if}
        {#if e.decode_tokens_per_s != null}{@render row('Decode', `${e.decode_tokens_per_s.toFixed(1)} tok/s`)}{/if}
        {#if e.prefill_tokens_per_s != null}{@render row('Prefill', `${e.prefill_tokens_per_s.toFixed(1)} tok/s`)}{/if}
        {#if lastResponse.usage}
          {@render row('Tokens in / out', `${lastResponse.usage.prompt_tokens} / ${lastResponse.usage.completion_tokens}`)}
        {/if}
        {#if e.cached_tokens != null}{@render row('Cached prompt tokens', e.cached_tokens)}{/if}
        {#if e.queue_ms != null}{@render row('Queue', formatMs(e.queue_ms))}{/if}
        {@render row('Inference', formatMs(t.inference_latency_ms))}
        {@render row('Model was in', tierName(t.previous_tier))}
        {@render row('Tier switch', formatMs(t.tier_switch_latency_ms))}
        {#if t.tier_switch_phases}
          {@const p = t.tier_switch_phases}
          <div class="pl-3 space-y-1 border-l border-slate-800">
            {#if p.load_mode}{@render row('Load mode', p.load_mode)}{/if}
            {#if p.weights_ms != null}{@render row('Weights', formatMs(p.weights_ms))}{/if}
            {#if p.projector_ms != null}{@render row('Projector', formatMs(p.projector_ms))}{/if}
            {#if p.context_ms != null}{@render row('Context', formatMs(p.context_ms))}{/if}
          </div>
        {/if}
        {#if t.placement}
          {@render row('GPU layers', t.placement.gpu_layers)}
          {@render row('Context window', t.placement.allocated_context_len)}
        {/if}
        {#if t.evicted_models?.length}
          {@render row('Evicted', t.evicted_models.join(', '))}
        {/if}
      </dl>
    {:else}
      <p class="text-slate-500">Send a message to see its timings.</p>
    {/if}
  </section>

  {#if evictions.length}
    <section class="space-y-2">
      <h2 class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Recent evictions</h2>
      <ul class="space-y-1 text-[11px]">
        {#each evictions as ev (ev.unix_ms + ev.model)}
          <li class="text-slate-400">
            <span class="text-slate-200">{ev.model}</span> → {tierName(ev.to_tier)} for
            <span class="text-slate-200">{ev.for_model}</span>
            <span class="text-slate-500">({EVICTION_REASONS[ev.reason] ?? ev.reason}, {timeAgo(ev.unix_ms)})</span>
          </li>
        {/each}
      </ul>
    </section>
  {/if}
</div>
