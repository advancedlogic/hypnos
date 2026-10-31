/** "42.1 tok/s · TTFT 125 ms · loaded from Tier 2 in 1.2 s" for an assistant message. */
export function telemetrySummary(telemetry, usage) {
  if (!telemetry) return '';
  const e = telemetry.engine ?? {};
  const parts = [];
  if (e.decode_tokens_per_s) parts.push(`${e.decode_tokens_per_s.toFixed(1)} tok/s`);
  if (e.ttft_ms != null) parts.push(`TTFT ${formatMs(e.ttft_ms)}`);
  if (usage?.completion_tokens) parts.push(`${usage.completion_tokens} tokens`);
  if (telemetry.previous_tier && telemetry.previous_tier !== 'tier1_gpu_vram') {
    parts.push(`loaded from ${tierName(telemetry.previous_tier)} in ${formatMs(telemetry.tier_switch_latency_ms)}`);
  }
  return parts.join(' · ');
}

export function tierName(tier) {
  return { tier1_gpu_vram: 'Tier 1', tier2_pinned_ram: 'Tier 2', tier3_nvme_ssd: 'Tier 3' }[tier] ?? tier;
}

export function formatMs(ms) {
  if (ms == null) return '–';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

export function formatBytes(bytes) {
  if (bytes == null) return '–';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}
