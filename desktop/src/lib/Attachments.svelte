<script>
  import { formatBytes } from './format.js';

  /** Chips for prepared (or preparing) attachments; `onremove` makes them removable. */
  let { items, onremove = null } = $props();

  let open = $state(null);

  const ICONS = { document: 'DOC', audio: 'AUDIO', video: 'VIDEO' };

  function detail(a) {
    if (a.kind === 'document') return `${a.pages > 1 ? `${a.pages} pages · ` : ''}${a.truncated ? 'cut to fit · ' : ''}${formatBytes(a.size)}`;
    if (a.kind === 'audio') return `${Math.round(a.durationSecs)} s · transcribed`;
    if (a.kind === 'video') return `${a.frames?.length ?? 0} frames${a.transcript ? ' · transcribed' : ''}`;
    return formatBytes(a.size);
  }

  function body(a) {
    if (a.kind === 'document') return a.text;
    return a.transcript ?? '';
  }
</script>

<div class="flex flex-wrap gap-2">
  {#each items as item, i (item.id ?? i)}
    {@const a = item.attachment ?? item}
    <div class="relative group">
      {#if item.status === 'preparing'}
        <div class="h-14 px-3 flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 text-xs text-slate-400">
          <span class="w-3 h-3 rounded-full border-2 border-slate-500 border-t-transparent animate-spin"></span>
          <span class="max-w-40 truncate">{item.name}</span>
          <span class="text-slate-600">{item.step ?? 'preparing'}</span>
        </div>
      {:else if item.status === 'error'}
        <div class="h-14 px-3 flex items-center gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 text-xs text-rose-300 max-w-80" title={item.error}>
          <span class="truncate">{item.error}</span>
        </div>
      {:else if a.kind === 'image'}
        <img src={a.dataUrl} alt={a.name} title={a.name} class="h-14 max-w-32 rounded-lg border border-slate-700 object-cover" />
      {:else}
        <button
          class="h-14 px-3 flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 text-left hover:border-slate-500"
          onclick={() => (open = open === i ? null : i)}
          title="Show the extracted text"
        >
          {#if a.kind === 'video' && a.frames?.length}
            <img src={a.frames[0]} alt="" class="h-9 w-12 rounded object-cover" />
          {:else}
            <span class="text-[9px] font-semibold tracking-wider px-1.5 py-1 rounded bg-slate-800 text-slate-300">{ICONS[a.kind]}</span>
          {/if}
          <span class="flex flex-col">
            <span class="max-w-44 truncate text-xs text-slate-200">{a.name}</span>
            <span class="text-[11px] text-slate-500">{detail(a)}</span>
          </span>
        </button>
      {/if}
      {#if onremove}
        <button
          class="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-slate-700 text-slate-200 text-xs leading-none opacity-0 group-hover:opacity-100"
          onclick={() => onremove(i)}
          aria-label="Remove {item.name ?? a.name}"
        >×</button>
      {/if}
    </div>
  {/each}
</div>
{#if open !== null && items[open]}
  {@const a = items[open].attachment ?? items[open]}
  <pre class="mt-2 max-h-60 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3 text-[11px] text-slate-400 whitespace-pre-wrap">{body(a) || 'No text.'}</pre>
{/if}
