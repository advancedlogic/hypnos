<script>
  import { onMount } from 'svelte';
  import { createClient, probe, inTauri } from './hypnos.js';
  import { allowServer, normalizeUrl } from './connection.js';

  let { connection, onsave, onclose } = $props();

  let url = $state(connection.url);
  let apiKey = $state(connection.apiKey);
  let showKey = $state(false);
  let testing = $state(false);
  let saving = $state(false);
  let result = $state(null);
  let urlInput = $state();

  const RESULTS = {
    ready: { text: 'Connected, ready', tone: 'text-emerald-300' },
    starting: { text: 'Connected, still loading models', tone: 'text-amber-300' },
    unauthorized: { text: 'The server refused the API key', tone: 'text-rose-300' },
    unreachable: { text: 'Could not reach the server', tone: 'text-rose-300' },
  };

  onMount(() => urlInput?.select());

  async function test() {
    result = null;
    let normalized;
    try {
      normalized = normalizeUrl(url);
    } catch (e) {
      result = { text: e.message, tone: 'text-rose-300' };
      return;
    }
    testing = true;
    try {
      await allowServer(normalized);
      const outcome = await probe(createClient({ baseUrl: normalized, apiKey: apiKey || undefined }));
      const r = RESULTS[outcome.state];
      result = { ...r, text: outcome.health ? `${r.text} (Hypnos ${outcome.health.version})` : r.text };
    } catch (e) {
      result = { text: String(e?.message ?? e), tone: 'text-rose-300' };
    } finally {
      testing = false;
    }
  }

  async function submit(e) {
    e.preventDefault();
    let normalized;
    try {
      normalized = normalizeUrl(url);
    } catch (err) {
      result = { text: err.message, tone: 'text-rose-300' };
      return;
    }
    saving = true;
    try {
      await onsave({ url: normalized, apiKey: apiKey.trim() });
    } catch (err) {
      result = { text: `Could not save: ${err?.message ?? err}`, tone: 'text-rose-300' };
    } finally {
      saving = false;
    }
  }

  function onKeydown(e) {
    if (e.key === 'Escape') onclose();
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="fixed inset-0 z-30 bg-black/60 flex items-center justify-center p-4" role="presentation" onclick={(e) => e.target === e.currentTarget && onclose()}>
  <div role="dialog" aria-modal="true" aria-labelledby="settings-title" class="w-full max-w-md">
  <form class="rounded-xl border border-slate-700 bg-slate-900 shadow-2xl p-5 space-y-4" onsubmit={submit}>
    <h2 id="settings-title" class="text-sm font-semibold">Connection</h2>

    <label class="block space-y-1">
      <span class="text-xs text-slate-400">Server URL</span>
      <input
        bind:this={urlInput}
        bind:value={url}
        class="w-full rounded-lg bg-slate-950 border border-slate-700 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
        placeholder="http://localhost:9999"
        spellcheck="false"
        autocomplete="off"
      />
    </label>

    <label class="block space-y-1">
      <span class="text-xs text-slate-400">API key <span class="text-slate-600">(only if the server sets <code>server.api_keys</code>)</span></span>
      <div class="flex gap-2">
        <input
          bind:value={apiKey}
          type={showKey ? 'text' : 'password'}
          class="flex-1 rounded-lg bg-slate-950 border border-slate-700 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
          placeholder="none"
          spellcheck="false"
          autocomplete="off"
        />
        <button type="button" class="px-2 text-xs text-slate-400 hover:text-slate-200" onclick={() => (showKey = !showKey)}>
          {showKey ? 'Hide' : 'Show'}
        </button>
      </div>
      <span class="block text-[11px] text-slate-500">
        {inTauri ? 'Kept in the system keychain, one key per server.' : 'Kept in memory for this session only (browser mode).'}
      </span>
    </label>

    {#if result}
      <p class="text-xs {result.tone}">{result.text}</p>
    {/if}

    <div class="flex items-center gap-2 pt-1">
      <button type="button" class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800 disabled:opacity-50" onclick={test} disabled={testing}>
        {testing ? 'Testing…' : 'Test connection'}
      </button>
      <span class="flex-1"></span>
      <button type="button" class="px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-slate-200" onclick={onclose}>Cancel</button>
      <button type="submit" class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50" disabled={saving}>Save</button>
    </div>
  </form>
  </div>
</div>
