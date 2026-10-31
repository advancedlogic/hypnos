<script>
  import { onMount, untrack } from 'svelte';
  import * as api from './hypnos.js';
  import { normalizeUrl } from './connection.js';

  /** `routes`: modality -> model id; `models`: the enabled models with their `modalities`. */
  let { connection, routes, models, onsave, onclose } = $props();

  // The form starts from the current connection and is applied on Save.
  const initial = untrack(() => ({ ...connection, routes: { ...routes } }));
  let url = $state(initial.url);
  let apiKey = $state(initial.apiKey);
  let chosen = $state(initial.routes);
  const prefs = api.voicePrefs();
  let voice = $state(prefs.voice);
  let speed = $state(prefs.speed);
  let imageTool = $state(api.imageToolOn());
  // Kokoro voices (any id the server knows works; OpenAI names are mapped).
  const VOICES = ['af_heart', 'af_bella', 'af_nova', 'af_sarah', 'am_adam', 'am_michael', 'bf_emma', 'bm_george'];

  // The jobs the app hands to a model besides the chat itself (DESK-7..11 use them).
  const JOBS = [
    ['vision', 'Images in messages'],
    ['speech_to_text', 'Speech to text'],
    ['text_to_speech', 'Text to speech'],
    ['image_generation', 'Image generation'],
    ['document_conversion', 'Documents and URLs'],
  ];
  const servingModels = (modality) => models.filter((m) => m.modalities.includes(modality));
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
      const outcome = await api.test(normalized, apiKey);
      const r = RESULTS[outcome.state];
      result = { ...r, text: outcome.health ? `${r.text} (Hypnos ${outcome.health.version})` : r.text };
    } catch (e) {
      result = { text: api.errorMessage(e), tone: 'text-rose-300' };
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
      const picked = Object.fromEntries(Object.entries(chosen).filter(([, id]) => id));
      api.setVoicePrefs({ voice: voice.trim(), speed: Number(speed) });
      api.setImageTool(imageTool);
      await onsave({ url: normalized, apiKey: apiKey.trim() }, picked);
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
        Kept in the system keychain, one key per server.
      </span>
    </label>

    {#if models.length}
      <div class="space-y-2 pt-1">
        <h3 class="text-xs text-slate-400">Models for each job <span class="text-slate-600">(automatic picks the chat's model when it can, else one already loaded)</span></h3>
        <div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 items-center">
          {#each JOBS as [modality, label] (modality)}
            <label for="route-{modality}" class="text-xs text-slate-400">{label}</label>
            <select
              id="route-{modality}"
              bind:value={chosen[modality]}
              class="rounded-md bg-slate-950 border border-slate-700 px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              <option value={undefined}>Automatic</option>
              {#each servingModels(modality) as m (m.id)}
                <option value={m.id}>{m.id}</option>
              {:else}
                <option value={undefined} disabled>No model serves this</option>
              {/each}
            </select>
          {/each}
        </div>
      </div>
    {/if}

    <label class="flex items-center gap-2 pt-1 text-xs text-slate-400">
      <input type="checkbox" bind:checked={imageTool} class="accent-indigo-500" />
      Let the chat model create images (when the server has an image model)
    </label>

    <div class="space-y-2 pt-1">
      <h3 class="text-xs text-slate-400">Reading aloud</h3>
      <div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 items-center">
        <label for="voice" class="text-xs text-slate-400">Voice</label>
        <input id="voice" list="voices" bind:value={voice} placeholder="server default"
          class="rounded-md bg-slate-950 border border-slate-700 px-2 py-1 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500" />
        <datalist id="voices">{#each VOICES as v (v)}<option value={v}></option>{/each}</datalist>
        <label for="speed" class="text-xs text-slate-400">Speed</label>
        <div class="flex items-center gap-2">
          <input id="speed" type="range" min="0.7" max="1.5" step="0.05" bind:value={speed} class="flex-1 accent-indigo-500" />
          <span class="w-10 text-right font-mono text-xs text-slate-400">{Number(speed).toFixed(2)}×</span>
        </div>
      </div>
    </div>

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
