<script>
  import { onMount, untrack } from 'svelte';
  import * as api from './hypnos.js';
  import * as models from './models.js';
  import * as setup from './setup.js';
  import { formatMs } from './format.js';
  import logo from '../assets/logo.png';

  /**
   * The guided setup (DESK-18): what Hypnos is for, what this machine holds, the recommended
   * models, their download, then a short demonstration. `config` / `binary`: the local server's;
   * `firstRun` adds the welcome; `onstop` stops the app's local server (before its config is
   * rewritten), `onrestart` (re)starts it and connects to it (throws on failure); `onpool` after
   * models were added to a running server; `onmodels` opens the Models dialog instead;
   * `serverExited`: the app's local server stopped by itself (a refused config, a crash).
   */
  let { config, binary = null, firstRun = false, serverExited = false, onstop, onrestart, onpool, onmodels, onclose } = $props();

  let step = $state('choose'); // 'choose' | 'install' | 'demo'
  let needs = $state(new Set(setup.DEFAULT_NEEDS));
  let advice = $state(null);
  let adviceError = $state(null);
  let advising = $state(false);
  let busy = $state(null); // what runs now, in words
  let progress = $state(null); // { done, total, file, verifying }
  let error = $state(null);
  let path = $state(null); // { kind: 'config' | 'live' | 'copy', text, steps? }
  let copied = $state(false);
  let ready = $state(false);
  let waited = $state(0);
  let demo = $state([]); // [{ id, title, status, ms, text, error, extra }]
  let demoRunning = $state(false);
  let closed = false;

  // Advice for the current needs, again after each change (it takes a fraction of a second).
  let token = 0;
  async function refreshAdvice() {
    const mine = ++token;
    adviceError = null;
    if (!needs.size) {
      advice = null;
      adviceError = 'Choose at least one thing to use Hypnos for.';
      return;
    }
    advising = true;
    try {
      const next = await setup.advise(binary, config, [...needs]);
      if (mine === token) advice = next;
    } catch (e) {
      if (mine === token) {
        advice = null;
        adviceError = String(e?.message ?? e);
      }
    } finally {
      if (mine === token) advising = false;
    }
  }

  onMount(() => {
    untrack(refreshAdvice);
    return () => (closed = true);
  });

  function toggleNeed(id) {
    const next = new Set(needs);
    next.has(id) ? next.delete(id) : next.add(id);
    needs = next;
    refreshAdvice();
  }

  let tooBig = $derived(advice?.freeBytes != null && advice.downloadBytes > advice.freeBytes);
  let chatModel = $derived(advice?.models.find((m) => m.serves.includes('chat'))?.id ?? null);
  let visionModel = $derived(advice?.models.find((m) => m.serves.includes('vision'))?.id ?? null);
  let hasVoice = $derived(
    !!advice && ['voice', 'speech', 'transcription'].some((n) => advice.models.some((m) => m.serves.includes(n))),
  );

  function pct(done, total) {
    return total ? Math.min(100, (100 * done) / total) : 0;
  }

  /** Download, then put the lineup to work: the app's config, the running server, or a copy. */
  async function install() {
    error = null;
    path = null;
    step = 'install';
    busy = 'Downloading the models…';
    progress = { done: 0, total: advice.downloadBytes || null, file: null, verifying: false };
    let finished = false;
    try {
      finished = await setup.install(config, advice.specs, (e) => {
        if (e.event === 'start') progress.total = e.totalBytes;
        else if (e.event === 'file') {
          progress.file = e.local;
          progress.verifying = false;
        } else if (e.event === 'progress') progress.done = e.doneBytes;
        else if (e.event === 'verifying') progress.verifying = true;
      });
    } catch (e) {
      error = `The download failed: ${e?.message ?? e}`;
    }
    progress = null;
    if (!finished) {
      busy = null;
      if (!error) error = 'Stopped. What was downloaded is kept; Install again resumes it.';
      return;
    }
    try {
      if (advice.managed) await applyToConfig();
      else await applyElsewhere();
    } catch (e) {
      error = String(e?.message ?? e);
    } finally {
      busy = null;
    }
    if (path && path.kind !== 'copy' && !error) await waitUntilReady();
  }

  async function applyToConfig() {
    busy = 'Setting up the server…';
    await onstop?.();
    const applied = await setup.applyConfig(config, advice.config);
    busy = 'Starting the server…';
    path = {
      kind: 'config',
      text:
        `The app's configuration now holds these ${applied.models.length} models; the previous one is saved as ` +
        `${applied.backup.split(/[\\/]/).pop()}` +
        (applied.overlayBackup ? ', and the models added earlier while the server ran were set aside with it' : '') +
        '. The local server restarted with them.',
    };
    await onrestart?.();
  }

  async function applyElsewhere() {
    busy = 'Adding the models to the running server…';
    let steps;
    try {
      steps = await setup.applyLive(config, advice.specs);
    } catch (e) {
      path = {
        kind: 'copy',
        text:
          `The models are downloaded. This configuration (${config.split(/[\\/]/).pop()}) is yours, so the app does not ` +
          `change it, and they could not be added to a running server (${e?.message ?? e}). Copy the recommended ` +
          'configuration below into it (or start the server and run the setup again), then restart the server.',
      };
      return;
    }
    const n = (a) => steps.filter((s) => s.action === a).length;
    path = {
      kind: 'live',
      steps,
      text:
        `Added to the running server without a restart: ${n('added')} added` +
        (n('present') ? `, ${n('present')} it already had` : '') +
        (n('failed') ? `, ${n('failed')} failed` : '') +
        '. Your configuration file was not changed; the server remembers the additions.',
    };
    await onpool?.();
  }

  async function waitUntilReady() {
    busy = 'Waiting for the server to load the models…';
    const started = Date.now();
    waited = 0;
    while (!closed && Date.now() - started < 10 * 60_000) {
      try {
        const p = await api.probe();
        if (p.state === 'ready') {
          ready = true;
          break;
        }
        if (p.state === 'unauthorized') throw new Error('The server refused the API key: set it in the connection settings.');
        if (serverExited) throw new Error('The local server stopped: its log (the server button in the header) says why.');
      } catch (e) {
        if (e instanceof Error) {
          error = e.message;
          break;
        }
      }
      waited = Math.round((Date.now() - started) / 1000);
      await new Promise((r) => setTimeout(r, 1000));
    }
    busy = null;
    if (!ready && !error && !closed) error = 'The server did not get ready within 10 minutes: see the local server log.';
    if (ready) runDemo();
  }

  function stepOf(id) {
    return demo.find((s) => s.id === id);
  }

  /** One chat request; resolves to the answer, filling the step as it streams. */
  async function ask(s, model, message) {
    const started = performance.now();
    let first = null;
    s.text = '';
    await api.chat(`setup-${s.id}-${Date.now()}`, model, [message], {
      onModel: (id) => (s.extra = { ...(s.extra ?? {}), model: id }),
      onDelta: (t) => {
        first ??= performance.now() - started;
        s.text += t;
      },
      onDone: () => {},
    });
    s.ms = performance.now() - started;
    s.first = first;
    if (!s.text.trim()) throw new Error('The model gave an empty answer.');
    return s.text;
  }

  async function runStep(id, fn) {
    const s = stepOf(id);
    s.status = 'running';
    try {
      await fn(s);
      s.status = 'ok';
    } catch (e) {
      s.status = 'failed';
      s.error = api.errorMessage(e);
    }
  }

  async function runDemo() {
    step = 'demo';
    demoRunning = true;
    demo = [
      { id: 'chat', title: 'Chat: a question', status: 'pending' },
      ...(visionModel ? [{ id: 'vision', title: 'Vision: describe the Hypnos logo', status: 'pending' }] : []),
      ...(hasVoice ? [{ id: 'voice', title: 'Voice: speak the answer, then listen to it', status: 'pending' }] : []),
    ];
    let answer = null;
    await runStep('chat', async (s) => {
      const model = await api.route('text_generation', chatModel);
      if (!model) throw new Error('No chat model is enabled on the server.');
      s.extra = { model };
      answer = await ask(s, model, {
        role: 'user',
        content: 'In two short sentences, say hello and tell me what you can help me with.',
      });
    });
    if (visionModel) {
      await runStep('vision', async (s) => {
        const bytes = new Uint8Array(await (await fetch(logo)).arrayBuffer());
        const image = await api.attachFromBytes('hypnos-logo.png', bytes, visionModel);
        s.extra = { model: visionModel, image: image.dataUrl };
        await ask(s, visionModel, {
          role: 'user',
          content: 'Describe this image in one sentence.',
          attachments: [image],
        });
      });
    }
    if (hasVoice) {
      await runStep('voice', async (s) => {
        const v = await setup.voiceCheck(answer ?? 'Hello from Hypnos. Everything is set up and ready.');
        s.voice = v;
        s.ms = v.speakMs + v.listenMs;
      });
    }
    demoRunning = false;
  }

  async function copyConfig() {
    await navigator.clipboard.writeText(advice.config);
    copied = true;
  }

  function onKeydown(e) {
    if (e.key === 'Escape' && !busy && !demoRunning) onclose();
  }

  const STATUS = {
    pending: 'text-slate-600',
    running: 'text-amber-300',
    ok: 'text-emerald-400',
    failed: 'text-rose-400',
  };
</script>

<svelte:window onkeydown={onKeydown} />

<div class="fixed inset-0 z-30 bg-black/60 flex items-center justify-center p-4" role="presentation"
  onclick={(e) => e.target === e.currentTarget && !busy && !demoRunning && onclose()}>
  <div role="dialog" aria-modal="true" aria-labelledby="setup-title"
    class="w-full max-w-2xl max-h-full flex flex-col rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
    <div class="p-5 pb-3 shrink-0 space-y-2">
      <div class="flex items-center gap-3">
        <h2 id="setup-title" class="text-sm font-semibold">Set up Hypnos</h2>
        <ol class="flex items-center gap-1.5 text-[11px] text-slate-500">
          {#each [['choose', 'Choose'], ['install', 'Install'], ['demo', 'Try it']] as [id, label], i (id)}
            <li class="flex items-center gap-1.5">
              {#if i}<span class="text-slate-700">›</span>{/if}
              <span class={step === id ? 'text-indigo-300' : ''}>{label}</span>
            </li>
          {/each}
        </ol>
        <button class="ml-auto text-slate-500 hover:text-slate-200 disabled:opacity-30" onclick={onclose}
          disabled={!!busy || demoRunning} aria-label="Close">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18" /></svg>
        </button>
      </div>
      {#if firstRun && step === 'choose'}
        <p class="text-xs text-slate-300">
          Welcome. Hypnos runs AI models on this computer, nothing leaves it. Tell it what you want to do and it picks
          models that fit this machine, downloads them and shows you that they work.
        </p>
      {/if}
    </div>

    <div class="flex-1 min-h-0 overflow-y-auto border-t border-slate-800 px-5 py-4 space-y-4">
      {#if step === 'choose'}
        <section class="space-y-2">
          <h3 class="text-[11px] uppercase tracking-wide text-slate-500">What do you want to use it for?</h3>
          <div class="grid grid-cols-2 gap-2">
            {#each setup.NEEDS as n (n.id)}
              <label class="flex items-start gap-2 rounded-lg border px-3 py-2 cursor-pointer text-sm
                {needs.has(n.id) ? 'border-indigo-700 bg-indigo-950/40' : 'border-slate-800 hover:border-slate-700'}">
                <input type="checkbox" class="mt-0.5 accent-indigo-500" checked={needs.has(n.id)} onchange={() => toggleNeed(n.id)} />
                <span>
                  <span class="block text-slate-200">{n.label}</span>
                  <span class="block text-[11px] text-slate-500">{n.hint}</span>
                </span>
              </label>
            {/each}
          </div>
        </section>

        {#if advice}
          <section class="space-y-1">
            <h3 class="text-[11px] uppercase tracking-wide text-slate-500">This computer</h3>
            <p class="text-xs text-slate-300">
              {#if advice.budgets.unified_memory}
                {setup.mem(advice.ramBytes)} of memory, shared by the processor and the graphics. Up to
                {setup.mem(advice.budgets.tier1_bytes)} of it can hold models for fast answers.
              {:else}
                {#if advice.ramBytes}{setup.mem(advice.ramBytes)} of memory and a{:else}A{/if} graphics card with about
                {setup.mem(advice.budgets.tier1_bytes)} for models; up to {setup.mem(advice.budgets.tier2_bytes)} of memory
                keeps other models ready to switch to.
              {/if}
            </p>
          </section>

          <section class="space-y-2">
            <h3 class="text-[11px] uppercase tracking-wide text-slate-500">Recommended for you</h3>
            <ul class="rounded-lg border border-slate-800 divide-y divide-slate-800">
              {#each advice.models as m (m.id)}
                <li class="px-3 py-2 text-sm space-y-1">
                  <div class="flex items-center gap-2 flex-wrap">
                    <span class="text-slate-200">{m.role}:</span>
                    <span class="text-slate-300 font-medium">{m.name}</span>
                    <span class="text-xs text-slate-500 tabular-nums">{models.gb(m.sizeBytes)}</span>
                    {#if m.installed}
                      <span class="text-[10px] px-1.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-900">Downloaded</span>
                    {/if}
                  </div>
                  <div class="flex items-center gap-1.5 flex-wrap text-[10px]">
                    {#if m.runsOnCpu}
                      <span class="px-1.5 rounded bg-slate-800 text-slate-400">Runs on the processor</span>
                    {:else if m.partial}
                      <span class="px-1.5 rounded bg-amber-950 text-amber-300">Partly in GPU memory: slower</span>
                    {:else if m.keptLoaded}
                      <span class="px-1.5 rounded bg-slate-800 text-slate-300">Kept in GPU memory</span>
                    {:else}
                      <span class="px-1.5 rounded bg-slate-800 text-slate-400" title="About {m.loadSecs} s to load">Loads when needed</span>
                    {/if}
                    {#if m.smaller}
                      <span class="px-1.5 rounded bg-slate-800 text-slate-400">Smaller, faster version</span>
                    {/if}
                  </div>
                </li>
              {/each}
            </ul>
            <p class="text-xs {advice.together ? 'text-emerald-300' : 'text-amber-300'}">
              {#if advice.together}
                The models you talk to stay loaded together, so there is no waiting when you switch between them.
              {:else}
                The models you talk to do not all fit at once: they take turns, and a switch takes a few seconds.
              {/if}
            </p>
            {#each advice.warnings as w}
              <p class="text-xs text-amber-300">{w}</p>
            {/each}
            <p class="text-xs tabular-nums {tooBig ? 'text-rose-300' : 'text-slate-400'}">
              {#if advice.downloadBytes}
                To download: {models.gb(advice.downloadBytes)}{advice.downloadBytes < advice.totalBytes ? ` (of ${models.gb(advice.totalBytes)}; the rest is here already)` : ''}.
              {:else}
                Everything is downloaded already.
              {/if}
              {#if advice.freeBytes != null}
                {models.gb(advice.freeBytes)} free in the models folder{tooBig ? ': not enough space' : ''}.
              {/if}
            </p>
            {#if !advice.managed}
              <p class="text-xs text-slate-500">
                This server uses your own configuration ({config.split(/[\\/]/).pop()}), which the app does not change: the
                models are added to the running server, or you get the configuration to copy.
              </p>
            {/if}
          </section>
        {:else if adviceError}
          <p class="text-xs text-rose-300">{adviceError}</p>
        {:else}
          <p class="text-xs text-slate-500">Looking at this computer…</p>
        {/if}
      {:else if step === 'install'}
        <section class="space-y-3">
          <ul class="text-sm space-y-1">
            {#each advice.models as m (m.id)}
              <li class="flex gap-2"><span class="text-slate-400">{m.role}:</span><span class="text-slate-200">{m.name}</span>
                <span class="ml-auto text-xs text-slate-500 tabular-nums">{models.gb(m.sizeBytes)}</span></li>
            {/each}
          </ul>
          {#if progress}
            <div class="space-y-1">
              <div class="h-1.5 rounded bg-slate-800 overflow-hidden">
                <div class="h-full bg-indigo-500 transition-all" style="width: {pct(progress.done, progress.total)}%"></div>
              </div>
              <p class="text-xs text-slate-400 tabular-nums flex gap-2">
                <span class="truncate">{progress.verifying ? `Checking ${progress.file}…` : (progress.file ?? 'Starting…')}</span>
                <span class="ml-auto shrink-0">{models.gb(progress.done)}{progress.total ? ` of ${models.gb(progress.total)}` : ''}</span>
              </p>
            </div>
          {/if}
          {#if busy}
            <p class="text-xs text-slate-300">{busy}{waited && busy.startsWith('Waiting') ? ` ${waited} s` : ''}</p>
          {/if}
          {#if path}
            <p class="text-xs text-emerald-300">{path.text}</p>
            {#if path.steps}
              <ul class="text-xs space-y-0.5">
                {#each path.steps as s (s.spec)}
                  <li class={s.action === 'failed' ? 'text-rose-300' : 'text-slate-400'}>
                    {s.id}: {s.action === 'added' ? 'added' : s.action === 'present' ? 'already there' : `failed: ${s.message}`}
                  </li>
                {/each}
              </ul>
            {/if}
            {#if path.kind === 'copy'}
              <div class="relative">
                <pre class="text-[11px] font-mono text-slate-300 bg-slate-950 border border-slate-800 rounded p-2 max-h-64 overflow-auto">{advice.config}</pre>
                <button class="absolute top-1.5 right-1.5 text-[11px] px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800"
                  onclick={copyConfig}>{copied ? 'Copied' : 'Copy'}</button>
              </div>
            {/if}
          {/if}
        </section>
      {:else}
        <section class="space-y-3">
          <p class="text-xs text-slate-400">A short check that each part works, with the models just installed.</p>
          <ul class="space-y-3">
            {#each demo as s (s.id)}
              <li class="rounded-lg border border-slate-800 px-3 py-2 space-y-1.5">
                <div class="flex items-center gap-2 text-sm">
                  <span class="{STATUS[s.status]} {s.status === 'running' ? 'animate-pulse' : ''}">●</span>
                  <span class="text-slate-200">{s.title}</span>
                  {#if s.extra?.model}<span class="font-mono text-[11px] text-slate-500">{s.extra.model}</span>{/if}
                  <span class="ml-auto text-xs text-slate-500 tabular-nums">
                    {#if s.status === 'ok' && s.ms != null}
                      {#if s.first != null}first words after {formatMs(s.first)}, {/if}{formatMs(s.ms)}
                    {:else if s.status === 'pending'}waiting{/if}
                  </span>
                </div>
                {#if s.extra?.image}
                  <img src={s.extra.image} alt="The Hypnos logo" class="w-12 h-12 rounded bg-white p-1" />
                {/if}
                {#if s.text}
                  <p class="text-xs text-slate-300 whitespace-pre-wrap">{s.text}</p>
                {/if}
                {#if s.voice}
                  <p class="text-xs text-slate-400">
                    Spoken by <span class="font-mono">{s.voice.speechModel}</span> in {formatMs(s.voice.speakMs)}
                    ({s.voice.audioSecs.toFixed(1)} s of audio), heard back by
                    <span class="font-mono">{s.voice.listenModel}</span> in {formatMs(s.voice.listenMs)}:
                  </p>
                  <p class="text-xs text-slate-300 italic">“{s.voice.heard}”</p>
                  <div class="flex items-center gap-3">
                    <audio controls src={s.voice.audio} class="h-8"></audio>
                    <span class="text-xs {s.voice.share >= 0.8 ? 'text-emerald-300' : 'text-amber-300'}">
                      {Math.round(s.voice.share * 100)}% of the words heard back
                    </span>
                  </div>
                {/if}
                {#if s.error}
                  <p class="text-xs text-rose-300">{s.error}</p>
                {/if}
              </li>
            {/each}
          </ul>
          {#if !demoRunning}
            <p class="text-xs {demo.every((s) => s.status === 'ok') ? 'text-emerald-300' : 'text-amber-300'}">
              {#if demo.every((s) => s.status === 'ok')}
                All set. Close this and start a chat.
              {:else}
                Some parts did not work; the local server log (the server button in the header) says why.
              {/if}
            </p>
          {/if}
        </section>
      {/if}
    </div>

    <div class="p-5 pt-3 space-y-2 border-t border-slate-800 shrink-0">
      {#if error}
        <p class="text-xs text-rose-300">{error}</p>
      {/if}
      <div class="flex items-center gap-3">
        {#if step === 'choose'}
          <button class="text-xs text-slate-400 hover:text-slate-200" onclick={onmodels}>Choose models myself</button>
          <span class="ml-auto"></span>
          <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800" onclick={onclose}>Not now</button>
          <button class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
            onclick={install} disabled={!advice || advising || tooBig || !binary}>
            {advice?.downloadBytes ? `Download ${models.gb(advice.downloadBytes)} and set up` : 'Set up'}
          </button>
        {:else if step === 'install'}
          {#if progress}
            <span class="ml-auto"></span>
            <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800" onclick={models.cancel}>Stop</button>
          {:else if !busy}
            <button class="text-xs text-slate-400 hover:text-slate-200" onclick={() => { step = 'choose'; error = null; path = null; }}>Back</button>
            <span class="ml-auto"></span>
            {#if path?.kind === 'copy' || (!path && error)}
              <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800" onclick={onclose}>Close</button>
            {/if}
            {#if !path && error}
              <button class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500" onclick={install}>Try again</button>
            {:else if path && path.kind !== 'copy' && !ready}
              <button class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500" onclick={() => { error = null; waitUntilReady(); }}>Wait again</button>
            {/if}
          {/if}
        {:else}
          <span class="ml-auto"></span>
          {#if !demoRunning}
            <button class="px-3 py-1.5 rounded-lg text-sm border border-slate-700 hover:bg-slate-800" onclick={runDemo}>Run again</button>
            <button class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500" onclick={onclose}>Done</button>
          {:else}
            <span class="text-xs text-slate-500">Checking…</span>
          {/if}
        {/if}
      </div>
    </div>
  </div>
</div>
