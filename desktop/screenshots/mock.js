// Stands in for the app's Rust side so the web UI runs in a plain browser, for the user
// manual's screenshots (docs/desktop-manual.md). Server data (/health, /v1/memory/status, the
// catalog, the request log) comes from a real Hypnos at `?server=` (default
// http://localhost:9988); the chats are the recorded conversations in fixtures.json. `?scene=`
// picks which chat is open (dolomites, document, tools, image). See capture.mjs.
import { mockIPC, mockWindows } from '@tauri-apps/api/mocks';
import fixtures from './fixtures.json';
// `hypnos advise --needs chat,vision,voice --ram 32GB --unified --json`: the guided setup on a 32 GB Mac.
import advice from './advice.json';

const params = new URLSearchParams(location.search);
const SERVER = params.get('server') ?? 'http://localhost:9988';
const SCENE = params.get('scene') ?? 'dolomites';

const get = async (path) => {
  const r = await fetch(`${SERVER}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
};

async function dataUrl(path) {
  const blob = await (await fetch(path)).blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

const minute = 60_000;
const at = (minutesAgo) => Date.now() - minutesAgo * minute;
const CHAT_MODEL = 'gemma4:26b-q4';

/** The recorded conversations as the ChatStore keeps them. */
async function chats() {
  const f = fixtures;
  const [day, night] = await Promise.all([dataUrl('/screenshots/assets/sign-day.jpg'), dataUrl('/screenshots/assets/sign-night.jpg')]);
  const image = (shot, dataUrl) => ({
    dataUrl,
    prompt: shot.prompt,
    size: shot.size,
    model: shot.model,
    seconds: shot.seconds,
    telemetry: null,
    seed: shot.seed,
    negativePrompt: null,
    steps: null,
    reference: shot.reference,
    more: [],
    note: null,
  });
  const imageStep = (id, shot) => ({
    id,
    tool: 'generate_image',
    label: shot.reference === 'none' ? 'Creating an image' : 'Editing an image',
    status: 'done',
    model: shot.model,
    seconds: shot.seconds,
    summary: `${shot.size}, seed ${shot.seed}`,
    input: { prompt: shot.prompt, size: shot.size, reference: shot.reference },
    output: `The image was generated with: prompt "${shot.prompt}", size ${shot.size}, seed ${shot.seed}, model ${shot.model}. It is now shown to the user.`,
  });
  const list = {
    dolomites: {
      id: 'c-dolomites',
      title: 'Packing for the Dolomites',
      model: CHAT_MODEL,
      system: '',
      updatedAt: at(3),
      messages: [
        { role: 'user', content: f.dolomites.question },
        {
          role: 'assistant',
          content: f.dolomites.answer,
          model: CHAT_MODEL,
          at: at(3),
          reasoning: f.dolomites.reasoning,
          thoughtSeconds: 6.4,
          telemetry: f.dolomites.telemetry,
          usage: f.dolomites.usage,
        },
      ],
    },
    document: {
      id: 'c-document',
      title: 'Summarize this document in five bullet points…',
      model: CHAT_MODEL,
      system: '',
      updatedAt: at(25),
      messages: [
        {
          role: 'user',
          content: f.document.question,
          attachments: [
            { kind: 'document', name: f.document.name, size: f.document.size, text: f.document.text, pages: f.document.pages, truncated: false },
          ],
        },
        { role: 'assistant', content: f.document.answer, model: CHAT_MODEL, at: at(25), telemetry: f.document.telemetry, usage: f.document.usage },
      ],
    },
    tools: {
      id: 'c-tools',
      title: 'Two things. First, read https://example.com…',
      model: CHAT_MODEL,
      system: '',
      updatedAt: at(60),
      messages: [
        { role: 'user', content: f.tools.question },
        { role: 'assistant', content: f.tools.answer, model: CHAT_MODEL, at: at(60), steps: f.tools.steps, telemetry: f.dolomites.telemetry },
      ],
    },
    image: {
      id: 'c-image',
      title: 'Make a photo of a weathered wooden sign…',
      model: CHAT_MODEL,
      system: '',
      updatedAt: at(120),
      messages: [
        { role: 'user', content: f.image.question },
        {
          role: 'assistant',
          content: f.image.first.comment,
          model: CHAT_MODEL,
          at: at(125),
          steps: [imageStep('img-1', f.image.first)],
          image: image(f.image.first, day),
        },
        { role: 'user', content: f.image.followup },
        {
          role: 'assistant',
          content: f.image.second.comment,
          model: CHAT_MODEL,
          at: at(120),
          steps: [imageStep('img-2', f.image.second)],
          image: image(f.image.second, night),
        },
      ],
    },
  };
  const older = [
    ['Rewrite the release notes for v0.6', 60 * 24],
    ['Regex for ISO dates', 60 * 26],
    ['Transcribe the standup recording', 60 * 50],
    ['Italian phrases for the trip', 60 * 75],
  ].map(([title, ago], i) => ({
    id: `c-old-${i}`,
    title,
    model: CHAT_MODEL,
    system: '',
    updatedAt: at(ago),
    messages: [{ role: 'user', content: title }, { role: 'assistant', content: '…', model: CHAT_MODEL, at: at(ago) }],
  }));
  // The scene's chat is the newest, so it opens.
  const all = [list.dolomites, list.document, list.tools, list.image, ...older];
  const first = list[SCENE] ?? list.dolomites;
  first.updatedAt = Date.now();
  return all;
}

/** The app's store (tauri-plugin-store) for this run. */
const store = new Map();

async function seedStore() {
  store.set('serverUrl', 'http://localhost:9999');
  store.set('chats', await chats());
  store.set('lastModel', CHAT_MODEL);
  store.set('thinking', { [CHAT_MODEL]: SCENE === 'dolomites' ? 'low' : 'none' });
  store.set('routes', { image_generation: 'flux3-image-klein' });
  store.set('modelsPrompted', true);
  store.set('localServer', {
    binary: '/Applications/Hypnos.app/Contents/MacOS/hypnos',
    config: '/Users/you/Library/Application Support/Hypnos/hypnos.toml',
    port: 9999,
    autoStart: true,
  });
}

const LOCAL_STATUS = {
  running: true,
  pid: 48213,
  options: { binary: '/Applications/Hypnos.app/Contents/MacOS/hypnos', config: '/Users/you/Library/Application Support/Hypnos/hypnos.toml', port: 9999 },
  exitCode: null,
  exited: false,
};

/** The server's lines for the Local server dialog: the head of its real log, ANSI codes removed. */
const LOCAL_LOG = [
  'INFO hypnos: Hypnos starting config=/Users/you/Library/Application Support/Hypnos/hypnos.toml version="0.7.0"',
  'INFO hypnos: Model files models_dir=/Users/you/Library/Application Support/Hypnos/models',
  'INFO hypnos_memory::hardware: Initialized hardware memory topology accelerator=AppleMetalUma topology=UnifiedMemory usable_vram_mb=97792',
  'INFO hypnos_memory::manager: Unified memory: Tier 1 budget set from the GPU\'s recommended working set device_working_set_mb=110100',
  'INFO hypnos: Hypnos 3-Tier Memory Coordinator ready accelerator=AppleMetalUma topology=UnifiedMemory models_registered=8',
  'INFO hypnos: listening addr=127.0.0.1:9999',
  'INFO hypnos_scheduler: Pre-staged model model=kokoro-82m-q4 tier=Tier1GpuVram',
  'INFO hypnos_scheduler: Pre-staged model model=gemmaembedding2 tier=Tier1GpuVram',
  'INFO hypnos_scheduler: Pre-staged model model=gemma4:26b-q4 tier=Tier2PinnedRam',
  'INFO hypnos: All startup tiers initialized; server ready',
];

/** The Models dialog's lineup (`models_catalog`): the manifest's models with what is on disk. */
async function lineup() {
  const status = await get('/v1/memory/status');
  const catalog = await get('/v1/models/catalog');
  const inLineup = new Set(catalog.data.filter((e) => e.lineup).map((e) => e.id));
  return {
    modelsDir: '/Users/you/Library/Application Support/Hypnos/models',
    dirExists: true,
    freeBytes: 1652e9,
    managed: true,
    errors: [],
    models: status.models
      .filter((m) => inLineup.has(m.id))
      .map((m) => ({ id: m.id, modality: m.modality, inConfig: true, enabled: m.enabled, size: m.total_weight_bytes, have: m.total_weight_bytes, installed: true, files: 1 })),
  };
}

/** The Catalog tab (`models_catalog_entries`), from the server's catalog. */
async function catalogEntries() {
  const catalog = await get('/v1/models/catalog');
  return {
    live: true,
    offline: null,
    entries: catalog.data.map((e) => ({
      id: e.id,
      name: e.name,
      publisher: e.publisher,
      license: e.license,
      modality: e.modality,
      additionalModalities: e.additional_modalities ?? [],
      engine: e.engine,
      capabilities: e.capabilities ?? [],
      summary: e.summary,
      lineup: !!e.lineup,
      variants: e.variants.map((v) => ({
        quant: v.quant,
        default: !!v.default,
        note: v.note ?? null,
        size: v.size_bytes,
        have: v.installed ? v.size_bytes : 0,
        installed: !!v.installed,
        gated: v.files.some((f) => f.gated),
      })),
      inPool: !!e.configured,
      enabled: e.configured ? e.enabled !== false : null,
      currentQuant: e.configured_quant ?? null,
    })),
  };
}

/** What a model does, in words (`role` in src-tauri/src/setup.rs). */
function role(serves, modality) {
  const parts = [];
  for (const need of serves) {
    const part =
      {
        chat: 'chat',
        vision: 'seeing images',
        embeddings: 'search (finding related text)',
        images: 'making images',
        documents: 'reading documents',
        entities: 'finding names, places and dates',
      }[need] ??
      (modality === 'text_to_speech'
        ? 'speaking answers aloud'
        : modality === 'speech_to_text'
          ? 'listening (speech to text)'
          : need === 'voice'
            ? 'voice conversations'
            : need);
    if (!parts.includes(part)) parts.push(part);
  }
  const kept = parts.includes('chat') ? parts.filter((p) => p !== 'voice conversations') : parts;
  const text = kept.length > 1 ? `${kept.slice(0, -1).join(', ')} and ${kept.at(-1)}` : (kept[0] ?? 'helper');
  return text[0].toUpperCase() + text.slice(1);
}

/** The guided setup's recommendation (`setup_advise`), from the recorded advice and the catalog. */
async function setupAdvice() {
  const catalog = await get('/v1/models/catalog');
  const models = advice.models.map((m) => {
    const entry = catalog.data.find((e) => e.id === m.id);
    const variant = entry?.variants.find((v) => v.quant === m.quant);
    const size = variant?.size_bytes ?? m.weights_bytes;
    // A 32 GB Mac that has not downloaded these yet.
    return {
      id: m.id,
      name: m.name,
      quant: m.quant,
      spec: `${m.id}:${m.quant}`,
      role: role(m.serves, entry?.modality ?? ''),
      serves: m.serves,
      sizeBytes: size,
      missingBytes: size,
      installed: false,
      keptLoaded: m.initial_tier === 'tier1_gpu_vram',
      runsOnCpu: m.runs_on_cpu,
      partial: !m.runs_on_cpu && m.num_layers > 0 && m.gpu_layers < m.num_layers,
      smaller: variant ? !variant.default : false,
      loadSecs: m.load_secs_warm,
    };
  });
  const total = models.reduce((n, m) => n + m.sizeBytes, 0);
  return {
    ramBytes: 32 * 2 ** 30,
    budgets: advice.budgets,
    models,
    together: advice.interactive_coresident,
    warnings: advice.warnings,
    totalBytes: total,
    downloadBytes: total,
    freeBytes: 412e9,
    modelsDir: '/Users/you/Library/Application Support/Hypnos/models',
    managed: true,
    specs: advice.specs,
    config: advice.config,
  };
}

const commands = {
  setup_advise: () => setupAdvice(),
  // The server (src-tauri/src/hypnos.rs): passed through to the real one.
  hypnos_connect: () => null,
  hypnos_set_routes: () => null,
  hypnos_probe: async () => ({ state: 'ready', health: await get('/health') }),
  hypnos_test: async () => ({ state: 'ready', health: await get('/health') }),
  hypnos_models: async () => (await get('/v1/memory/status')).models.filter((m) => m.enabled),
  hypnos_memory_status: () => get('/v1/memory/status'),
  hypnos_logs: ({ filter }) => {
    const q = new URLSearchParams();
    for (const k of ['model', 'route', 'status', 'limit']) if (filter?.[k]) q.set(k, filter[k]);
    return get(`/v1/logs?${q}`);
  },
  hypnos_log: ({ id }) => get(`/v1/logs/${id}`),
  hypnos_route: () => CHAT_MODEL,
  // The app's own state.
  get_api_key: () => null,
  set_api_key: () => null,
  hf_token_state: () => ({ saved: true, fromEnv: false }),
  tray_update: () => null,
  tray_settings: () => null,
  local_server_detect: () => ({ binary: LOCAL_STATUS.options.binary, config: LOCAL_STATUS.options.config, source: 'bundled' }),
  local_server_status: () => LOCAL_STATUS,
  local_server_probe: () => [true, true],
  local_server_logs: () => LOCAL_LOG,
  models_catalog: () => lineup(),
  models_catalog_entries: () => catalogEntries(),
  'plugin:store|load': () => 1,
  'plugin:store|get_store': () => 1,
  'plugin:store|get': ({ key }) => [store.get(key) ?? null, store.has(key)],
  'plugin:store|set': ({ key, value }) => {
    store.set(key, value);
    return null;
  },
  'plugin:store|has': ({ key }) => store.has(key),
  'plugin:store|save': () => null,
  'plugin:notification|is_permission_granted': () => true,
};

mockWindows('main');
mockIPC(
  async (cmd, args) => {
    if (params.has('trace')) console.log('ipc', cmd, JSON.stringify(args ?? {}).slice(0, 120));
    const handler = commands[cmd];
    if (handler) {
      const out = await handler(args ?? {});
      if (params.has('trace')) console.log('ipc ->', cmd, JSON.stringify(out)?.slice(0, 80));
      return out;
    }
    console.warn('screenshot mock: no answer for', cmd, args);
    return null;
  },
  { shouldMockEvents: true },
);
/** Resolves once the store holds the scene: `main.js` starts the app only then. */
export const ready = seedStore().catch((e) => console.error('screenshot mock: seeding failed', e?.stack ?? e));
window.__mockStore = store;
try {
  localStorage.setItem('hypnos.desktop.panelOpen', params.get('panel') ?? 'true');
} catch {}
