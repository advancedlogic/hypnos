<script>
  // Svelte 5 Runes State
  let backendUrl = $state('http://localhost:8080');
  // API key (server.api_keys), kept in this browser only. Storage can be unavailable (private
  // windows, blocked site data): the field then simply starts empty.
  const API_KEY_STORAGE = 'hypnos.apiKey';
  function loadApiKey() {
    try {
      return localStorage.getItem(API_KEY_STORAGE) || '';
    } catch {
      return '';
    }
  }
  let apiKey = $state(loadApiKey());
  $effect(() => {
    try {
      if (apiKey) localStorage.setItem(API_KEY_STORAGE, apiKey);
      else localStorage.removeItem(API_KEY_STORAGE);
    } catch {
      // not persisted; the key still works for this page
    }
  });
  // Set when the backend answers 401: the key is missing or wrong.
  let authRejected = $state(false);

  /** fetch() against the backend with `Authorization: Bearer <key>` when a key is set. */
  async function apiFetch(path, init = {}) {
    const headers = { ...(init.headers || {}) };
    if (apiKey.trim()) headers.Authorization = `Bearer ${apiKey.trim()}`;
    const res = await fetch(`${backendUrl}${path}`, { ...init, headers });
    authRejected = res.status === 401;
    return res;
  }
  let isConnected = $state(false);
  let lastError = $state('');
  let isLoading = $state(false);

  // Memory Hierarchy Status from GET /v1/memory/status
  let memoryStatus = $state(null);

  // Active Playground Tab: 'chat' | 'stt' | 'tts' | 'embeddings' | 'image'
  let activeTab = $state('chat');

  // Telemetry Log History
  let telemetryLog = $state([]);

  // -------------------------------------------------------------------------
  // 1. Chat & Vision Playground State (gemma4:26b-q4 / gemma4:e4b-q4)
  // -------------------------------------------------------------------------
  let chatModel = $state('gemma4:26b-q4');
  let chatPrompt = $state('Explain how Gemma 4 hybrid Sliding Window Attention reduces KV cache VRAM usage.');
  let includeVisionImage = $state(false);
  // 1x1 transparent PNG: lets the vision code path be exercised without a real picture.
  const PLACEHOLDER_IMAGE_URL =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
  let visionImageUrl = $state(PLACEHOLDER_IMAGE_URL);
  // Metadata of an uploaded picture (null when the URL field was typed by hand).
  let visionImageFile = $state(null);
  // Mirrors `max_image_bytes` in backend/hypnos.example.toml (32 MiB).
  const MAX_VISION_IMAGE_BYTES = 32 * 1024 * 1024;
  let chatContextLen = $state(16384);
  let chatMaxTokens = $state(128);
  let chatStream = $state(false);
  let chatMessages = $state([]);

  // -------------------------------------------------------------------------
  // 2. Audio STT Playground State (gemma4:e4b-q4)
  // -------------------------------------------------------------------------
  let sttModel = $state('gemma4:e4b-q4');
  let sttLanguage = $state('en');
  let sttPrompt = $state('Transcribe technical AI discussion');
  let sttSelectedFile = $state(null);
  let sttResult = $state(null);
  // Microphone capture (encoded client-side as 16 kHz mono PCM WAV, which the audio encoder reads directly)
  let sttRecording = $state(false);
  let sttRecordedBlob = $state(null);
  let sttRecordedUrl = $state(null);
  let sttRecordSeconds = $state(0);
  let sttSourceLabel = $state('');
  let micStream = null;
  let micContext = null;
  let micProcessor = null;
  let micChunks = [];
  let micTimer = null;

  // -------------------------------------------------------------------------
  // 3. Text-to-Speech Playground State (kokoro-82m-q4)
  // -------------------------------------------------------------------------
  let ttsModel = $state('kokoro-82m-q4');
  let ttsInput = $state('Hypnos keeps Kokoro 82M pinned in GPU VRAM for zero millisecond voice synthesis.');
  let ttsVoice = $state('alloy');
  let ttsSpeed = $state(1.0);
  let ttsAudioUrl = $state(null);
  let ttsHeadersInfo = $state(null);

  // -------------------------------------------------------------------------
  // 4. Embeddings Playground State (gemmaembedding2)
  // -------------------------------------------------------------------------
  let embedModel = $state('gemmaembedding2');
  let embedInputsRaw = $state(
    '3-tier GPU VRAM and Pinned Host RAM memory manager\nDynamic KV cache budgeting with Sliding Window Attention\nA crispy Margherita pizza baked in a wood-fired oven'
  );
  let embedResult = $state(null);

  // -------------------------------------------------------------------------
  // 5. Image Generation Playground State (moss-image)
  // -------------------------------------------------------------------------
  let imageModel = $state('moss-image');
  let imagePrompt = $state('A futuristic cyberpunk Rust crab orchestrating glowing GPU memory blocks');
  let imageSize = $state('1024x1024');
  let imageCount = $state(1);
  let imageResult = $state(null);

  // -------------------------------------------------------------------------
  // 6. Request Logs (GET /v1/logs): every API call the server recorded
  // -------------------------------------------------------------------------
  const LOG_LEVELS = ['off', 'error', 'warn', 'info', 'debug', 'trace'];
  let logRecords = $state([]);
  let logLevel = $state('info');
  let logCapacity = $state(0);
  let logStored = $state(0);
  let logFilterModel = $state('');
  let logFilterRoute = $state('');
  let logFilterStatus = $state('');
  let logAutoRefresh = $state(true);
  let logSelected = $state(null);
  let logError = $state('');

  // -------------------------------------------------------------------------
  // Derived Computed Metrics
  // -------------------------------------------------------------------------
  let vramTotalBytes = $derived(
    memoryStatus?.capabilities?.usable_vram_budget_bytes || 24 * 1024 * 1024 * 1024
  );
  let vramUsedBytes = $derived(memoryStatus?.tier1_vram_used_bytes || 0);
  let vramUsagePct = $derived(Math.min(100, (vramUsedBytes / vramTotalBytes) * 100));

  let pinnedRamTotalBytes = $derived(
    memoryStatus?.capabilities?.pinned_ram_budget_bytes || 32 * 1024 * 1024 * 1024
  );
  let pinnedRamUsedBytes = $derived(memoryStatus?.tier2_pinned_ram_used_bytes || 0);
  let pinnedRamUsagePct = $derived(
    Math.min(100, (pinnedRamUsedBytes / pinnedRamTotalBytes) * 100)
  );

  // Unified memory (Apple Silicon): Tier 1 and Tier 2 are one pool of physical RAM.
  let isUnifiedMemory = $derived(
    memoryStatus?.capabilities?.topology === 'unified_memory' &&
      (memoryStatus?.capabilities?.physical_memory_bytes || 0) > 0
  );
  let umaPhysicalBytes = $derived(memoryStatus?.capabilities?.physical_memory_bytes || 1);
  let umaResidentPct = $derived(Math.min(100, (vramUsedBytes / umaPhysicalBytes) * 100));
  let umaWarmPct = $derived(
    Math.min(100 - umaResidentPct, (pinnedRamUsedBytes / umaPhysicalBytes) * 100)
  );
  let umaGpuLimitPct = $derived(Math.min(100, (vramTotalBytes / umaPhysicalBytes) * 100));

  // hardware.simulate_physical_memory_bytes: budgets come from a pretend, smaller machine.
  let isSimulated = $derived(memoryStatus?.capabilities?.simulated === true);

  // hardware.max_process_memory_bytes: memory the hypnos process holds vs. its cap.
  let processUsedBytes = $derived(memoryStatus?.process_memory_bytes ?? null);
  let processCapBytes = $derived(memoryStatus?.process_memory_cap_bytes ?? null);
  let processPct = $derived(
    processCapBytes ? Math.min(100, ((processUsedBytes || 0) / processCapBytes) * 100) : 0
  );
  let recentEvictions = $derived((memoryStatus?.recent_evictions || []).slice(0, 5));

  const EVICTION_REASONS = {
    memory_budget: 'did not fit in the memory budget',
    full_gpu_placement: 'to fit all layers on the GPU',
    process_memory_cap: 'process memory cap',
  };

  function timeAgo(unixMs) {
    const secs = Math.max(0, Math.round((Date.now() - unixMs) / 1000));
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    return `${Math.round(secs / 3600)}h ago`;
  }

  let cosineMatrix = $derived.by(() => {
    if (!embedResult?.data || embedResult.data.length < 2) return [];
    const vecs = embedResult.data.map((d) => d.embedding);
    const n = vecs.length;
    const matrix = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let k = 0; k < vecs[i].length; k++) {
          dot += vecs[i][k] * vecs[j][k];
        }
        row.push(dot);
      }
      matrix.push(row);
    }
    return matrix;
  });

  // -------------------------------------------------------------------------
  // Helpers & API Polling
  // -------------------------------------------------------------------------
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 MB';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1.0) return `${gb.toFixed(2)} GiB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MiB`;
  }

  function tierBadgeClass(tier) {
    if (tier === 'tier1_gpu_vram')
      return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
    if (tier === 'tier2_pinned_ram')
      return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
    return 'bg-slate-700/60 text-slate-300 border-slate-600';
  }

  function tierLabel(tier) {
    if (tier === 'tier1_gpu_vram') return isUnifiedMemory ? 'Tier 1 • GPU resident' : 'Tier 1 • GPU VRAM';
    if (tier === 'tier2_pinned_ram') return isUnifiedMemory ? 'Tier 2 • Warm (page cache)' : 'Tier 2 • Pinned RAM';
    return 'Tier 3 • NVMe SSD';
  }

  function recordTelemetry(model, modality, telemetry) {
    if (!telemetry) return;
    telemetryLog = [
      {
        id: crypto.randomUUID(),
        timestamp: new Date().toLocaleTimeString(),
        model,
        modality,
        ...telemetry,
      },
      ...telemetryLog.slice(0, 24),
    ];
  }

  async function fetchMemoryStatus() {
    try {
      const res = await apiFetch(`/v1/memory/status`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      memoryStatus = await res.json();
      isConnected = true;
      lastError = '';
    } catch (err) {
      isConnected = false;
    }
  }

  $effect(() => {
    fetchMemoryStatus();
    const timer = setInterval(fetchMemoryStatus, 2000);
    return () => clearInterval(timer);
  });

  // ---- Request log -------------------------------------------------------
  async function fetchLogs() {
    try {
      const params = new URLSearchParams({ limit: '200' });
      if (logFilterModel) params.set('model', logFilterModel);
      if (logFilterRoute) params.set('route', logFilterRoute);
      if (logFilterStatus) params.set('status', logFilterStatus);
      const res = await apiFetch(`/v1/logs?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      logRecords = data.records;
      logLevel = data.level;
      logCapacity = data.capacity;
      logStored = data.stored;
      logError = '';
    } catch (err) {
      logError = err.message;
    }
  }

  async function openLog(id) {
    try {
      const res = await apiFetch(`/v1/logs/${id}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      logSelected = await res.json();
    } catch (err) {
      logError = err.message;
    }
  }

  async function setLogLevel(level) {
    try {
      const res = await apiFetch(`/v1/logs/level`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      logLevel = (await res.json()).level;
    } catch (err) {
      logError = err.message;
    }
  }

  async function clearLogs() {
    try {
      const res = await apiFetch(`/v1/logs`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      logSelected = null;
      await fetchLogs();
    } catch (err) {
      logError = err.message;
    }
  }

  function logStatusClass(status) {
    if (status >= 500) return 'bg-rose-500/20 text-rose-300';
    if (status >= 400) return 'bg-amber-500/20 text-amber-300';
    return 'bg-emerald-500/20 text-emerald-300';
  }

  function logTime(iso) {
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleTimeString();
  }

  function logNum(v, digits = 0) {
    return v === null || v === undefined ? '–' : Number(v).toFixed(digits);
  }

  function logPretty(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  // Refresh the log while its tab is open (and the filters are part of the request).
  $effect(() => {
    if (activeTab !== 'logs') return;
    fetchLogs();
    if (!logAutoRefresh) return;
    const timer = setInterval(fetchLogs, 3000);
    return () => clearInterval(timer);
  });

  async function manualSetTier(modelId, targetTier) {
    isLoading = true;
    lastError = '';
    try {
      const res = await apiFetch(`/v1/memory/tier`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelId,
          target_tier: targetTier,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);
      await fetchMemoryStatus();
    } catch (err) {
      lastError = err.message;
    } finally {
      isLoading = false;
    }
  }

  // -------------------------------------------------------------------------
  // Action Handlers for the 5 Playground Tabs
  // -------------------------------------------------------------------------
  // Vision image upload: the picture is embedded in the request as a base64 `data:` URL, which
  // is what the backend's `image_url` content part accepts (no server-side upload endpoint needed).
  function onVisionFileChange(e) {
    const file = e.target.files?.[0] || null;
    if (!file) return;
    lastError = '';
    if (!file.type.startsWith('image/')) {
      lastError = `"${file.name}" is not an image file.`;
      e.target.value = '';
      return;
    }
    if (file.size > MAX_VISION_IMAGE_BYTES) {
      lastError = `"${file.name}" is ${formatFileSize(file.size)}; the backend accepts at most ${formatFileSize(MAX_VISION_IMAGE_BYTES)} per image.`;
      e.target.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      visionImageUrl = reader.result;
      visionImageFile = { name: file.name, size: file.size, type: file.type };
      includeVisionImage = true;
    };
    reader.onerror = () => {
      lastError = `Could not read "${file.name}".`;
    };
    reader.readAsDataURL(file);
  }

  function clearVisionImage() {
    visionImageUrl = PLACEHOLDER_IMAGE_URL;
    visionImageFile = null;
    const input = document.getElementById('vision-file');
    if (input) input.value = '';
  }

  // Typing or pasting into the URL field replaces whatever was uploaded.
  function onVisionUrlInput() {
    visionImageFile = null;
  }

  function formatFileSize(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
  }

  // True when the URL field holds something the browser can render as a preview.
  let visionPreviewUrl = $derived(
    includeVisionImage && visionImageUrl && visionImageUrl !== PLACEHOLDER_IMAGE_URL
      ? visionImageUrl
      : ''
  );

  async function runChatCompletion(e) {
    e?.preventDefault();
    if (!chatPrompt.trim()) return;
    if (includeVisionImage && !visionImageUrl.trim()) {
      lastError = 'Attach an image or enter an image URL, or untick "Attach Vision Image".';
      return;
    }
    isLoading = true;
    lastError = '';

    const userContent = includeVisionImage
      ? [
          { type: 'text', text: chatPrompt },
          { type: 'image_url', image_url: { url: visionImageUrl } },
        ]
      : chatPrompt;
    // Keep a thumbnail with the message so the log shows what the model actually saw.
    const sentImage = includeVisionImage ? visionImageUrl : null;

    const payload = {
      model: chatModel,
      messages: [{ role: 'user', content: userContent }],
      max_tokens: Number(chatMaxTokens),
      num_ctx: Number(chatContextLen),
      stream: chatStream,
    };

    try {
      const res = await apiFetch(`/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData?.error?.message || `HTTP ${res.status}`);
      }

      if (chatStream) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let capturedTelemetry = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value);
          for (const line of text.split('\n')) {
            if (line.startsWith('data: ') && line !== 'data: [DONE]') {
              try {
                const chunk = JSON.parse(line.slice(6));
                if (chunk.choices?.[0]?.delta?.content) {
                  accumulated += chunk.choices[0].delta.content;
                }
                if (chunk.hypnos_telemetry) {
                  capturedTelemetry = chunk.hypnos_telemetry;
                }
              } catch {}
            }
          }
        }

        chatMessages = [
          {
            prompt: chatPrompt,
            hasImage: includeVisionImage,
            image: sentImage,
            reply: accumulated,
            model: chatModel,
            telemetry: capturedTelemetry,
          },
          ...chatMessages,
        ];
        recordTelemetry(
          chatModel,
          includeVisionImage ? 'vision' : 'text_generation',
          capturedTelemetry
        );
      } else {
        const data = await res.json();
        const reply = data.choices?.[0]?.message?.content || '';
        chatMessages = [
          {
            prompt: chatPrompt,
            hasImage: includeVisionImage,
            image: sentImage,
            reply,
            model: chatModel,
            telemetry: data.hypnos_telemetry,
          },
          ...chatMessages,
        ];
        recordTelemetry(
          chatModel,
          includeVisionImage ? 'vision' : 'text_generation',
          data.hypnos_telemetry
        );
      }
      await fetchMemoryStatus();
    } catch (err) {
      lastError = err.message;
    } finally {
      isLoading = false;
    }
  }

  // ---- WAV encoding helpers for microphone capture --------------------------------------
  function downsampleTo16k(float32, inputRate) {
    const targetRate = 16000;
    if (inputRate === targetRate) return float32;
    const ratio = inputRate / targetRate;
    const outLen = Math.floor(float32.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(Math.floor((i + 1) * ratio), float32.length);
      let sum = 0;
      let n = 0;
      for (let j = start; j < end; j++) { sum += float32[j]; n++; }
      out[i] = n ? sum / n : 0;
    }
    return out;
  }

  function encodeWavPcm16(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  function setRecordedBlob(blob, label) {
    if (sttRecordedUrl) URL.revokeObjectURL(sttRecordedUrl);
    sttRecordedBlob = blob;
    sttRecordedUrl = blob ? URL.createObjectURL(blob) : null;
    sttSourceLabel = label;
  }

  function onSttFileChange(e) {
    const file = e.target.files?.[0] || null;
    sttSelectedFile = file;
    if (file) setRecordedBlob(null, '');
  }

  async function startRecording() {
    lastError = '';
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = micContext.createMediaStreamSource(micStream);
      micProcessor = micContext.createScriptProcessor(4096, 1, 1);
      micChunks = [];
      micProcessor.onaudioprocess = (ev) => {
        micChunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
      };
      source.connect(micProcessor);
      micProcessor.connect(micContext.destination);
      sttRecording = true;
      sttRecordSeconds = 0;
      micTimer = setInterval(() => (sttRecordSeconds += 1), 1000);
    } catch (err) {
      lastError = `Microphone unavailable: ${err.message}`;
    }
  }

  function stopRecording() {
    if (!sttRecording) return;
    clearInterval(micTimer);
    sttRecording = false;
    const inputRate = micContext?.sampleRate || 48000;
    try { micProcessor?.disconnect(); } catch {}
    micStream?.getTracks().forEach((t) => t.stop());
    micContext?.close();
    const total = micChunks.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of micChunks) { merged.set(c, off); off += c.length; }
    micChunks = [];
    const wav = encodeWavPcm16(downsampleTo16k(merged, inputRate), 16000);
    sttSelectedFile = null;
    setRecordedBlob(wav, `microphone recording (${sttRecordSeconds}s, 16 kHz WAV)`);
  }

  async function loadSampleClip() {
    lastError = '';
    try {
      const res = await fetch('/samples/jfk.wav');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      sttSelectedFile = null;
      setRecordedBlob(await res.blob(), 'sample clip: JFK inaugural address excerpt (11 s)');
    } catch (err) {
      lastError = `Could not load sample clip: ${err.message}`;
    }
  }

  async function runAudioTranscription(e) {
    e?.preventDefault();
    lastError = '';
    const source = sttSelectedFile || sttRecordedBlob;
    if (!source) {
      lastError = 'Choose an audio file (WAV, MP3 or FLAC), record from the microphone, or load the sample clip first.';
      return;
    }
    isLoading = true;
    try {
      const formData = new FormData();
      formData.append('model', sttModel);
      if (sttLanguage.trim()) formData.append('language', sttLanguage.trim());
      if (sttPrompt.trim()) formData.append('prompt', sttPrompt.trim());
      if (sttSelectedFile) {
        formData.append('file', sttSelectedFile, sttSelectedFile.name);
      } else {
        formData.append('file', sttRecordedBlob, 'recording.wav');
      }

      const res = await apiFetch(`/v1/audio/transcriptions`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);

      sttResult = data;
      recordTelemetry(sttModel, 'speech_to_text', data.hypnos_telemetry);
      await fetchMemoryStatus();
    } catch (err) {
      lastError = err.message;
    } finally {
      isLoading = false;
    }
  }

  async function runTextToSpeech(e) {
    e?.preventDefault();
    isLoading = true;
    lastError = '';
    try {
      const res = await apiFetch(`/v1/audio/speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: ttsModel,
          input: ttsInput,
          voice: ttsVoice,
          speed: Number(ttsSpeed),
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData?.error?.message || `HTTP ${res.status}`);
      }

      const prevTier = res.headers.get('x-hypnos-previous-tier') || 'Tier1GpuVram';
      const gpuLayers = Number(res.headers.get('x-hypnos-gpu-layers') || 12);
      const switchMs = Number(res.headers.get('x-hypnos-switch-ms') || 0);

      const blob = await res.blob();
      if (ttsAudioUrl) URL.revokeObjectURL(ttsAudioUrl);
      ttsAudioUrl = URL.createObjectURL(blob);
      ttsHeadersInfo = {
        previous_tier: prevTier,
        gpu_layers: gpuLayers,
        switch_ms: switchMs,
        size_kb: (blob.size / 1024).toFixed(1),
      };

      await fetchMemoryStatus();
    } catch (err) {
      lastError = err.message;
    } finally {
      isLoading = false;
    }
  }

  async function runEmbeddings(e) {
    e?.preventDefault();
    isLoading = true;
    lastError = '';
    try {
      const inputs = embedInputsRaw
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);

      const res = await apiFetch(`/v1/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: embedModel,
          input: inputs,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);

      embedResult = { ...data, inputs };
      recordTelemetry(embedModel, 'embeddings', data.hypnos_telemetry);
      await fetchMemoryStatus();
    } catch (err) {
      lastError = err.message;
    } finally {
      isLoading = false;
    }
  }

  async function runImageGeneration(e) {
    e?.preventDefault();
    isLoading = true;
    lastError = '';
    try {
      const res = await apiFetch(`/v1/images/generations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: imageModel,
          prompt: imagePrompt,
          size: imageSize,
          n: Number(imageCount),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);

      imageResult = data;
      recordTelemetry(imageModel, 'image_generation', data.hypnos_telemetry);
      await fetchMemoryStatus();
    } catch (err) {
      lastError = err.message;
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="min-h-screen flex flex-col bg-slate-950 text-slate-100">
  <!-- Top Header Bar -->
  <header class="border-b border-slate-800 bg-slate-900/70 backdrop-blur sticky top-0 z-20">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-lg shadow-lg shadow-indigo-600/30">
          L
        </div>
        <div>
          <h1 class="font-bold text-base tracking-tight flex items-center gap-2">
            Hypnos Studio
            <span class="text-xs font-normal px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              3-Tier GPU / Pinned RAM / NVMe Orchestrator
            </span>
          </h1>
          <p class="text-xs text-slate-400">
            {#if memoryStatus}
              {memoryStatus.capabilities.device_name} • Topology:
              <span class="text-slate-200 font-medium">{memoryStatus.capabilities.topology}</span>
            {:else}
              Connecting to Hypnos Rust Server...
            {/if}
          </p>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <input
          type="text"
          bind:value={backendUrl}
          class="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 w-52"
          placeholder="http://localhost:8080"
        />
        <input
          type="password"
          bind:value={apiKey}
          autocomplete="off"
          aria-label="API key"
          title="Sent as Authorization: Bearer on every request; stored in this browser only"
          class="bg-slate-950 border rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:outline-none w-40 {authRejected ? 'border-rose-500 focus:border-rose-400' : 'border-slate-800 focus:border-indigo-500'}"
          placeholder="API key (if required)"
        />
        <button
          onclick={fetchMemoryStatus}
          class="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 transition"
        >
          Refresh
        </button>
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border {isConnected ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-rose-500/10 text-rose-300 border-rose-500/30'}">
          <span class="w-2 h-2 rounded-full {isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}"></span>
          {isConnected ? 'Backend Online' : authRejected ? 'API Key Required' : 'Backend Offline'}
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content Grid -->
  <main class="max-w-7xl w-full mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
    <!-- Left Column (5 cols): 3-Tier Memory Hierarchy & Model Residency Matrix -->
    <section class="lg:col-span-5 flex flex-col gap-5">
      <!-- Tier 1 & Tier 2 Real-Time Capacity Gauges -->
      <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold text-slate-200 flex items-center gap-2">
            Hardware Memory Pools
            {#if isSimulated}
              <span
                class="text-[10px] font-medium px-1.5 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-300 border border-fuchsia-500/40"
                title="hardware.simulate_physical_memory_bytes: budgets are derived as on this smaller machine"
              >
                Simulated {formatBytes(memoryStatus.capabilities.physical_memory_bytes)} machine
                {#if memoryStatus.capabilities.installed_memory_bytes}
                  ({formatBytes(memoryStatus.capabilities.installed_memory_bytes)} installed)
                {/if}
              </span>
            {/if}
          </h2>
          {#if memoryStatus && !isUnifiedMemory}
            <span class="text-xs text-slate-400">
              DMA Speed: ~{memoryStatus.capabilities.tier2_to_tier1_bandwidth_gbps} GB/s
            </span>
          {/if}
        </div>

        {#if isUnifiedMemory}
        <!-- Unified memory: one physical pool; the marker is the GPU working-set limit -->
        <div class="space-y-1.5">
          <div class="flex justify-between text-xs">
            <span class="font-medium text-slate-200">Unified Memory (CPU + GPU share one pool)</span>
            <span class="text-slate-300 font-mono">
              {formatBytes(vramUsedBytes + pinnedRamUsedBytes)} / {formatBytes(umaPhysicalBytes)}
            </span>
          </div>
          <div class="relative w-full h-3 rounded-full bg-slate-950 overflow-hidden border border-slate-800 flex">
            <div
              class="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300"
              style="width: {umaResidentPct}%"
            ></div>
            <div
              class="h-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-300"
              style="width: {umaWarmPct}%"
            ></div>
            <div class="absolute top-0 h-full w-0.5 bg-slate-300/70" style="left: {umaGpuLimitPct}%"></div>
          </div>
          <div class="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
            <span><span class="text-emerald-300">■</span> GPU resident {formatBytes(vramUsedBytes)} / {formatBytes(vramTotalBytes)}</span>
            <span><span class="text-amber-300">■</span> Warm {formatBytes(pinnedRamUsedBytes)} / {formatBytes(pinnedRamTotalBytes)}</span>
            <span>│ GPU working-set limit</span>
          </div>
        </div>
        {:else}
        <!-- Tier 1 GPU VRAM Bar -->
        <div class="space-y-1.5">
          <div class="flex justify-between text-xs">
            <span class="font-medium text-emerald-300">Tier 1 • GPU VRAM (Weights + KV Cache)</span>
            <span class="text-slate-300 font-mono">
              {formatBytes(vramUsedBytes)} / {formatBytes(vramTotalBytes)} ({vramUsagePct.toFixed(1)}%)
            </span>
          </div>
          <div class="w-full h-3 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
            <div
              class="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300"
              style="width: {vramUsagePct}%"
            ></div>
          </div>
        </div>

        <!-- Tier 2 Pinned Host RAM Bar -->
        <div class="space-y-1.5">
          <div class="flex justify-between text-xs">
            <span class="font-medium text-amber-300">Tier 2 • Pinned Host RAM (VirtualLock / mlock)</span>
            <span class="text-slate-300 font-mono">
              {formatBytes(pinnedRamUsedBytes)} / {formatBytes(pinnedRamTotalBytes)} ({pinnedRamUsagePct.toFixed(1)}%)
            </span>
          </div>
          <div class="w-full h-3 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
            <div
              class="h-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-300"
              style="width: {pinnedRamUsagePct}%"
            ></div>
          </div>
        </div>
        {/if}

        <!-- hypnos process memory (hardware.max_process_memory_bytes) -->
        {#if processUsedBytes !== null}
          {#if processCapBytes}
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <span class="font-medium text-sky-300">Hypnos Process (cap)</span>
                <span class="text-slate-300 font-mono">
                  {formatBytes(processUsedBytes)} / {formatBytes(processCapBytes)} ({processPct.toFixed(1)}%)
                </span>
              </div>
              <div class="w-full h-3 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                <div
                  class="h-full transition-all duration-300 {processPct > 95 ? 'bg-rose-500' : processPct > 80 ? 'bg-amber-400' : 'bg-gradient-to-r from-sky-500 to-cyan-400'}"
                  style="width: {processPct}%"
                ></div>
              </div>
              <p class="text-[11px] text-slate-400">
                Loads that would cross the cap evict idle models first, or fail with HTTP 507.
              </p>
            </div>
          {:else}
            <p class="text-xs text-slate-400">
              Hypnos process: <span class="font-mono text-slate-300">{formatBytes(processUsedBytes)}</span>
              <span class="text-slate-500">(no cap; set hardware.max_process_memory_bytes)</span>
            </p>
          {/if}
        {/if}

        <!-- Automatic evictions, newest first -->
        {#if recentEvictions.length > 0}
          <div class="space-y-1 pt-1 border-t border-slate-800">
            <h3 class="text-xs font-medium text-slate-300 pt-2">Recent evictions</h3>
            <ul class="space-y-1">
              {#each recentEvictions as ev (ev.unix_ms + ev.model)}
                <li class="text-[11px] text-slate-400 flex justify-between gap-2">
                  <span>
                    <span class="text-slate-200 font-medium">{ev.model}</span>
                    → {tierLabel(ev.to_tier)} for
                    <span class="text-slate-200">{ev.for_model}</span>:
                    <span class={ev.reason === 'process_memory_cap' ? 'text-sky-300' : 'text-amber-300'}>
                      {EVICTION_REASONS[ev.reason] || ev.reason}
                    </span>
                  </span>
                  <span class="text-slate-500 shrink-0">{timeAgo(ev.unix_ms)}</span>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>

      <!-- 5-Model Residency Cards -->
      <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold text-slate-200">Registered Multi-Modal Models</h2>
          <span class="text-xs text-slate-400">Click tier buttons to pre-stage or evict</span>
        </div>

        {#if memoryStatus?.models}
          <div class="space-y-3">
            {#each memoryStatus.models as m (m.id)}
              {@const totalLayers =
                m.placement.gpu_layers + m.placement.pinned_ram_layers + m.placement.ssd_layers || 1}
              {@const gpuPct = (m.placement.gpu_layers / totalLayers) * 100}
              {@const ramPct = (m.placement.pinned_ram_layers / totalLayers) * 100}

              <div class="rounded-lg border border-slate-800 bg-slate-950/80 p-3.5 space-y-2.5">
                <div class="flex items-start justify-between gap-2">
                  <div>
                    <div class="flex items-center gap-2 flex-wrap">
                      <span class="font-mono font-semibold text-sm text-white">{m.id}</span>
                      <span class="text-[10px] px-2 py-0.5 rounded border font-medium {tierBadgeClass(m.current_tier)}">
                        {tierLabel(m.current_tier)}
                      </span>
                    </div>
                    <div class="flex flex-wrap gap-1 mt-1">
                      {#each m.modalities || [m.modality] as mod}
                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                          {mod}
                        </span>
                      {/each}
                      <span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800/40">
                        Priority: {m.priority}
                      </span>
                    </div>
                  </div>

                  <div class="text-right font-mono text-xs text-slate-400">
                    <div>VRAM: <span class="text-emerald-300">{formatBytes(m.placement.vram_weight_bytes + m.placement.vram_kv_and_workspace_bytes)}</span></div>
                    <div>KV+Scratch: <span class="text-sky-300">{formatBytes(m.placement.vram_kv_and_workspace_bytes)}</span></div>
                  </div>
                </div>

                <!-- Layer Distribution Bar (GPU vs Pinned RAM vs NVMe SSD) -->
                <div class="space-y-1">
                  <div class="flex justify-between text-[11px] text-slate-400">
                    <span>
                      Layers: <strong class="text-emerald-300">{m.placement.gpu_layers} GPU</strong> /
                      <strong class="text-amber-300">{m.placement.pinned_ram_layers} RAM</strong> /
                      <span>{m.placement.ssd_layers} SSD</span>
                    </span>
                    {#if m.placement.allocated_context_len > 0}
                      <span class="text-sky-300 font-mono">{m.placement.allocated_context_len} ctx</span>
                    {/if}
                  </div>
                  <div class="w-full h-2 rounded-full bg-slate-900 overflow-hidden flex border border-slate-800">
                    <div class="bg-emerald-500 h-full transition-all" style="width: {gpuPct}%"></div>
                    <div class="bg-amber-500 h-full transition-all" style="width: {ramPct}%"></div>
                  </div>
                </div>

                <!-- Manual Tier Control Actions -->
                <div class="flex items-center justify-between pt-1 border-t border-slate-800/80 text-[11px]">
                  <span class="text-slate-500">
                    Served: {m.total_requests_served} reqs • Idle: {m.idle_seconds.toFixed(0)}s
                  </span>
                  <div class="flex items-center gap-1.5">
                    <button
                      onclick={() => manualSetTier(m.id, 'tier1_gpu_vram')}
                      disabled={isLoading}
                      class="px-2 py-0.5 rounded bg-emerald-500/15 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30 transition"
                    >
                      To VRAM
                    </button>
                    <button
                      onclick={() => manualSetTier(m.id, 'tier2_pinned_ram')}
                      disabled={isLoading}
                      class="px-2 py-0.5 rounded bg-amber-500/15 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 transition"
                    >
                      To RAM
                    </button>
                    <button
                      onclick={() => manualSetTier(m.id, 'tier3_nvme_ssd')}
                      disabled={isLoading}
                      class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
                    >
                      Evict SSD
                    </button>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <div class="text-xs text-slate-400 py-8 text-center">
            Start the Rust backend (<code class="text-indigo-300">cd backend && cargo run</code>) to view live model tiers.
          </div>
        {/if}
      </div>
    </section>

    <!-- Right Column (7 cols): Multi-Modal OpenAI API Playground & Telemetry -->
    <section class="lg:col-span-7 flex flex-col gap-5">
      {#if lastError}
        <div class="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-200 flex items-center justify-between">
          <span><strong>Error:</strong> {lastError}</span>
          <button onclick={() => (lastError = '')} class="text-rose-300 hover:text-white">Dismiss</button>
        </div>
      {/if}

      <!-- Modality Navigation Tabs -->
      <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-4 flex flex-col gap-4">
        <div class="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
          <button
            onclick={() => (activeTab = 'chat')}
            class="px-3.5 py-2 rounded-lg text-xs font-semibold transition {activeTab === 'chat' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25' : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800'}"
          >
            1. Text & Vision (gemma4:26b-q4)
          </button>
          <button
            onclick={() => (activeTab = 'stt')}
            class="px-3.5 py-2 rounded-lg text-xs font-semibold transition {activeTab === 'stt' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25' : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800'}"
          >
            2. Audio STT (gemma4:e4b-q4)
          </button>
          <button
            onclick={() => (activeTab = 'tts')}
            class="px-3.5 py-2 rounded-lg text-xs font-semibold transition {activeTab === 'tts' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25' : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800'}"
          >
            3. Speech TTS (kokoro-82m-q4)
          </button>
          <button
            onclick={() => (activeTab = 'embeddings')}
            class="px-3.5 py-2 rounded-lg text-xs font-semibold transition {activeTab === 'embeddings' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25' : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800'}"
          >
            4. Embeddings (gemmaembedding2)
          </button>
          <button
            onclick={() => (activeTab = 'image')}
            class="px-3.5 py-2 rounded-lg text-xs font-semibold transition {activeTab === 'image' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25' : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800'}"
          >
            5. Image Gen (moss-image)
          </button>
          <button
            onclick={() => (activeTab = 'logs')}
            class="px-3.5 py-2 rounded-lg text-xs font-semibold transition {activeTab === 'logs' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25' : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800'}"
          >
            6. Request Logs
          </button>
        </div>

        <!-- TAB 1: TEXT & VISION CHAT -->
        {#if activeTab === 'chat'}
          <form onsubmit={runChatCompletion} class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="chat-model">Model</label>
                <select
                  id="chat-model"
                  bind:value={chatModel}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                >
                  <option value="gemma4:26b-q4">gemma4:26b-q4 (Text + Vision)</option>
                  <option value="gemma4:e4b-q4">gemma4:e4b-q4 (Edge Multimodal)</option>
                </select>
              </div>

              <div>
                <label class="block text-xs text-slate-400 mb-1" for="chat-ctx">
                  KV Context Window: <span class="text-indigo-300 font-mono">{chatContextLen}</span>
                </label>
                <input
                  id="chat-ctx"
                  type="range"
                  min="2048"
                  max="131072"
                  step="2048"
                  bind:value={chatContextLen}
                  class="w-full accent-indigo-500 mt-2"
                />
              </div>

              <div>
                <label class="block text-xs text-slate-400 mb-1" for="chat-max-tokens">Max New Tokens</label>
                <input
                  id="chat-max-tokens"
                  type="number"
                  bind:value={chatMaxTokens}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white font-mono"
                />
              </div>
            </div>

            <div class="flex flex-wrap items-center gap-4 text-xs">
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" bind:checked={includeVisionImage} class="rounded accent-indigo-500" />
                <span>Attach Vision Image (<code class="text-indigo-300">image_url</code> multimodal block)</span>
              </label>

              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" bind:checked={chatStream} class="rounded accent-indigo-500" />
                <span>Stream SSE Response</span>
              </label>
            </div>

            {#if includeVisionImage}
              <div class="rounded-lg border border-slate-800 bg-slate-900/60 p-3 space-y-3">
                <div class="space-y-2">
                  <label class="block text-xs text-slate-400" for="vision-file">
                    Upload an image (<strong class="text-slate-200">PNG, JPEG, WebP or GIF</strong>, up to {formatFileSize(MAX_VISION_IMAGE_BYTES)}).
                    It is sent inline as a base64 <code class="text-indigo-300">data:</code> URL.
                  </label>
                  <input
                    id="vision-file"
                    type="file"
                    accept="image/png,image/jpeg,image/webp,image/gif,image/*"
                    onchange={onVisionFileChange}
                    class="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
                  />
                </div>

                {#if visionPreviewUrl}
                  <div class="flex items-start gap-3">
                    <img
                      src={visionPreviewUrl}
                      alt="Vision attachment preview"
                      class="max-h-40 max-w-[16rem] rounded-lg border border-slate-700 object-contain bg-slate-950"
                    />
                    <div class="flex-1 min-w-0 text-xs text-slate-400 space-y-1">
                      {#if visionImageFile}
                        <div class="truncate text-slate-200" title={visionImageFile.name}>{visionImageFile.name}</div>
                        <div>{visionImageFile.type} • {formatFileSize(visionImageFile.size)}</div>
                        <div class="font-mono text-[11px]">data URL: {formatFileSize(visionImageUrl.length)}</div>
                      {:else}
                        <div class="text-slate-200">From URL field</div>
                      {/if}
                      <button
                        type="button"
                        onclick={clearVisionImage}
                        class="mt-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition"
                      >
                        Remove image
                      </button>
                    </div>
                  </div>
                {:else}
                  <div class="text-[11px] text-slate-500">
                    No image selected: the request will carry a 1×1 transparent placeholder, which only exercises the vision code path.
                  </div>
                {/if}

                <div>
                  <label class="block text-xs text-slate-400 mb-1" for="vision-url">
                    …or paste an image URL / Base64 Data URI
                  </label>
                  <input
                    id="vision-url"
                    type="text"
                    bind:value={visionImageUrl}
                    oninput={onVisionUrlInput}
                    class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-300"
                  />
                </div>
              </div>
            {/if}

            <div>
              <label class="block text-xs text-slate-400 mb-1" for="chat-prompt">Prompt</label>
              <textarea
                id="chat-prompt"
                rows="3"
                bind:value={chatPrompt}
                class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
              ></textarea>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              class="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white transition shadow-lg shadow-indigo-600/20"
            >
              {isLoading ? 'Allocating VRAM & Generating...' : `Send to /v1/chat/completions (${chatModel})`}
            </button>
          </form>

          {#if chatMessages.length > 0}
            <div class="space-y-3 pt-2 border-t border-slate-800 max-h-80 overflow-y-auto">
              {#each chatMessages as msg}
                <div class="rounded-lg border border-slate-800 bg-slate-950 p-3 space-y-2 text-xs">
                  <div class="flex items-center justify-between text-slate-400">
                    <span>
                      <strong class="text-indigo-300">{msg.model}</strong>
                      {msg.hasImage ? '• [Vision + Text]' : '• [Text]'}
                    </span>
                    {#if msg.telemetry}
                      <span class="font-mono text-[11px] text-emerald-300">
                        Switch: {msg.telemetry.tier_switch_latency_ms.toFixed(2)}ms •
                        Infer: {msg.telemetry.inference_latency_ms.toFixed(2)}ms •
                        GPU Layers: {msg.telemetry.placement.gpu_layers}
                      </span>
                    {/if}
                  </div>
                  <div class="text-slate-300"><strong>User:</strong> {msg.prompt}</div>
                  {#if msg.image && msg.image !== PLACEHOLDER_IMAGE_URL}
                    <img
                      src={msg.image}
                      alt="Attachment sent with this prompt"
                      class="max-h-32 rounded border border-slate-700 object-contain bg-slate-950"
                    />
                  {/if}
                  <div class="text-emerald-200 font-mono bg-slate-900/90 p-2.5 rounded border border-slate-800">
                    {msg.reply}
                  </div>
                </div>
              {/each}
            </div>
          {/if}

        <!-- TAB 2: AUDIO SPEECH-TO-TEXT (gemma4:e4b-q4) -->
        {:else if activeTab === 'stt'}
          <form onsubmit={runAudioTranscription} class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="stt-model">Audio Model</label>
                <input
                  id="stt-model"
                  type="text"
                  bind:value={sttModel}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="stt-lang">Language Code</label>
                <input
                  id="stt-lang"
                  type="text"
                  bind:value={sttLanguage}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                />
              </div>
            </div>

            <div class="space-y-2">
              <label class="block text-xs text-slate-400" for="stt-file">
                Audio source: upload a <strong class="text-slate-200">WAV, MP3 or FLAC</strong> file, record from the microphone, or use the sample clip
              </label>
              <input
                id="stt-file"
                type="file"
                accept=".wav,.mp3,.flac,audio/wav,audio/x-wav,audio/mpeg,audio/flac"
                onchange={onSttFileChange}
                class="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
              />
              <div class="flex flex-wrap items-center gap-2">
                {#if !sttRecording}
                  <button type="button" onclick={startRecording} disabled={isLoading}
                    class="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-xs font-semibold text-white transition">
                    Record from microphone
                  </button>
                {:else}
                  <button type="button" onclick={stopRecording}
                    class="px-3 py-1.5 rounded-lg bg-rose-700 hover:bg-rose-600 text-xs font-semibold text-white transition animate-pulse">
                    Stop recording ({sttRecordSeconds}s)
                  </button>
                {/if}
                <button type="button" onclick={loadSampleClip} disabled={isLoading || sttRecording}
                  class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-xs font-semibold text-slate-200 transition">
                  Use sample clip (JFK, 11 s)
                </button>
                {#if sttSelectedFile}
                  <span class="text-xs text-slate-400 font-mono">file: {sttSelectedFile.name} ({(sttSelectedFile.size / 1024).toFixed(0)} KB)</span>
                {:else if sttRecordedBlob}
                  <span class="text-xs text-slate-400 font-mono">{sttSourceLabel}</span>
                {/if}
              </div>
              {#if sttRecordedUrl}
                <audio controls src={sttRecordedUrl} class="w-full h-8"></audio>
              {/if}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              class="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white transition"
            >
              {isLoading ? 'Loading gemma4:e4b-q4 & Transcribing...' : 'Transcribe via /v1/audio/transcriptions'}
            </button>
          </form>

          {#if sttResult}
            <div class="rounded-lg border border-slate-800 bg-slate-950 p-3.5 space-y-2 text-xs">
              <div class="flex justify-between text-slate-400">
                <span>Language: <strong class="text-white">{sttResult.language}</strong></span>
                {#if sttResult.hypnos_telemetry}
                  <span class="font-mono text-emerald-300">
                    Switch: {sttResult.hypnos_telemetry.tier_switch_latency_ms.toFixed(2)}ms •
                    GPU Layers: {sttResult.hypnos_telemetry.placement.gpu_layers}
                  </span>
                {/if}
              </div>
              <p class="font-mono text-emerald-200 bg-slate-900 p-2.5 rounded border border-slate-800">
                {sttResult.text}
              </p>
            </div>
          {/if}

        <!-- TAB 3: TEXT-TO-SPEECH (kokoro-82m-q4) -->
        {:else if activeTab === 'tts'}
          <form onsubmit={runTextToSpeech} class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="tts-model">TTS Model</label>
                <input
                  id="tts-model"
                  type="text"
                  bind:value={ttsModel}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="tts-voice">Voice Preset</label>
                <select
                  id="tts-voice"
                  bind:value={ttsVoice}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                >
                  <option value="alloy">alloy</option>
                  <option value="af_bella">af_bella (Kokoro)</option>
                  <option value="am_adam">am_adam (Kokoro)</option>
                </select>
              </div>
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="tts-speed">Speed ({ttsSpeed}x)</label>
                <input
                  id="tts-speed"
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  bind:value={ttsSpeed}
                  class="w-full accent-indigo-500 mt-2"
                />
              </div>
            </div>

            <div>
              <label class="block text-xs text-slate-400 mb-1" for="tts-input">Input Text</label>
              <textarea
                id="tts-input"
                rows="3"
                bind:value={ttsInput}
                class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-white"
              ></textarea>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              class="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white transition"
            >
              {isLoading ? 'Synthesizing Audio...' : 'Synthesize Speech via /v1/audio/speech'}
            </button>
          </form>

          {#if ttsAudioUrl}
            <div class="rounded-lg border border-slate-800 bg-slate-950 p-3.5 space-y-3 text-xs">
              {#if ttsHeadersInfo}
                <div class="flex flex-wrap justify-between text-slate-400 font-mono">
                  <span>Previous Tier: <strong class="text-emerald-300">{ttsHeadersInfo.previous_tier}</strong></span>
                  <span>Switch: <strong class="text-emerald-300">{ttsHeadersInfo.switch_ms} ms</strong></span>
                  <span>GPU Layers: <strong class="text-emerald-300">{ttsHeadersInfo.gpu_layers}</strong></span>
                  <span>WAV Size: {ttsHeadersInfo.size_kb} KB</span>
                </div>
              {/if}
              <audio controls src={ttsAudioUrl} class="w-full h-9"></audio>
            </div>
          {/if}

        <!-- TAB 4: VECTOR EMBEDDINGS (gemmaembedding2) -->
        {:else if activeTab === 'embeddings'}
          <form onsubmit={runEmbeddings} class="space-y-4">
            <div>
              <label class="block text-xs text-slate-400 mb-1" for="embed-input">
                Batch Input Texts (one sentence per line — computes pairwise Cosine Similarity)
              </label>
              <textarea
                id="embed-input"
                rows="4"
                bind:value={embedInputsRaw}
                class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-white font-mono"
              ></textarea>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              class="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white transition"
            >
              {isLoading ? 'Computing Embeddings...' : 'Generate Embeddings via /v1/embeddings (gemmaembedding2)'}
            </button>
          </form>

          {#if embedResult}
            <div class="rounded-lg border border-slate-800 bg-slate-950 p-3.5 space-y-3 text-xs">
              <div class="flex justify-between text-slate-400">
                <span>Vectors returned: <strong>{embedResult.data.length}</strong> (64-d preview)</span>
                {#if embedResult.hypnos_telemetry}
                  <span class="font-mono text-emerald-300">
                    Switch: {embedResult.hypnos_telemetry.tier_switch_latency_ms.toFixed(2)}ms
                  </span>
                {/if}
              </div>

              {#if cosineMatrix.length > 0}
                <div class="overflow-x-auto">
                  <table class="w-full text-left border-collapse font-mono text-[11px]">
                    <thead>
                      <tr class="border-b border-slate-800 text-slate-400">
                        <th class="py-1 pr-2">Cosine Sim</th>
                        {#each embedResult.inputs as _, idx}
                          <th class="py-1 px-2">#{idx + 1}</th>
                        {/each}
                      </tr>
                    </thead>
                    <tbody>
                      {#each cosineMatrix as row, i}
                        <tr class="border-b border-slate-900">
                          <td class="py-1 pr-2 text-slate-300 truncate max-w-[160px]" title={embedResult.inputs[i]}>
                            #{i + 1}: {embedResult.inputs[i].slice(0, 24)}...
                          </td>
                          {#each row as val}
                            <td class="py-1 px-2">
                              <span class="px-1.5 py-0.5 rounded {val > 0.85 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-slate-800 text-slate-300'}">
                                {val.toFixed(3)}
                              </span>
                            </td>
                          {/each}
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {/if}
            </div>
          {/if}

        <!-- TAB 6: REQUEST LOGS (GET /v1/logs) -->
        {:else if activeTab === 'logs'}
          <div class="space-y-3 text-xs">
            <div class="flex flex-wrap items-end gap-2">
              <div>
                <label class="block text-slate-400 mb-1" for="log-model">Model</label>
                <input
                  id="log-model"
                  bind:value={logFilterModel}
                  placeholder="gemma4"
                  class="w-32 bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-white font-mono"
                />
              </div>
              <div>
                <label class="block text-slate-400 mb-1" for="log-route">Route</label>
                <input
                  id="log-route"
                  bind:value={logFilterRoute}
                  placeholder="chat"
                  class="w-32 bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-white font-mono"
                />
              </div>
              <div>
                <label class="block text-slate-400 mb-1" for="log-status">Status</label>
                <select
                  id="log-status"
                  bind:value={logFilterStatus}
                  class="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-white"
                >
                  <option value="">all</option>
                  <option value="ok">ok</option>
                  <option value="error">errors</option>
                  <option value="4xx">4xx</option>
                  <option value="5xx">5xx</option>
                </select>
              </div>
              <div>
                <label class="block text-slate-400 mb-1" for="log-level">Capture level</label>
                <select
                  id="log-level"
                  value={logLevel}
                  onchange={(e) => setLogLevel(e.currentTarget.value)}
                  class="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-white"
                >
                  {#each LOG_LEVELS as lvl}
                    <option value={lvl}>{lvl}</option>
                  {/each}
                </select>
              </div>
              <label class="flex items-center gap-1.5 text-slate-300 pb-2">
                <input type="checkbox" bind:checked={logAutoRefresh} />
                auto-refresh
              </label>
              <button
                onclick={fetchLogs}
                class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold"
              >
                Refresh
              </button>
              <button
                onclick={clearLogs}
                class="px-3 py-1.5 rounded-lg bg-rose-900/60 hover:bg-rose-800/60 text-rose-200 font-semibold"
              >
                Clear
              </button>
              <span class="text-slate-500 pb-2 ml-auto">
                {logStored} / {logCapacity} stored
                {#if logLevel === 'off'}
                  · capture is off
                {:else if logLevel === 'error' || logLevel === 'warn'}
                  · failed calls only
                {:else if logLevel === 'info'}
                  · metadata only (debug adds text, trace adds payloads)
                {:else if logLevel === 'debug'}
                  · with prompt and completion text
                {:else}
                  · with full payloads
                {/if}
              </span>
            </div>

            {#if logError}
              <div class="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-rose-200">
                {logError}
              </div>
            {/if}

            <div class="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950">
              <table class="w-full text-left border-collapse font-mono text-[11px]">
                <thead>
                  <tr class="border-b border-slate-800 text-slate-400">
                    <th class="py-1.5 px-2">time</th>
                    <th class="py-1.5 px-2">route</th>
                    <th class="py-1.5 px-2">model</th>
                    <th class="py-1.5 px-2">status</th>
                    <th class="py-1.5 px-2 text-right">ms</th>
                    <th class="py-1.5 px-2 text-right">in</th>
                    <th class="py-1.5 px-2 text-right">out</th>
                    <th class="py-1.5 px-2 text-right">cached</th>
                    <th class="py-1.5 px-2 text-right">tok/s</th>
                    <th class="py-1.5 px-2 text-right">switch ms</th>
                  </tr>
                </thead>
                <tbody>
                  {#each logRecords as r (r.id)}
                    <tr
                      onclick={() => openLog(r.id)}
                      class="border-b border-slate-900 cursor-pointer hover:bg-slate-900 {logSelected?.id === r.id ? 'bg-indigo-950/60' : ''}"
                    >
                      <td class="py-1 px-2 text-slate-400 whitespace-nowrap">{logTime(r.started_at)}</td>
                      <td class="py-1 px-2 text-slate-200 whitespace-nowrap">
                        {r.route.replace('/v1/', '')}{r.stream ? ' ⇢' : ''}
                      </td>
                      <td class="py-1 px-2 text-slate-300 whitespace-nowrap">{r.model ?? '–'}</td>
                      <td class="py-1 px-2">
                        <span class="px-1.5 py-0.5 rounded {logStatusClass(r.status)}">{r.status}</span>
                      </td>
                      <td class="py-1 px-2 text-right text-slate-300">{logNum(r.duration_ms)}</td>
                      <td class="py-1 px-2 text-right text-slate-300">{logNum(r.tokens_in)}</td>
                      <td class="py-1 px-2 text-right text-slate-300">{logNum(r.tokens_out)}</td>
                      <td class="py-1 px-2 text-right text-slate-300">{logNum(r.cached_tokens)}</td>
                      <td class="py-1 px-2 text-right text-emerald-300">{logNum(r.tokens_per_s, 1)}</td>
                      <td class="py-1 px-2 text-right text-sky-300">{logNum(r.tier_switch_ms)}</td>
                    </tr>
                  {:else}
                    <tr>
                      <td colspan="10" class="py-4 text-center text-slate-500">
                        No calls recorded yet. Use another tab, then come back.
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>

            {#if logSelected}
              <div class="rounded-lg border border-slate-800 bg-slate-950 p-3.5 space-y-3">
                <div class="flex items-center justify-between">
                  <div class="font-mono text-slate-300">
                    <span class="text-slate-500">{logSelected.id}</span>
                    · {logSelected.method} {logSelected.route}
                    · <span class="px-1.5 py-0.5 rounded {logStatusClass(logSelected.status)}">{logSelected.status}</span>
                    {#if logSelected.error}
                      <span class="text-rose-300 ml-2">{logSelected.error}</span>
                    {/if}
                  </div>
                  <button onclick={() => (logSelected = null)} class="text-slate-400 hover:text-white">Close</button>
                </div>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-2 font-mono text-[11px] text-slate-300">
                  <span>model: <strong class="text-white">{logSelected.model ?? '–'}</strong></span>
                  <span>duration: <strong class="text-white">{logNum(logSelected.duration_ms)} ms</strong></span>
                  <span>inference: <strong class="text-white">{logNum(logSelected.inference_ms)} ms</strong></span>
                  <span>tier switch: <strong class="text-white">{logNum(logSelected.tier_switch_ms)} ms</strong></span>
                  <span>tokens in / out: <strong class="text-white">{logNum(logSelected.tokens_in)} / {logNum(logSelected.tokens_out)}</strong></span>
                  <span>cached: <strong class="text-white">{logNum(logSelected.cached_tokens)}</strong></span>
                  <span>tok/s: <strong class="text-white">{logNum(logSelected.tokens_per_s, 1)}</strong></span>
                  <span>previous tier: <strong class="text-white">{logSelected.previous_tier ?? '–'}</strong></span>
                </div>
                {#if logSelected.input !== undefined || logSelected.output !== undefined}
                  <div class="grid md:grid-cols-2 gap-3">
                    <div>
                      <div class="text-slate-400 mb-1">Input</div>
                      <pre class="whitespace-pre-wrap break-words bg-slate-900 rounded-lg p-2 max-h-72 overflow-auto text-slate-200">{logSelected.input ?? ''}</pre>
                    </div>
                    <div>
                      <div class="text-slate-400 mb-1">Output</div>
                      <pre class="whitespace-pre-wrap break-words bg-slate-900 rounded-lg p-2 max-h-72 overflow-auto text-slate-200">{logSelected.output ?? ''}</pre>
                    </div>
                  </div>
                {:else}
                  <div class="text-slate-500">
                    Captured at level <strong>{logSelected.level}</strong>: no text. Set the capture level to
                    <strong>debug</strong> for prompt and completion text, <strong>trace</strong> for payloads.
                  </div>
                {/if}
                {#if logSelected.request_payload !== undefined || logSelected.response_payload !== undefined}
                  <div class="grid md:grid-cols-2 gap-3">
                    <div>
                      <div class="text-slate-400 mb-1">Request payload</div>
                      <pre class="whitespace-pre-wrap break-words bg-slate-900 rounded-lg p-2 max-h-72 overflow-auto text-slate-300">{logPretty(logSelected.request_payload)}</pre>
                    </div>
                    <div>
                      <div class="text-slate-400 mb-1">Response payload</div>
                      <pre class="whitespace-pre-wrap break-words bg-slate-900 rounded-lg p-2 max-h-72 overflow-auto text-slate-300">{logPretty(logSelected.response_payload)}</pre>
                    </div>
                  </div>
                {/if}
                {#if logSelected.truncated}
                  <div class="text-amber-300">Some captured text or payload was shortened.</div>
                {/if}
              </div>
            {/if}
          </div>

        <!-- TAB 5: IMAGE GENERATION (moss-image) -->
        {:else if activeTab === 'image'}
          <form onsubmit={runImageGeneration} class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="img-model">Diffusion / Flow Model</label>
                <input
                  id="img-model"
                  type="text"
                  bind:value={imageModel}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label class="block text-xs text-slate-400 mb-1" for="img-size">Resolution</label>
                <select
                  id="img-size"
                  bind:value={imageSize}
                  class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                >
                  <option value="1024x1024">1024x1024</option>
                  <option value="1024x768">1024x768</option>
                  <option value="512x512">512x512</option>
                </select>
              </div>
            </div>

            <div>
              <label class="block text-xs text-slate-400 mb-1" for="img-prompt">Image Prompt</label>
              <textarea
                id="img-prompt"
                rows="3"
                bind:value={imagePrompt}
                class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-white"
              ></textarea>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              class="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white transition"
            >
              {isLoading ? 'Swapping VRAM Tiers & Rendering...' : 'Generate Image via /v1/images/generations (moss-image)'}
            </button>
          </form>

          {#if imageResult}
            <div class="rounded-lg border border-slate-800 bg-slate-950 p-3.5 space-y-2 text-xs">
              {#if imageResult.hypnos_telemetry}
                <div class="flex flex-wrap justify-between gap-2 text-slate-400 font-mono">
                  <span>
                    From: <strong class="text-amber-300">{imageResult.hypnos_telemetry.previous_tier}</strong>
                  </span>
                  <span>
                    Switch: <strong class="text-emerald-300">{imageResult.hypnos_telemetry.tier_switch_latency_ms.toFixed(2)}ms</strong>
                  </span>
                  <span>
                    Evicted to RAM:
                    <strong class="text-rose-300">
                      {imageResult.hypnos_telemetry.evicted_models.length > 0
                        ? imageResult.hypnos_telemetry.evicted_models.join(', ')
                        : 'None'}
                    </strong>
                  </span>
                </div>
              {/if}
              {#each imageResult.data as img, i}
                {#if img.b64_json}
                  <div class="rounded bg-slate-900 border border-slate-800 overflow-hidden">
                    <img
                      src={`data:image/png;base64,${img.b64_json}`}
                      alt={`Generated image ${i + 1}`}
                      class="w-full h-auto block"
                    />
                    <div class="flex items-center justify-between px-2.5 py-2 font-mono text-slate-400">
                      <span>{imageSize} · PNG · {(img.b64_json.length * 0.75 / 1024).toFixed(0)} KB</span>
                      <a
                        href={`data:image/png;base64,${img.b64_json}`}
                        download={`hypnos-${imageModel}-${Date.now()}-${i + 1}.png`}
                        class="text-indigo-300 hover:text-indigo-200 underline"
                      >
                        Download
                      </a>
                    </div>
                  </div>
                {/if}
                {#if img.revised_prompt}
                  <div class="p-2.5 rounded bg-slate-900 border border-slate-800 text-emerald-200 font-mono">
                    {img.revised_prompt}
                  </div>
                {/if}
              {/each}
            </div>
          {/if}
        {/if}
      </div>

      <!-- Live Telemetry & Tier Transition Event Log -->
      <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3 flex-1">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold text-slate-200">Tier Switch & Eviction Telemetry Log</h2>
          {#if telemetryLog.length > 0}
            <button
              onclick={() => (telemetryLog = [])}
              class="text-xs text-slate-400 hover:text-slate-200"
            >
              Clear
            </button>
          {/if}
        </div>

        {#if telemetryLog.length === 0}
          <p class="text-xs text-slate-500 py-6 text-center">
            Run any request above to inspect real-time tier promotions, GPU/CPU layer splits, and LRU evictions.
          </p>
        {:else}
          <div class="space-y-2 max-h-60 overflow-y-auto">
            {#each telemetryLog as entry (entry.id)}
              <div class="rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs flex flex-wrap items-center justify-between gap-2 font-mono">
                <div class="flex items-center gap-2">
                  <span class="text-slate-500">{entry.timestamp}</span>
                  <span class="text-indigo-300 font-semibold">{entry.model}</span>
                  <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                    {entry.previous_tier} → {entry.placement.primary_tier}
                  </span>
                </div>
                <div class="flex items-center gap-3 text-[11px]">
                  <span class="text-emerald-300">
                    {entry.placement.gpu_layers} GPU / {entry.placement.pinned_ram_layers} RAM layers
                  </span>
                  <span class="text-sky-300">
                    Switch: {entry.tier_switch_latency_ms.toFixed(2)}ms
                  </span>
                  {#if entry.evicted_models?.length > 0}
                    <span class="text-rose-300">
                      Evicted→RAM: {entry.evicted_models.join(', ')}
                    </span>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </section>
  </main>
</div>
