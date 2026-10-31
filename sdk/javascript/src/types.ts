/**
 * Response and message shapes of the Hypnos HTTP API (docs/api.md). The server only ever adds
 * fields, so every object may carry more than is declared here.
 */

export type MemoryTier = "tier1_gpu_vram" | "tier2_pinned_ram" | "tier3_nvme_ssd";
/** Mixture-of-Experts expert offload mode (`POST /v1/models/{id}/moe-offload`). */
export type MoeOffload = "off" | "auto" | "on";
export type Modality =
  | "text_generation"
  | "vision"
  | "speech_to_text"
  | "text_to_speech"
  | "embeddings"
  | "image_generation"
  | "document_conversion"
  | "entity_extraction";

// --- shared -----------------------------------------------------------------------------------

export interface Placement {
  primary_tier: MemoryTier;
  gpu_layers: number;
  pinned_ram_layers: number;
  ssd_layers: number;
  vram_weight_bytes: number;
  vram_kv_and_workspace_bytes: number;
  pinned_ram_bytes: number;
  allocated_context_len: number;
  /** Layers whose MoE expert tensors run on the CPU (`moe_offload`); absent on older servers. */
  cpu_expert_layers?: number;
}

export interface EngineTiming {
  queue_ms?: number;
  context_ms?: number;
  prefill_ms?: number;
  decode_ms?: number;
  ttft_ms?: number;
  prefill_tokens_per_s?: number;
  decode_tokens_per_s?: number;
  cached_tokens?: number;
  concurrent_slots?: number;
  audio_secs?: number;
  rtf?: number;
  ttfa_ms?: number;
  pages?: number;
  ms_per_page?: number;
}

export interface LoadPhases {
  load_mode?: string;
  weights_ms?: number;
  projector_ms?: number;
  context_ms?: number;
  total_ms?: number;
  kv_restore_ms?: number;
  kv_restored_tokens?: number;
}

/** `hypnos_telemetry`: where the model was, what loading it cost, engine timings. */
export interface Telemetry {
  previous_tier: MemoryTier;
  placement: Placement;
  tier_switch_latency_ms: number;
  inference_latency_ms: number;
  evicted_models: string[];
  engine: EngineTiming;
  tier_switch_phases?: LoadPhases;
}

// --- chat -------------------------------------------------------------------------------------

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } }
  | { type: "input_audio"; input_audio: { data: string; format?: string } };

/** A function the model may call (`tools`, docs/api.md section 7.5). */
export interface ToolSpec {
  type: "function";
  function: {
    name: string;
    description?: string;
    /** JSON Schema of the arguments object. */
    parameters?: Record<string, unknown>;
  };
}

/** `tool_choice`: whether and which tool the model must call. */
export type ToolChoice =
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } };

/** A call the model asked for; `arguments` is a JSON object serialized as a string. */
export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/** One element of a streamed `delta.tool_calls`, accumulated by `index`. */
export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

export interface SystemMessage {
  role: "system";
  content: string | ContentPart[];
}

export interface UserMessage {
  role: "user";
  content: string | ContentPart[];
}

/** An assistant turn; `content` is `null` when it only calls tools. */
export interface AssistantMessage {
  role: "assistant";
  content: string | ContentPart[] | null;
  tool_calls?: ToolCall[];
}

/** The result of a tool call, sent back to the model. */
export interface ToolMessage {
  role: "tool";
  tool_call_id: string;
  content: string | ContentPart[];
  name?: string;
}

export type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

/** The assistant message of a completion. */
export interface CompletionMessage {
  role: string;
  content: string;
  /** Present only when the model called tools (`finish_reason` `tool_calls`). */
  tool_calls?: ToolCall[];
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens: number };
}

export interface ChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: { index: number; message: CompletionMessage; finish_reason: string }[];
  usage: Usage;
  hypnos_telemetry?: Telemetry;
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: { role?: string; content?: string; tool_calls?: ToolCallDelta[] };
    finish_reason: string | null;
  }[];
  usage?: Usage;
  hypnos_telemetry?: Telemetry;
  error?: { message?: string; type?: string };
}

// --- embeddings, audio, images, entities --------------------------------------------------------

export interface EmbeddingResponse {
  object: string;
  model: string;
  data: { object: string; index: number; embedding: number[] }[];
  usage: { prompt_tokens: number; total_tokens: number };
  hypnos_telemetry?: Telemetry;
}

export interface Transcription {
  text: string;
  language?: string;
  duration_secs?: number;
  hypnos_telemetry?: Telemetry;
}

export interface ImageResponse {
  created: number;
  data: { b64_json: string; revised_prompt?: string }[];
  hypnos_telemetry?: Telemetry;
}

/** `start`/`end` are Unicode scalar (code point) offsets: use `sliceByCodePoints`. */
export interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  score: number;
}

export interface EntityResponse {
  object: string;
  model: string;
  data: { object: string; index: number; entities: Entity[] }[];
  usage: { texts: number; characters: number };
  hypnos_telemetry?: Telemetry;
}

// --- documents --------------------------------------------------------------------------------

export interface DocumentPage {
  page: number;
  markdown: string;
  pictures: number;
  tables: number;
  ms: number;
  json?: Record<string, unknown>;
  picture_urls?: string[];
}

export interface Document {
  object: string;
  model: string;
  page_count: number;
  pages: DocumentPage[];
  markdown: string;
  usage: { pages: number; pictures: number; tables: number; pictures_described?: number };
  id?: string;
  pictures_expire_at?: string;
  filename?: string;
  hypnos_telemetry?: Telemetry;
}

export interface DocumentStarted {
  object: "document.started";
  page_count: number;
  pages: number[];
  id?: string;
}

/** One event of a document stream; dispatch on `object`. */
export type DocumentEvent =
  | DocumentStarted
  | (DocumentPage & { object: "document.page" })
  | (Document & { object: "document" });

export interface Picture {
  object: string;
  document_id: string;
  index: number;
  page: number;
  url: string;
  mime: string;
  bytes: number;
  width?: number;
  height?: number;
  dpi?: number;
  bbox?: { l: number; t: number; r: number; b: number; coord_origin: string };
  page_size?: { width: number; height: number };
  caption?: string;
  ref?: string;
  /** Base64, only from `documentPicture`. */
  image?: string;
}

export interface PictureList {
  object: string;
  document_id: string;
  expires_at: string;
  data: Picture[];
}

// --- models, memory, logs ---------------------------------------------------------------------

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  modality: Modality;
  current_tier: MemoryTier;
  placement: Placement;
  active_leases: number;
}

export interface ModelList {
  object: string;
  data: ModelInfo[];
}

export interface ModelSwitch {
  model: string;
  enabled: boolean;
  placement: Placement;
}

export interface MoeOffloadSwitch {
  model: string;
  moe_offload: MoeOffload;
  /** `"next_load"` when the model was busy. */
  applies: "now" | "next_load";
  placement: Placement;
}

export interface TierMove {
  model: string;
  placement: Placement;
}

export interface ModelStatus {
  id: string;
  enabled: boolean;
  modality: Modality;
  modalities: Modality[];
  engine: string;
  total_weight_bytes: number;
  priority: string;
  current_tier: MemoryTier;
  placement: Placement;
  moe_offload?: MoeOffload;
  expert_bytes?: number;
  active_leases: number;
  idle_seconds: number;
  total_requests_served: number;
}

/** One model's parked KV state (`kv_park.models`). */
export interface ParkedModel {
  model: string;
  slots: number;
  tokens: number;
  bytes: number;
  parked_secs_ago: number;
}

/** `GET`/`POST /v1/memory/kv-park`, and `kv_park` in `/v1/memory/status`. */
export interface KvParkStatus {
  enabled: boolean;
  max_bytes: number;
  used_bytes: number;
  models: ParkedModel[];
}

/** `GET /v1/memory/status`; docs/api.md section 14 lists every field. */
export interface MemoryStatus {
  capabilities: Record<string, unknown>;
  tier1_vram_used_bytes: number;
  tier1_vram_free_bytes: number;
  tier2_pinned_ram_used_bytes: number;
  tier2_pinned_ram_free_bytes: number;
  models: ModelStatus[];
  recent_evictions: {
    model: string;
    for_model: string;
    reason: string;
    to_tier: MemoryTier;
    unix_ms: number;
  }[];
  gpu?: Record<string, unknown>;
  kv_park?: KvParkStatus;
  [field: string]: unknown;
}

export interface LogRecord {
  id: string;
  route: string;
  method?: string;
  status: number;
  key?: string;
  started_at?: string;
  duration_ms?: number;
  model?: string;
  stream?: boolean;
  tokens_in?: number;
  tokens_out?: number;
  cached_tokens?: number;
  tokens_per_s?: number;
  inference_ms?: number;
  tier_switch_ms?: number;
  previous_tier?: string;
  error?: string;
  level?: string;
  truncated?: boolean;
  input?: string;
  output?: string;
  request_payload?: unknown;
  response_payload?: unknown;
}

export interface LogList {
  level: string;
  capacity: number;
  stored: number;
  records: LogRecord[];
}

export interface Health {
  status: string;
  service: string;
  version: string;
  uptime_secs: number;
}

/** The `x-hypnos-*` headers of a speech response (docs/api.md section 5). */
export interface SpeechTelemetry {
  previousTier?: string;
  gpuLayers?: number;
  switchMs?: number;
  inferenceMs?: number;
  audioSecs?: number;
  rtf?: number;
}

/** `POST /v1/audio/speech`: the audio bytes, their media type and the telemetry headers. */
export interface Speech {
  audio: Uint8Array;
  contentType: string;
  telemetry: SpeechTelemetry;
  requestId?: string;
}
