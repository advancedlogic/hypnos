"""Response and message shapes of the Hypnos HTTP API (docs/api.md).

They are ``TypedDict``s: plain dicts with type hints. The server only ever adds fields
(docs/api.md section 17, "Versioning"), and unknown fields pass through untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from typing_extensions import NotRequired

MemoryTier = Literal["tier1_gpu_vram", "tier2_pinned_ram", "tier3_nvme_ssd"]
MoeOffload = Literal["off", "auto", "on"]
Modality = Literal[
    "text_generation",
    "vision",
    "speech_to_text",
    "text_to_speech",
    "embeddings",
    "image_generation",
    "document_conversion",
    "entity_extraction",
]


# --- shared ----------------------------------------------------------------------------------


class Placement(TypedDict):
    primary_tier: MemoryTier
    gpu_layers: int
    pinned_ram_layers: int
    ssd_layers: int
    vram_weight_bytes: int
    vram_kv_and_workspace_bytes: int
    pinned_ram_bytes: int
    allocated_context_len: int
    cpu_expert_layers: NotRequired[int]
    """Layers whose MoE expert tensors run on the CPU (``moe_offload``); absent on older servers."""


class EngineTiming(TypedDict, total=False):
    queue_ms: float
    context_ms: float
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    cached_tokens: int
    concurrent_slots: int
    audio_secs: float
    rtf: float
    ttfa_ms: float
    pages: int
    ms_per_page: float


class LoadPhases(TypedDict, total=False):
    load_mode: str
    weights_ms: float
    projector_ms: float
    context_ms: float
    total_ms: float
    kv_restore_ms: float
    kv_restored_tokens: int


class Telemetry(TypedDict):
    """``hypnos_telemetry``: where the model was, what loading it cost, engine timings."""

    previous_tier: MemoryTier
    placement: Placement
    tier_switch_latency_ms: float
    inference_latency_ms: float
    evicted_models: List[str]
    engine: EngineTiming
    tier_switch_phases: NotRequired[LoadPhases]


# --- chat ------------------------------------------------------------------------------------


class ImageUrl(TypedDict):
    url: str
    detail: NotRequired[str]


class InputAudio(TypedDict):
    data: str
    format: NotRequired[str]


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


class AudioPart(TypedDict):
    type: Literal["input_audio"]
    input_audio: InputAudio


ContentPart = Union[TextPart, ImagePart, AudioPart]


class FunctionSpec(TypedDict):
    """A function the model may call: ``parameters`` is a JSON Schema object."""

    name: str
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]


class ToolSpec(TypedDict):
    """An entry of ``tools`` (docs/api.md section 7.5)."""

    type: Literal["function"]
    function: FunctionSpec


class ToolChoiceFunctionName(TypedDict):
    name: str


class ToolChoiceFunction(TypedDict):
    type: Literal["function"]
    function: ToolChoiceFunctionName


#: ``"none"``, ``"auto"``, ``"required"`` or ``{"type": "function", "function": {"name": ...}}``.
ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceFunction]

#: Thinking level of a chat request (``reasoning_effort``, docs/api.md section 7): ``"none"``
#: turns thinking off, the others turn it on with a budget of thinking tokens (512, 2048, 6144).
ReasoningEffort = Literal["none", "low", "medium", "high"]


class ToolCallFunction(TypedDict):
    name: str
    #: A JSON object serialized as a string; :func:`hypnos_client.parse_arguments` decodes it.
    arguments: str


class ToolCall(TypedDict):
    """A call the model asked for, in ``message.tool_calls``."""

    id: str
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessage(TypedDict):
    """A message of ``messages``.

    ``content`` is ``None`` on an assistant message that only calls tools (``tool_calls``); a
    ``tool`` message carries a result for ``tool_call_id``.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[ContentPart], None]
    tool_calls: NotRequired[List[ToolCall]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


class PromptTokensDetails(TypedDict):
    cached_tokens: int


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: NotRequired[PromptTokensDetails]


class ResponseMessage(TypedDict):
    role: str
    #: ``""`` when the model only called tools.
    content: str
    #: The model's thoughts before the answer, when thinking was on and the model thinks.
    reasoning_content: NotRequired[str]
    #: Present only when the model called tools (``finish_reason`` is ``"tool_calls"``).
    tool_calls: NotRequired[List[ToolCall]]


class ChatChoice(TypedDict):
    index: int
    message: ResponseMessage
    finish_reason: str


class ChatCompletion(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage
    hypnos_telemetry: NotRequired[Telemetry]


class ChunkToolCallFunction(TypedDict, total=False):
    name: str
    arguments: str


class ChunkToolCall(TypedDict):
    """An element of ``delta.tool_calls``: accumulate by ``index``."""

    index: int
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]
    function: NotRequired[ChunkToolCallFunction]


class ChunkDelta(TypedDict, total=False):
    role: str
    content: str
    #: A piece of the model's thoughts; all of them arrive before the first ``content``.
    reasoning_content: str
    tool_calls: List[ChunkToolCall]


class ChunkChoice(TypedDict):
    index: int
    delta: ChunkDelta
    finish_reason: Optional[str]


class ChatCompletionChunk(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChunkChoice]
    usage: NotRequired[Usage]
    hypnos_telemetry: NotRequired[Telemetry]
    error: NotRequired[Dict[str, Any]]


# --- embeddings, audio, images ---------------------------------------------------------------


class Embedding(TypedDict):
    object: str
    index: int
    embedding: List[float]


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(TypedDict):
    object: str
    model: str
    data: List[Embedding]
    usage: EmbeddingUsage
    hypnos_telemetry: NotRequired[Telemetry]


class Transcription(TypedDict):
    text: str
    language: NotRequired[str]
    duration_secs: NotRequired[float]
    hypnos_telemetry: NotRequired[Telemetry]


class GeneratedImage(TypedDict):
    b64_json: str
    revised_prompt: NotRequired[str]


class ImageResponse(TypedDict):
    created: int
    data: List[GeneratedImage]
    hypnos_telemetry: NotRequired[Telemetry]


# --- documents -------------------------------------------------------------------------------


class DocumentPage(TypedDict):
    page: int
    markdown: str
    pictures: int
    tables: int
    ms: float
    json: NotRequired[Dict[str, Any]]
    picture_urls: NotRequired[List[str]]


class DocumentUsage(TypedDict):
    pages: int
    pictures: int
    tables: int
    pictures_described: NotRequired[int]


class Document(TypedDict):
    object: str
    model: str
    page_count: int
    pages: List[DocumentPage]
    markdown: str
    usage: DocumentUsage
    id: NotRequired[str]
    pictures_expire_at: NotRequired[str]
    filename: NotRequired[str]
    hypnos_telemetry: NotRequired[Telemetry]


class DocumentStarted(TypedDict):
    object: Literal["document.started"]
    page_count: int
    pages: List[int]
    id: NotRequired[str]


class PictureBox(TypedDict):
    l: float  # noqa: E741 (docling's field name)
    t: float
    r: float
    b: float
    coord_origin: str


class PageSize(TypedDict):
    width: float
    height: float


class Picture(TypedDict):
    object: str
    document_id: str
    index: int
    page: int
    url: str
    mime: str
    bytes: int
    width: NotRequired[int]
    height: NotRequired[int]
    dpi: NotRequired[int]
    bbox: NotRequired[PictureBox]
    page_size: NotRequired[PageSize]
    caption: NotRequired[str]
    ref: NotRequired[str]
    image: NotRequired[str]


class PictureList(TypedDict):
    object: str
    document_id: str
    expires_at: str
    data: List[Picture]


# --- entities --------------------------------------------------------------------------------


class Entity(TypedDict):
    text: str
    label: str
    start: int
    end: int
    score: float


class EntityResult(TypedDict):
    object: str
    index: int
    entities: List[Entity]


class EntityUsage(TypedDict):
    texts: int
    characters: int


class EntityResponse(TypedDict):
    object: str
    model: str
    data: List[EntityResult]
    usage: EntityUsage
    hypnos_telemetry: NotRequired[Telemetry]


# --- models, memory, logs --------------------------------------------------------------------


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int
    owned_by: str
    modality: Modality
    current_tier: MemoryTier
    placement: Placement
    active_leases: int


class ModelList(TypedDict):
    object: str
    data: List[ModelInfo]


class ModelSwitch(TypedDict):
    model: str
    enabled: bool
    placement: Placement


ModelPoolAction = Literal["added", "replaced", "removed"]


class ModelPoolChange(TypedDict):
    """``POST /v1/models``, ``PUT`` / ``DELETE /v1/models/{id}``."""

    model: str
    action: ModelPoolAction
    #: False when the overlay next to the config could not be written (live until a restart).
    persisted: bool
    #: Absent on a removal.
    placement: NotRequired[Placement]


class MoeOffloadSwitch(TypedDict):
    """``POST /v1/models/{id}/moe-offload``."""

    model: str
    moe_offload: MoeOffload
    applies: Literal["now", "next_load"]
    placement: Placement


CatalogCapability = Literal["thinking", "tools"]
CatalogFileRole = Literal["weights", "aux", "text_encoder", "tokenizer", "voices", "data"]
CatalogPlatform = Literal["all", "macos-arm64", "other"]


class CatalogFile(TypedDict):
    """One file a catalog variant needs (``GET /v1/models/catalog``)."""

    role: CatalogFileRole
    #: A Hugging Face repository or an ``https://`` base URL ``remote`` is appended to.
    repo: str
    remote: str
    #: Path under ``models_dir``.
    local: str
    entry: NotRequired[str]
    size: int
    platform: CatalogPlatform
    present: bool
    outdated: NotRequired[bool]


class CatalogVariant(TypedDict):
    quant: str
    default: bool
    note: NotRequired[str]
    size_bytes: int
    files: List[CatalogFile]
    installed: bool
    outdated: bool
    problem: NotRequired[str]
    #: The ``[[models]]`` TOML block that runs this variant.
    config_block: str


class CatalogEntry(TypedDict):
    id: str
    name: str
    publisher: str
    license: str
    architecture: str
    modality: Modality
    additional_modalities: NotRequired[List[Modality]]
    engine: str
    capabilities: NotRequired[List[CatalogCapability]]
    summary: str
    lineup: bool
    configured: bool
    #: Only when ``configured``.
    enabled: NotRequired[bool]
    variants: List[CatalogVariant]


class ModelCatalog(TypedDict):
    """``GET /v1/models/catalog``; see docs/api.md section 6."""

    object: str
    catalog_version: int
    models_dir: str
    tested_architectures: List[str]
    data: List[CatalogEntry]


AdvisorNeed = Literal[
    "chat",
    "vision",
    "voice",
    "transcription",
    "speech",
    "embeddings",
    "images",
    "documents",
    "entities",
]
ModelPriority = Literal["realtime_pinned", "high", "low"]


class AdvisorMachine(TypedDict):
    """The machine the advice is for (``POST /v1/advisor``)."""

    vram_bytes: int
    ram_bytes: int
    unified_memory: bool


class AdvisorBudgets(TypedDict):
    """The tier budgets the server would derive on that machine."""

    tier1_bytes: int
    tier2_bytes: int
    unified_memory: bool


class AdvisedModel(TypedDict):
    """One model of the recommended lineup."""

    id: str
    name: str
    quant: str
    #: The needs this model is the lineup's choice for.
    serves: List[AdvisorNeed]
    weights_bytes: int
    resident_bytes: int
    runs_on_cpu: bool
    gpu_layers: int
    num_layers: int
    cpu_expert_layers: int
    context_len: int
    slots: int
    initial_tier: MemoryTier
    priority: ModelPriority
    keep_warm_in_ram: bool
    moe_offload: MoeOffload
    #: Rough load times (3 GB/s).
    load_secs_cold: float
    load_secs_warm: float


class Advice(TypedDict):
    """``POST /v1/advisor``; see docs/api.md section 6."""

    machine: AdvisorMachine
    budgets: AdvisorBudgets
    models: List[AdvisedModel]
    #: The interactive models (chat, vision, speech, embeddings) fit in Tier 1 together.
    interactive_coresident: bool
    interactive_bytes: int
    warnings: List[str]
    #: ``"<id>:<quant>"`` for ``./run.sh models`` and ``add_model(catalog=...)``.
    specs: List[str]
    #: A complete config file for the lineup.
    config: str


class TierMove(TypedDict):
    model: str
    placement: Placement


class Eviction(TypedDict):
    model: str
    for_model: str
    reason: str
    to_tier: MemoryTier
    unix_ms: int


class ModelStatus(TypedDict):
    id: str
    enabled: bool
    modality: Modality
    modalities: List[Modality]
    engine: str
    total_weight_bytes: int
    priority: str
    current_tier: MemoryTier
    placement: Placement
    moe_offload: NotRequired[MoeOffload]
    expert_bytes: NotRequired[int]
    active_leases: int
    idle_seconds: float
    total_requests_served: int


class ParkedModel(TypedDict):
    """One model's parked KV state (``kv_park.models``)."""

    model: str
    slots: int
    tokens: int
    bytes: int
    parked_secs_ago: float


class KvParkStatus(TypedDict):
    """``GET``/``POST /v1/memory/kv-park`` and ``kv_park`` in ``/v1/memory/status``."""

    enabled: bool
    max_bytes: int
    used_bytes: int
    models: List[ParkedModel]


class MemoryStatus(TypedDict):
    """``GET /v1/memory/status``; see docs/api.md section 14 for every field."""

    capabilities: Dict[str, Any]
    tier1_vram_used_bytes: int
    tier1_vram_free_bytes: int
    tier2_pinned_ram_used_bytes: int
    tier2_pinned_ram_free_bytes: int
    models: List[ModelStatus]
    recent_evictions: List[Eviction]
    gpu: NotRequired[Dict[str, Any]]
    kv_park: NotRequired[KvParkStatus]


class LogRecord(TypedDict, total=False):
    id: str
    route: str
    method: str
    status: int
    key: str
    started_at: str
    duration_ms: float
    model: str
    stream: bool
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    tokens_per_s: float
    inference_ms: float
    tier_switch_ms: float
    previous_tier: str
    error: str
    level: str
    truncated: bool
    input: str
    output: str
    request_payload: Any
    response_payload: Any


class LogList(TypedDict):
    level: str
    capacity: int
    stored: int
    records: List[LogRecord]


class Health(TypedDict):
    status: str
    service: str
    version: str
    uptime_secs: int
