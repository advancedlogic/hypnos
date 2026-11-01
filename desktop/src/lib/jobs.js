/**
 * The jobs the app hands to a model besides the chat itself, as [modality, label]. The model for
 * each is chosen in Settings or in the Models dialog's lineup (the same `routes` map), applied
 * at once without a restart: `route` in `src-tauri/src/hypnos.rs`.
 */
export const JOBS = [
  ['vision', 'Images in messages'],
  ['speech_to_text', 'Speech to text'],
  ['text_to_speech', 'Text to speech'],
  ['image_generation', 'Image generation'],
  ['document_conversion', 'Documents and URLs'],
];

/** Every job a model can be the default for in the Models dialog: the chat, then `JOBS`. */
export const DEFAULT_JOBS = [['text_generation', 'Chat'], ...JOBS];

/** The enabled models (from `hypnos_models`) that can do `modality`. */
export const servingModels = (models, modality) => models.filter((m) => m.modalities.includes(modality));
