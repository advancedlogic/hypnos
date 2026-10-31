import { convertFileSrc } from '@tauri-apps/api/core';

const FRAMES = 6;
const MAX_SIDE = 768;

/**
 * `count` frames spread over the video at `path`, as JPEG data URLs no larger than 768 px. The
 * webview decodes the video (the asset protocol serves the file; Rust allowed this path only).
 */
export async function sampleFrames(path, count = FRAMES) {
  const video = document.createElement('video');
  video.muted = true;
  video.preload = 'auto';
  video.src = convertFileSrc(path);
  await once(video, 'loadeddata');
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const ctx = canvas.getContext('2d');
  const frames = [];
  for (let i = 1; i <= count; i++) {
    video.currentTime = (duration * i) / (count + 1);
    await once(video, 'seeked');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(canvas.toDataURL('image/jpeg', 0.8));
  }
  video.removeAttribute('src');
  video.load();
  return frames;
}

function once(el, event) {
  return new Promise((resolve, reject) => {
    const ok = () => (cleanup(), resolve());
    const fail = () => (cleanup(), reject(new Error('this video format cannot be decoded here')));
    const cleanup = () => {
      el.removeEventListener(event, ok);
      el.removeEventListener('error', fail);
    };
    el.addEventListener(event, ok, { once: true });
    el.addEventListener('error', fail, { once: true });
  });
}
