// The mock first, then the app once the mock has seeded the store with the scene.
const { ready } = await import('./mock.js');
await ready;
await import('../src/main.js');
