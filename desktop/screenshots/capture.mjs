// Takes the user manual's screenshots (docs/desktop-manual.md) into docs/images/desktop/.
//
//   1. a Hypnos server with the lineup on a spare port (it gives the real memory, models,
//      catalog and request log), e.g. ./backend/target/release/hypnos <config with port 9988>
//   2. cd desktop && npx vite --port 1430            (serves screenshots/index.html)
//   3. npm i --no-save playwright-core && node screenshots/capture.mjs [--server URL] [--only name]
//
// The web UI runs in Chrome against screenshots/mock.js in place of the Rust side; the chats are
// screenshots/fixtures.json (recorded answers). Full-window shots are 1x, close-ups 2x.
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '../../docs/images/desktop');
mkdirSync(out, { recursive: true });
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const SERVER = arg('server', 'http://localhost:9988');
const UI = arg('ui', 'http://localhost:1430');
const ONLY = arg('only', null);
const CHROME = arg('chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');

const browser = await chromium.launch({ executablePath: CHROME });

async function open(scene, { scale = 1, panel = true, width = 1360, height = 860 } = {}) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: scale, colorScheme: 'dark' });
  page.on('pageerror', (e) => console.error(`  ${scene}: ${e.message}`));
  await page.goto(`${UI}/screenshots/index.html?scene=${scene}&server=${encodeURIComponent(SERVER)}&panel=${panel}`);
  await page.getByText('Ready', { exact: true }).waitFor();
  await page.waitForTimeout(1500);
  return page;
}

/** Scrolls the open chat to its start (it follows the end while streaming). */
const toTop = (page) =>
  page.evaluate(() => {
    for (const el of document.querySelectorAll('main .overflow-y-auto')) el.scrollTop = 0;
  });

const shot = (page, name, target) =>
  (target ?? page).screenshot({ path: join(out, `${name}.png`), animations: 'disabled' });

const scenes = {
  async 'main-window'() {
    const page = await open('dolomites');
    await toTop(page);
    await shot(page, 'main-window');
    return page;
  },
  async 'thinking-panel'() {
    const page = await open('dolomites', { scale: 2, panel: false, width: 1000, height: 760 });
    await toTop(page);
    await page.getByRole('button', { name: /Thought for/ }).click();
    await page.waitForTimeout(300);
    await shot(page, 'thinking-panel', page.locator('main'));
    return page;
  },
  async 'tool-panel'() {
    const page = await open('tools', { scale: 2, panel: false, width: 1000, height: 900 });
    await toTop(page);
    await page.getByRole('button', { name: /Used 2 tools/ }).click();
    await page.waitForTimeout(300);
    await shot(page, 'tool-panel', page.locator('main'));
    return page;
  },
  async 'image-answer'() {
    const page = await open('image', { panel: false, width: 1100, height: 1000 });
    await page.waitForTimeout(500);
    await shot(page, 'image-answer', page.locator('main'));
    return page;
  },
  async 'document-attachment'() {
    const page = await open('document', { scale: 2, panel: false, width: 1000, height: 700 });
    await toTop(page);
    await shot(page, 'document-attachment', page.locator('main'));
    return page;
  },
  async 'composer-link'() {
    const page = await open('dolomites', { scale: 2, panel: false, width: 1000, height: 700 });
    await page.getByRole('button', { name: /New chat/ }).click();
    await page.locator('textarea').fill('Which peaks are the highest? https://en.wikipedia.org/wiki/Dolomites');
    await page.waitForTimeout(600);
    // The composer and the chips above it: the bottom of the chat column.
    const main = await page.locator('main').boundingBox();
    const box = await page.locator('main textarea').boundingBox();
    const chip = await page.getByText(/^Read en\.wikipedia/).boundingBox();
    const top = Math.max(main.y, Math.min(chip?.y ?? box.y, box.y) - 14);
    await page.screenshot({
      path: join(out, 'composer-link.png'),
      clip: { x: main.x, y: top, width: main.width, height: main.y + main.height - top },
    });
    return page;
  },
  async 'stats-panel'() {
    const page = await open('dolomites', { scale: 2, height: 1500 });
    await shot(page, 'stats-panel', page.locator('aside'));
    return page;
  },
  async 'models-lineup'() {
    const page = await open('dolomites', { scale: 2, width: 1360, height: 1000 });
    await page.getByRole('button', { name: 'Models', exact: true }).click();
    await page.getByRole('tab', { name: 'Lineup' }).click();
    await page.waitForTimeout(1200);
    await shot(page, 'models-lineup', page.locator('div.fixed > div').first());
    return page;
  },
  async 'models-catalog'() {
    const page = await open('dolomites', { scale: 2, width: 1360, height: 1000 });
    await page.getByRole('button', { name: 'Models', exact: true }).click();
    await page.getByRole('tab', { name: /Catalog/ }).waitFor();
    await page.waitForTimeout(1500);
    await shot(page, 'models-catalog', page.locator('div.fixed > div').first());
    return page;
  },
  async 'models-catalog-pool'() {
    const page = await open('dolomites', { scale: 2, width: 1360, height: 1000 });
    await page.getByRole('button', { name: 'Models', exact: true }).click();
    await page.getByRole('tab', { name: /^In the pool/ }).click();
    await page.waitForTimeout(800);
    await shot(page, 'models-catalog-pool', page.locator('div.fixed > div').first());
    return page;
  },
  async settings() {
    const page = await open('dolomites', { scale: 2, width: 1360, height: 1100 });
    await page.getByRole('button', { name: /Ready/ }).click();
    await page.waitForTimeout(800);
    // The URL field selects its text when the dialog opens.
    await page.locator('div.fixed input').first().evaluate((el) => {
      el.setSelectionRange(el.value.length, el.value.length);
      el.blur();
    });
    await shot(page, 'settings', page.locator('div.fixed > div').first());
    return page;
  },
  async 'local-server'() {
    const page = await open('dolomites', { scale: 2, width: 1360, height: 1000 });
    await page.getByRole('button', { name: 'Local server' }).click();
    await page.waitForTimeout(800);
    await shot(page, 'local-server', page.locator('div.fixed > div').first());
    return page;
  },
  async 'guided-setup'() {
    const page = await open('dolomites', { scale: 2, width: 1360, height: 1100 });
    await page.getByRole('button', { name: 'Models', exact: true }).click();
    await page.getByRole('button', { name: /Guided setup/ }).click();
    await page.getByText('Recommended for you').waitFor();
    await page.waitForTimeout(800);
    await shot(page, 'guided-setup', page.locator('div.fixed > div').first());
    return page;
  },
  async 'request-log'() {
    const page = await open('dolomites');
    await page.getByRole('button', { name: 'Request log' }).click();
    await page.waitForTimeout(1500);
    await shot(page, 'request-log');
    return page;
  },
  async 'request-record'() {
    const page = await open('dolomites');
    await page.getByRole('button', { name: 'Request log' }).click();
    await page.waitForTimeout(1500);
    await page.locator('main tbody tr').first().click();
    await page.waitForTimeout(800);
    await shot(page, 'request-record');
    return page;
  },
};

for (const [name, run] of Object.entries(scenes)) {
  if (ONLY && !ONLY.split(',').includes(name)) continue;
  process.stdout.write(`${name}… `);
  try {
    const page = await run();
    await page.close();
    console.log('ok');
  } catch (e) {
    console.log(`failed: ${e.message.split('\n')[0]}`);
  }
}
await browser.close();
