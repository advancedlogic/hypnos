// Renders the documentation pages of the website from the Markdown in docs/:
//
//   cd website && npm install && npm run build:docs
//
// Writes website/docs/<name>.html for each entry of PAGES (the site's header, a table of
// contents from the headings, the article, the footer), website/docs/index.html (the landing
// page) and copies the screenshots the manual uses into website/docs/images/. The pages share
// styles.css and script.js with the main page (theme switch, copy buttons) plus docs.css.
//
// Markdown conventions handled: GitHub-style heading anchors (the docs link to their own
// sections with them), fenced code blocks as the site's code figures with a copy button,
// tables in a scrolling wrapper, links to other .md files turned into links to their page here
// or to the file on GitHub, and each document's own "Contents" list dropped in favour of the
// sidebar.

import { Marked } from 'marked';
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, readdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const website = join(here, '..');
const repo = join(website, '..');
const docsDir = join(repo, 'docs');
const outDir = join(website, 'docs');
const GITHUB = 'https://github.com/advancedlogic/hypnos';

const PAGES = [
  {
    file: 'desktop-manual.md',
    out: 'desktop-manual.html',
    nav: 'Desktop manual',
    title: 'Hypnos Desktop manual',
    description: 'The user manual of the Hypnos desktop app: installing, the guided setup, chatting, attachments, voice, images, tools, models, settings, the local server and troubleshooting.',
    blurb: 'How to install the app, let it set itself up, chat with attachments, voice and images, choose models, and what every panel shows.',
    audience: 'For everyone who uses the app.',
  },
  {
    file: 'api.md',
    out: 'api.html',
    nav: 'API reference',
    title: 'Hypnos HTTP API reference',
    description: 'Every route of the Hypnos server, request and response fields, streaming formats, errors, telemetry and the behaviour a client has to handle.',
    blurb: 'Every route with its request and response fields, the streaming formats, the error shape and the telemetry on each response.',
    audience: 'For client and SDK authors.',
  },
  {
    file: 'operating.md',
    out: 'operating.html',
    nav: 'Operating guide',
    title: 'Operating Hypnos',
    description: 'Sizing a machine for a lineup, which knob to turn for which symptom, reading the telemetry, what each error means, logs, exposure, upgrades and running Hypnos as a service.',
    blurb: 'Sizing a machine, which knob to turn for which symptom, reading the telemetry and the errors, logs, exposure, upgrades, running it as a service.',
    audience: 'For whoever runs a server.',
  },
];

// GitHub's heading anchors: lower case, punctuation dropped, spaces to hyphens, duplicates
// numbered. The documents' own cross-references rely on them.
function slugger() {
  const seen = new Map();
  return (text) => {
    let slug = text
      .toLowerCase()
      .replace(/<[^>]+>/g, '')
      .replace(/&[a-z]+;|&#\d+;/g, '')
      .replace(/[^\p{L}\p{N} _-]/gu, '')
      .trim()
      .replace(/ /g, '-');
    const n = seen.get(slug) ?? 0;
    seen.set(slug, n + 1);
    return n ? `${slug}-${n}` : slug;
  };
}

const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function render(markdown, page) {
  const slug = slugger();
  const toc = [];
  const pageByFile = new Map(PAGES.map((p) => [p.file, p.out]));

  const marked = new Marked({
    gfm: true,
    renderer: {
      heading({ tokens, depth }) {
        const html = this.parser.parseInline(tokens);
        const text = this.parser.parseInline(tokens, this.parser.textRenderer);
        if (depth === 1) return `<h1>${html}</h1>\n`;
        const id = slug(text);
        if (depth === 2 || depth === 3) toc.push({ depth, id, html });
        return `<h${depth} id="${id}">${html}<a class="anchor" href="#${id}" aria-label="Link to this section">#</a></h${depth}>\n`;
      },
      code({ text, lang }) {
        const label = (lang || '').trim() || 'text';
        return `<figure class="code"><figcaption>${escapeHtml(label)}</figcaption><pre><code>${escapeHtml(text)}</code></pre><button class="copy" type="button" aria-label="Copy">Copy</button></figure>\n`;
      },
      image({ href, title, text }) {
        const t = title ? ` title="${escapeHtml(title)}"` : '';
        return `<figure class="shot"><img src="${href}" alt="${escapeHtml(text)}" loading="lazy"${t}><figcaption>${escapeHtml(text)}</figcaption></figure>`;
      },
    },
  });

  // marked applies walkTokens in parse() only; the tokens are lexed and parsed separately
  // here (to drop the Contents list), so the link rewriting runs by hand.
  const rewrite = (token) => {
      if (token.type !== 'link' && token.type !== 'image') return;
      const href = token.href;
      if (/^(https?:|mailto:|#)/.test(href)) return;
      if (token.type === 'image') {
        // docs/images/... is copied to website/docs/images/...
        token.href = href.replace(/^\.\//, '');
        return;
      }
      const m = href.match(/^([^#]*)(#.*)?$/);
      const [, path, frag = ''] = m;
      const name = basename(path);
      if (pageByFile.has(name)) {
        token.href = pageByFile.get(name) + frag;
      } else {
        // Anything else lives in the repository: the README, other docs, config files.
        const rel = path.startsWith('../') ? path.slice(3) : `docs/${path}`;
        token.href = `${GITHUB}/blob/main/${rel}${frag}`;
      }
  };

  // Drop the document's own "Contents" list (the sidebar replaces it) and the rule after it.
  const tokens = marked.lexer(markdown);
  marked.walkTokens(tokens, rewrite);
  const kept = [];
  let skipping = false;
  for (const t of tokens) {
    if (t.type === 'heading' && t.depth === 2 && /^contents$/i.test(t.text)) {
      skipping = true;
      continue;
    }
    if (skipping) {
      if (t.type === 'heading') skipping = false;
      else continue;
    }
    kept.push(t);
  }
  // The h1 becomes the page title.
  const h1 = kept.find((t) => t.type === 'heading' && t.depth === 1);
  const body = marked.parser(kept.filter((t) => t !== h1));
  const article = body
    .replace(/<table>/g, '<div class="table-wrap"><table>')
    .replace(/<\/table>/g, '</table></div>')
    // A figure cannot sit inside a paragraph (a screenshot on a line of its own).
    .replace(/<p><figure class="shot">/g, '<figure class="shot">')
    .replace(/<\/figure><\/p>/g, '</figure>');
  return { article, toc, heading: h1 ? marked.parser([h1]).replace(/<\/?h1[^>]*>/g, '').trim() : page.title };
}

function layout({ title, description, nav, main, crumb }) {
  const links = PAGES.map((p) => `<a href="${p.out}"${p.nav === nav ? ' aria-current="page"' : ''}>${p.nav}</a>`).join('\n        ');
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script>
    // Google Analytics, loaded only after the visitor accepts it in the cookie banner (script.js).
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('consent', 'default', { ad_storage: 'denied', ad_user_data: 'denied', ad_personalization: 'denied', analytics_storage: 'denied' });
    window.hypnosLoadAnalytics = function () {
      if (window.hypnosAnalyticsLoaded) return;
      window.hypnosAnalyticsLoaded = true;
      gtag('consent', 'update', { analytics_storage: 'granted' });
      var s = document.createElement('script');
      s.async = true;
      s.src = 'https://www.googletagmanager.com/gtag/js?id=G-XWGGG9649G';
      document.head.appendChild(s);
      gtag('js', new Date());
      gtag('config', 'G-XWGGG9649G');
    };
    try { if (localStorage.getItem('hypnos-consent') === 'granted') window.hypnosLoadAnalytics(); } catch (e) {}
  </script>
  <title>${escapeHtml(title)} · Hypnos</title>
  <meta name="description" content="${escapeHtml(description)}">
  <meta name="color-scheme" content="dark light">
  <meta name="theme-color" content="#0d0f43">
  <meta property="og:title" content="${escapeHtml(title)} · Hypnos">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:image" content="../assets/logo-256.png">
  <link rel="icon" type="image/png" sizes="64x64" href="../assets/favicon.png">
  <link rel="apple-touch-icon" href="../assets/logo-256.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../styles.css">
  <link rel="stylesheet" href="../docs.css">
  <script>
    // Apply the stored theme before the first paint to avoid a flash.
    try {
      var t = localStorage.getItem('hypnos-theme');
      if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
    } catch (e) {}
  </script>
</head>
<body class="docs-page">
  <a class="skip" href="#main">Skip to content</a>

  <header class="site-header">
    <div class="container header-row">
      <a class="brand" href="../index.html" aria-label="Hypnos home">
        <img src="../assets/logo-256.png" alt="" width="36" height="36">
        <span>Hypnos</span>
      </a>
      <nav class="nav" id="nav" aria-label="Documentation">
        <a href="index.html"${nav === 'Docs' ? ' aria-current="page"' : ''}>Docs</a>
        ${links}
        <a href="${GITHUB}" rel="noopener">GitHub</a>
      </nav>
      <div class="header-actions">
        <button class="icon-btn" id="theme-toggle" type="button" aria-label="Switch theme" title="Switch theme">
          <svg class="sun" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
          <svg class="moon" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>
        </button>
        <button class="icon-btn menu-btn" id="menu-toggle" type="button" aria-label="Open menu" aria-expanded="false" aria-controls="nav">
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        </button>
      </div>
    </div>
  </header>

  <main id="main">
    <div class="container">
      <p class="crumbs"><a href="../index.html">Hypnos</a> › <a href="index.html">Docs</a>${crumb ? ` › <span>${escapeHtml(crumb)}</span>` : ''}</p>
${main}
    </div>
  </main>

  <footer class="site-footer">
    <div class="container footer-grid">
      <div class="footer-brand">
        <img src="../assets/logo-256.png" alt="" width="40" height="40">
        <div>
          <strong>Hypnos</strong>
          <p>OpenAI-compatible multi-modal inference server with a 3-tier memory scheduler.</p>
        </div>
      </div>
      <nav class="footer-links" aria-label="Project links">
        <a href="${GITHUB}" rel="noopener">Source</a>
        <a href="${GITHUB}/releases" rel="noopener">Releases</a>
        <a href="index.html">Docs</a>
        <a href="${GITHUB}/blob/main/README.md" rel="noopener">README</a>
        <a href="${GITHUB}/blob/main/backend/hypnos.example.toml" rel="noopener">Example config</a>
        <a href="privacy.html" data-privacy-link>Privacy</a>
        <button type="button" class="footer-link-button" data-cookie-settings>Cookie settings</button>
      </nav>
      <p class="footer-license">Source code under Apache-2.0 or MIT, at your option. Binaries built with the default engines link espeak-ng and are distributed under GPL-3.0-or-later. Model weights carry their own licenses.</p>
    </div>
  </footer>

  <script src="../script.js" defer></script>
</body>
</html>
`;
}

function tocHtml(toc) {
  if (!toc.length) return '';
  const items = toc.map((t) => `<li class="l${t.depth}"><a href="#${t.id}">${t.html}</a></li>`).join('\n          ');
  return `      <details class="docs-toc" open>
        <summary>On this page</summary>
        <ol>
          ${items}
        </ol>
      </details>`;
}

mkdirSync(outDir, { recursive: true });

for (const page of PAGES) {
  const markdown = readFileSync(join(docsDir, page.file), 'utf8');
  const { article, toc, heading } = render(markdown, page);
  const main = `      <div class="docs-layout">
${tocHtml(toc)}
        <article class="prose">
          <h1>${heading}</h1>
          <p class="docs-source">Rendered from <a href="${GITHUB}/blob/main/docs/${page.file}" rel="noopener"><code>docs/${page.file}</code></a>; edits go there.</p>
${article}
        </article>
      </div>`;
  writeFileSync(join(outDir, page.out), layout({ title: page.title, description: page.description, nav: page.nav, main, crumb: page.nav }));
  console.log(`wrote docs/${page.out} (${toc.length} sections)`);
}

// Landing page.
const cards = PAGES.map(
  (p) => `          <a class="card doc-card" href="${p.out}">
            <h3>${escapeHtml(p.nav)}</h3>
            <p>${escapeHtml(p.blurb)}</p>
            <p class="doc-audience">${escapeHtml(p.audience)}</p>
          </a>`,
).join('\n');
const landing = `      <div class="section-head docs-head">
        <p class="eyebrow">Documentation</p>
        <h1>Hypnos documentation</h1>
        <p class="section-lede">Three guides, one per reader: the person using the desktop app, the one writing a client, and the one running a server. The README on GitHub is the introduction; <code>backend/hypnos.example.toml</code> documents every option.</p>
      </div>
      <div class="feature-grid doc-grid">
${cards}
      </div>
      <div class="docs-more">
        <h2>More on GitHub</h2>
        <ul>
          <li><a href="${GITHUB}/blob/main/README.md" rel="noopener">README</a>: what Hypnos is, the model lineup, the endpoint table and the quickstart.</li>
          <li><a href="${GITHUB}/blob/main/backend/hypnos.example.toml" rel="noopener">hypnos.example.toml</a>: the reference config, every option with a comment.</li>
          <li><a href="${GITHUB}/blob/main/desktop/README.md" rel="noopener">desktop/README.md</a>: the desktop app for developers: building it, how it reaches the server, the bundled server.</li>
          <li><a href="${GITHUB}/blob/main/docs/apple-silicon.md" rel="noopener">docs/apple-silicon.md</a>: unified memory, Metal's working set, the design record for the Mac.</li>
          <li><a href="${GITHUB}/blob/main/docs/signing.md" rel="noopener">docs/signing.md</a>: signing and notarizing the macOS and Windows builds.</li>
          <li><a href="${GITHUB}/tree/main/docs/bench" rel="noopener">docs/bench</a>: the measured runs on CUDA and Apple Silicon.</li>
          <li><a href="${GITHUB}/tree/main/sdk" rel="noopener">sdk/</a>: the Python, JavaScript and Rust clients.</li>
          <li><a href="${GITHUB}/blob/main/CHANGELOG.md" rel="noopener">CHANGELOG.md</a>: what each release changed.</li>
        </ul>
      </div>`;
writeFileSync(join(outDir, 'index.html'), layout({ title: 'Documentation', description: 'The Hypnos documentation: the desktop app manual, the HTTP API reference and the operating guide.', nav: 'Docs', main: landing, crumb: '' }));
console.log('wrote docs/index.html');

// Privacy policy (GDPR art. 13). Keep it in step with what the pages load: the analytics tag in
// layout() and index.html, the consent banner in script.js, the fonts, and what goes in storage.
const privacy = `      <article class="prose">
        <h1>Privacy policy</h1>
        <p class="docs-source">Last updated 3 November 2026.</p>
        <p>This policy covers this website (the Hypnos landing page and its documentation). It does not cover the Hypnos software: the server and the desktop app run on your own machine and send nothing to us.</p>

        <h2 id="who">Who is responsible</h2>
        <p>The site is run by Antonio Linari, the maintainer of Hypnos (the data controller under the GDPR). For any question or request about your data, contact the maintainer through GitHub at <a href="https://github.com/advancedlogic" rel="noopener">github.com/advancedlogic</a>.</p>

        <h2 id="always">What happens on every visit</h2>
        <ul>
          <li><strong>Hosting.</strong> Like any web server, the server that delivers these pages receives your IP address, the page requested, the time, the referring page and your browser's user agent, which it needs to send you the page and to keep the service secure (legitimate interest, art. 6(1)(f) GDPR). We do not use these logs to profile anyone.</li>
          <li><strong>Fonts.</strong> The pages load their typefaces (Inter, Space Grotesk) from Google Fonts, so your browser sends your IP address and user agent to Google when it fetches them (legitimate interest in a consistent presentation, art. 6(1)(f) GDPR). See <a href="https://developers.google.com/fonts/faq/privacy" rel="noopener">Google Fonts' privacy FAQ</a>.</li>
          <li><strong>Your browser's storage.</strong> The site keeps two values in your browser's local storage: your theme choice (<code>hypnos-theme</code>) and your cookie choice (<code>hypnos-consent</code>). They stay on your device, are never sent to anyone, and you can delete them by clearing the site's data.</li>
        </ul>

        <h2 id="analytics">Google Analytics, only with your consent</h2>
        <p>If you click <em>Accept</em> in the cookie banner, the site loads Google Analytics 4 to count visits and see which pages are useful. Nothing from Google Analytics is loaded and no analytics cookie is set before that, or if you click <em>Decline</em>.</p>
        <ul>
          <li><strong>What is collected:</strong> the pages you view, how you reached the site, your browser, device type, screen size, language and approximate location (country or city, derived from your IP address, which Google Analytics 4 does not store), and a random identifier.</li>
          <li><strong>Cookies:</strong> <code>_ga</code> and <code>_ga_&lt;id&gt;</code>, which recognise a returning browser; they last up to two years.</li>
          <li><strong>Legal basis:</strong> your consent (art. 6(1)(a) GDPR and the ePrivacy rules on cookies).</li>
          <li><strong>Recipient:</strong> Google Ireland Limited, Gordon House, Barrow Street, Dublin 4, Ireland, as our processor. Data may be transferred to Google LLC in the United States, which is certified under the EU-US Data Privacy Framework.</li>
          <li><strong>Retention:</strong> the analytics data is deleted after the retention period set for the property (two months).</li>
          <li><strong>Withdrawing consent:</strong> click <em>Cookie settings</em> at the bottom of any page and choose <em>Decline</em>; the site stops loading Google Analytics and deletes its cookies. Withdrawal does not affect what was collected before it.</li>
        </ul>
        <p>More in <a href="https://policies.google.com/technologies/partner-sites" rel="noopener">how Google uses data from sites that use its services</a> and <a href="https://policies.google.com/privacy" rel="noopener">Google's privacy policy</a>.</p>

        <h2 id="rights">Your rights</h2>
        <p>You have the right to access the data about you, to have it corrected or deleted, to restrict or object to its processing, to data portability, and to withdraw consent at any time. You also have the right to lodge a complaint with a data protection supervisory authority, in particular the one where you live or work.</p>
        <p>We do not sell your data, use it for advertising or make automated decisions about you.</p>
      </article>`;
writeFileSync(join(outDir, 'privacy.html'), layout({ title: 'Privacy policy', description: 'What the Hypnos website collects, why, and how to withdraw consent for Google Analytics.', nav: 'Privacy', main: privacy, crumb: 'Privacy' }));
console.log('wrote docs/privacy.html');

// The manual's screenshots.
const imgSrc = join(docsDir, 'images', 'desktop');
const imgOut = join(outDir, 'images', 'desktop');
mkdirSync(imgOut, { recursive: true });
let copied = 0;
for (const f of readdirSync(imgSrc)) {
  if (!/\.(png|jpg|jpeg|gif|webp)$/i.test(f)) continue;
  copyFileSync(join(imgSrc, f), join(imgOut, f));
  copied++;
}
console.log(`copied ${copied} screenshots`);
