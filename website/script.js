// Hypnos website: theme switch, mobile menu, quickstart tabs, copy buttons, cookie consent. No dependencies.
(function () {
  'use strict';

  var root = document.documentElement;

  // Theme: the stored choice wins, otherwise the OS preference applies through CSS.
  function currentTheme() {
    var stored = root.getAttribute('data-theme');
    if (stored === 'light' || stored === 'dark') return stored;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('hypnos-theme', next); } catch (e) { /* storage unavailable */ }
    });
  }

  // Mobile menu
  var menu = document.getElementById('menu-toggle');
  var nav = document.getElementById('nav');
  if (menu && nav) {
    menu.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      menu.setAttribute('aria-expanded', open ? 'true' : 'false');
      menu.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
    nav.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') {
        nav.classList.remove('open');
        menu.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Tabs (quickstart)
  Array.prototype.forEach.call(document.querySelectorAll('[data-tabs]'), function (tabs) {
    var buttons = Array.prototype.slice.call(tabs.querySelectorAll('[role="tab"]'));
    var panels = Array.prototype.slice.call(tabs.querySelectorAll('[role="tabpanel"]'));
    function select(index) {
      buttons.forEach(function (b, i) {
        var on = i === index;
        b.setAttribute('aria-selected', on ? 'true' : 'false');
        b.tabIndex = on ? 0 : -1;
      });
      panels.forEach(function (p, i) { p.hidden = i !== index; });
    }
    buttons.forEach(function (b, i) {
      b.addEventListener('click', function () { select(i); });
      b.addEventListener('keydown', function (e) {
        var dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!dir) return;
        e.preventDefault();
        var next = (i + dir + buttons.length) % buttons.length;
        select(next);
        buttons[next].focus();
      });
    });
  });

  // Cookie consent for Google Analytics (GDPR / ePrivacy): nothing loads before "Accept", declining
  // is one click, and the footer's "Cookie settings" reopens the banner to change the choice.
  var CONSENT_KEY = 'hypnos-consent';
  function storedConsent() {
    try { return localStorage.getItem(CONSENT_KEY); } catch (e) { return null; }
  }
  function clearAnalyticsCookies() {
    var host = location.hostname.split('.');
    document.cookie.split(';').forEach(function (c) {
      var name = c.split('=')[0].trim();
      if (!/^_ga/.test(name)) return;
      // _ga cookies sit on the top-level domain the site is served from; try each suffix.
      for (var i = 0; i < host.length; i++) {
        var domain = host.slice(i).join('.');
        document.cookie = name + '=; Max-Age=0; path=/; domain=' + domain;
      }
      document.cookie = name + '=; Max-Age=0; path=/';
    });
  }
  var banner = null;
  function closeBanner() {
    if (banner) { banner.remove(); banner = null; }
  }
  function decide(choice) {
    try { localStorage.setItem(CONSENT_KEY, choice); } catch (e) { /* storage unavailable */ }
    closeBanner();
    if (choice === 'granted') {
      if (window.hypnosLoadAnalytics) window.hypnosLoadAnalytics();
    } else if (window.hypnosAnalyticsLoaded) {
      // Withdrawn after it was given: stop collection and drop the cookies, then reload so the
      // already-loaded script is gone too.
      if (window.gtag) window.gtag('consent', 'update', { analytics_storage: 'denied' });
      clearAnalyticsCookies();
      location.reload();
    } else {
      clearAnalyticsCookies();
    }
  }
  function showBanner() {
    if (banner) return;
    var privacyLink = document.querySelector('[data-privacy-link]');
    var privacyHref = privacyLink ? privacyLink.getAttribute('href') : 'docs/privacy.html';
    banner = document.createElement('div');
    banner.className = 'cookie-banner';
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-live', 'polite');
    banner.setAttribute('aria-label', 'Cookie consent');
    banner.innerHTML =
      '<p>May we use Google Analytics cookies to count visits and see which pages help? ' +
      'Nothing is stored or sent to Google unless you accept. You can change this any time under ' +
      '&ldquo;Cookie settings&rdquo; at the bottom of the page. ' +
      '<a href="' + privacyHref + '">Privacy policy</a>.</p>' +
      '<div class="cookie-actions">' +
      '<button type="button" class="cookie-btn" data-choice="denied">Decline</button>' +
      '<button type="button" class="cookie-btn" data-choice="granted">Accept</button>' +
      '</div>';
    banner.addEventListener('click', function (e) {
      var choice = e.target.getAttribute && e.target.getAttribute('data-choice');
      if (choice) decide(choice);
    });
    document.body.appendChild(banner);
  }
  var consent = storedConsent();
  if (consent !== 'granted' && consent !== 'denied') showBanner();
  Array.prototype.forEach.call(document.querySelectorAll('[data-cookie-settings]'), function (btn) {
    btn.addEventListener('click', showBanner);
  });

  // Copy buttons on code blocks
  Array.prototype.forEach.call(document.querySelectorAll('.code .copy'), function (btn) {
    btn.addEventListener('click', function () {
      var code = btn.parentNode.querySelector('pre code');
      if (!code || !navigator.clipboard) return;
      navigator.clipboard.writeText(code.textContent).then(function () {
        btn.textContent = 'Copied';
        btn.classList.add('done');
        setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1500);
      }).catch(function () { /* clipboard denied; nothing to do */ });
    });
  });
})();
