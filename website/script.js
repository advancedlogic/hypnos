// Hypnos website: theme switch, mobile menu, quickstart tabs, copy buttons. No dependencies.
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
