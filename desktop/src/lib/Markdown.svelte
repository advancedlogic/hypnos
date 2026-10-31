<script>
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';

  let { text } = $props();

  let html = $derived(DOMPurify.sanitize(marked.parse(text ?? '', { gfm: true, breaks: true })));

  // Copy buttons on code blocks, by event delegation since the HTML is generated.
  function addCopyButtons(node) {
    const decorate = () => {
      for (const pre of node.querySelectorAll('pre:not([data-copy])')) {
        pre.dataset.copy = '';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'copy-code';
        button.textContent = 'Copy';
        pre.appendChild(button);
      }
    };
    const onClick = async (e) => {
      const button = e.target.closest('.copy-code');
      if (!button) return;
      const code = button.parentElement.querySelector('code')?.innerText ?? '';
      await navigator.clipboard.writeText(code);
      button.textContent = 'Copied';
      setTimeout(() => (button.textContent = 'Copy'), 1200);
    };
    const observer = new MutationObserver(decorate);
    observer.observe(node, { childList: true, subtree: true });
    node.addEventListener('click', onClick);
    decorate();
    return {
      destroy() {
        observer.disconnect();
        node.removeEventListener('click', onClick);
      },
    };
  }
</script>

<div class="markdown prose prose-invert prose-sm max-w-none" use:addCopyButtons>
  {@html html}
</div>

<style>
  .markdown :global(pre) {
    position: relative;
    background: rgb(15 23 42);
    border: 1px solid rgb(30 41 59);
  }
  .markdown :global(.copy-code) {
    position: absolute;
    top: 0.4rem;
    right: 0.4rem;
    font-size: 0.7rem;
    padding: 0.1rem 0.45rem;
    border-radius: 0.3rem;
    color: rgb(148 163 184);
    background: rgb(30 41 59);
    opacity: 0;
    transition: opacity 0.15s;
  }
  .markdown :global(pre:hover .copy-code) {
    opacity: 1;
  }
  .markdown :global(code:not(pre code)) {
    background: rgb(30 41 59);
    padding: 0.1rem 0.3rem;
    border-radius: 0.25rem;
  }
  .markdown :global(code::before),
  .markdown :global(code::after) {
    content: none;
  }
</style>
