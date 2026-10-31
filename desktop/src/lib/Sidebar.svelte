<script>
  let { store } = $props();

  function remove(e, id) {
    e.stopPropagation();
    store.remove(id);
  }
</script>

<nav class="w-60 shrink-0 border-r border-slate-800 bg-slate-900/40 flex flex-col">
  <div class="p-3">
    <button
      class="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm border border-slate-700 hover:bg-slate-800"
      onclick={() => store.newChat()}
      title="New chat (⌘/Ctrl + N)"
    >
      <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14" /></svg>
      New chat
    </button>
  </div>

  <ul class="flex-1 overflow-y-auto px-2 pb-3 space-y-0.5">
    {#each store.chats as chat (chat.id)}
      <li>
        <div
          class="group w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm cursor-pointer
            {chat.id === store.currentId ? 'bg-slate-800 text-slate-100' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'}"
          role="button"
          tabindex="0"
          onclick={() => store.select(chat.id)}
          onkeydown={(e) => e.key === 'Enter' && store.select(chat.id)}
        >
          {#if store.isRunning(chat.id)}
            <span class="w-1.5 h-1.5 shrink-0 rounded-full bg-indigo-400 animate-pulse"></span>
          {/if}
          <span class="flex-1 truncate">{chat.title}</span>
          <button
            class="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400"
            onclick={(e) => remove(e, chat.id)}
            title="Delete chat"
            aria-label="Delete chat"
          >
            <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
      </li>
    {/each}
  </ul>
</nav>
