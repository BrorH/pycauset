// Initialize Mermaid in a way that plays nicely with MkDocs Material.
// If the user toggles light/dark mode, we re-render Mermaid with the new theme.
(function () {
  function currentMermaidTheme() {
    // MkDocs Material sets this attribute on <body>
    const scheme = document.body.getAttribute('data-md-color-scheme');
    return scheme === 'slate' ? 'dark' : 'default';
  }

  function renderMermaid() {
    if (!window.mermaid) return;

    const theme = currentMermaidTheme();

    // Re-initialize (safe) so theme changes apply.
    window.mermaid.initialize({
      startOnLoad: false,
      theme,
      securityLevel: 'strict'
    });

    // Clear any previous render artifacts and render again.
    document.querySelectorAll('.mermaid').forEach((el) => {
      // If Mermaid already rendered, it may have injected an <svg> sibling.
      // Keeping the source text node intact gives consistent results.
      const source = el.textContent;
      el.removeAttribute('data-processed');
      el.innerHTML = source;
    });

    window.mermaid.run({ querySelector: '.mermaid' });
  }

  // Initial render
  document.addEventListener('DOMContentLoaded', renderMermaid);

  // Re-render on palette changes (body attribute mutation)
  const obs = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === 'attributes' && m.attributeName === 'data-md-color-scheme') {
        renderMermaid();
        break;
      }
    }
  });

  document.addEventListener('DOMContentLoaded', () => {
    obs.observe(document.body, { attributes: true });
  });
})();
