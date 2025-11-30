// Initialize mermaid with startOnLoad: false so we can manually trigger it
if (typeof mermaid !== "undefined") {
  mermaid.initialize({ startOnLoad: false });
}

// Hook into MkDocs Material's instant loading event
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (typeof mermaid !== "undefined") {
      // Check for newer v10+ API
      if (mermaid.run) {
        mermaid.run({
          nodes: document.querySelectorAll('.mermaid')
        });
      } 
      // Fallback for older versions
      else if (mermaid.init) {
        mermaid.init(undefined, document.querySelectorAll(".mermaid"));
      }
    }
  });
}