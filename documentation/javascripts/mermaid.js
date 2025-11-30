// Initialize mermaid
// We use a robust check for the new v10+ API (mermaid.run) vs the old API (mermaid.init)

var initMermaid = function() {
  if (typeof mermaid === "undefined") {
    return;
  }
  
  // Configuration
  var config = {
    startOnLoad: false,
    theme: "default",
    flowchart: { htmlLabels: false }
  };
  mermaid.initialize(config);

  // Render
  var elements = document.querySelectorAll(".mermaid");
  if (elements.length > 0) {
    if (mermaid.run) {
      // Mermaid v10+
      mermaid.run({ nodes: elements });
    } else if (mermaid.init) {
      // Mermaid v9-
      mermaid.init(undefined, elements);
    }
  }
};

if (typeof document$ !== "undefined") {
  // MkDocs Material event
  document$.subscribe(function() {
    initMermaid();
  });
} else {
  // Standard fallback
  document.addEventListener("DOMContentLoaded", function() {
    initMermaid();
  });
}