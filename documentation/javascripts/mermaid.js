document$.subscribe(() => {
  mermaid.init(undefined, document.querySelectorAll(".mermaid"));
});