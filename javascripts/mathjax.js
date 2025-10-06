window.MathJax = {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    packages: { "[+]": ["ams"] },
    macros: {
      TeV: "\\,\\mathrm{TeV}",
      GeV: "\\,\\mathrm{GeV}",
      MeV: "\\,\\mathrm{MeV}",
      ptmiss: "\\,p_{T}^{\\mathrm{miss}}",
    },
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"],
  },
};
