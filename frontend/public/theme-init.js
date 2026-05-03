// Resolve theme before paint to avoid light/dark flash.
// Lives outside index.html so a strict script-src CSP ('self', no 'unsafe-inline')
// still permits it.
(function () {
  try {
    var stored = localStorage.getItem("podcodex-theme") || "system";
    var isDark = stored === "dark" ||
      (stored === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    if (isDark) document.documentElement.classList.add("dark");
  } catch (e) {}
})();
