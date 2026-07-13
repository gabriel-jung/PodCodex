/** Sidebar geometry shared by AppSidebar (fixed-positioned) and RootLayout
 *  (content padding). Both derive the width here so they never drift; keep
 *  the literal class names so Tailwind's JIT picks all four up. */

export const sidebarWidth = (expanded: boolean): string =>
  expanded ? "w-48" : "w-14";
export const sidebarPad = (expanded: boolean): string =>
  expanded ? "pl-48" : "pl-14";
