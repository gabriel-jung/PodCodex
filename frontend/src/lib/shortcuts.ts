/** Global keyboard shortcuts registry, rendered by ShortcutsHelp and the
 *  Settings page reference section. */

type Shortcut = { keys: string[]; label: string };
type ShortcutGroup = { heading: string; items: Shortcut[] };

// The palette opens on Cmd+K or Ctrl+K (CommandPalette accepts either), so
// name the one this platform's users reach for.
const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent);

export const SHORTCUTS: ShortcutGroup[] = [
  {
    heading: "Global",
    items: [
      { keys: [IS_MAC ? "⌘" : "Ctrl", "K"], label: "Open command palette" },
      { keys: ["Space"], label: "Play / pause current audio" },
      { keys: ["Ctrl", "Space"], label: "Play / pause without leaving the text field" },
      { keys: ["Shift", "Space"], label: "Play / pause (alternate; not French AZERTY)" },
      { keys: ["Esc"], label: "Pause audio and exit the text field" },
      { keys: ["Shift", "?"], label: "Show this shortcuts dialog" },
    ],
  },
  {
    heading: "Navigation",
    items: [
      { keys: ["Esc"], label: "Close dialogs / clear focus" },
    ],
  },
];
