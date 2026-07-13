/** Global keyboard shortcuts registry, rendered by ShortcutsHelp and the
 *  Settings page reference section. */

type Shortcut = { keys: string[]; label: string };
type ShortcutGroup = { heading: string; items: Shortcut[] };

export const SHORTCUTS: ShortcutGroup[] = [
  {
    heading: "Global",
    items: [
      { keys: ["⌘", "K"], label: "Open command palette" },
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
