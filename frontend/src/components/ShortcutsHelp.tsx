/** Global keyboard shortcuts reference — opened with Shift+? or from command palette. */

import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

export type Shortcut = { keys: string[]; label: string };
export type ShortcutGroup = { heading: string; items: Shortcut[] };

export const SHORTCUTS: ShortcutGroup[] = [
  {
    heading: "Global",
    items: [
      { keys: ["⌘", "K"], label: "Open command palette" },
      { keys: ["Space"], label: "Play / pause current audio" },
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

export default function ShortcutsHelp() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) return;
      if (e.key === "?" && e.shiftKey) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard shortcuts</DialogTitle>
          <DialogDescription>Press <Kbd>Shift</Kbd> <Kbd>?</Kbd> anytime to open this.</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          {SHORTCUTS.map((group) => (
            <div key={group.heading}>
              <p className="text-xs font-medium text-muted-foreground mb-1.5">{group.heading}</p>
              <ul className="space-y-1">
                {group.items.map((sc) => (
                  <li key={sc.label} className="flex items-center justify-between text-sm py-1">
                    <span className="text-foreground">{sc.label}</span>
                    <span className="flex gap-1">
                      {sc.keys.map((k) => <Kbd key={k}>{k}</Kbd>)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[1.5rem] h-6 px-1.5 text-xs font-mono text-foreground bg-muted border border-border rounded">
      {children}
    </kbd>
  );
}
