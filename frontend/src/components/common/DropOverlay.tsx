/** Full-screen overlay shown when files are being dragged over the window. */

import { Upload } from "lucide-react";

interface DropOverlayProps {
  message: string;
}

export default function DropOverlay({ message }: DropOverlayProps) {
  return (
    <div className="fixed inset-0 z-50 bg-primary/10 border-2 border-dashed border-primary flex items-center justify-center pointer-events-none">
      <div className="bg-background border border-border rounded-lg px-6 py-4 shadow-lg flex items-center gap-3">
        <Upload className="h-5 w-5 text-primary" />
        <p className="text-sm font-medium">{message}</p>
      </div>
    </div>
  );
}
