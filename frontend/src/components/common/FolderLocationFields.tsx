import { useState } from "react";
import { Button } from "@/components/ui/button";
import { splitPath } from "@/lib/utils";
import FolderPicker from "./FolderPicker";

interface FolderLocationFieldsProps {
  folderName: string;
  onFolderNameChange: (name: string) => void;
  parentPath: string;
  onParentPathChange: (path: string) => void;
  /** Placeholder for the folder name input. */
  placeholder?: string;
  /** Auto-focus the folder name input. */
  autoFocus?: boolean;
}

export default function FolderLocationFields({
  folderName,
  onFolderNameChange,
  parentPath,
  onParentPathChange,
  placeholder,
  autoFocus,
}: FolderLocationFieldsProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const { sep } = splitPath(parentPath || "/");
  const fullPath = `${parentPath.replace(/[\\/]+$/, "")}${sep}${folderName}`;

  return (
    <>
      <div>
        <label htmlFor="folder-name-input" className="text-xs text-muted-foreground block mb-1">Folder name</label>
        <input
          id="folder-name-input"
          value={folderName}
          onChange={(e) => onFolderNameChange(e.target.value)}
          placeholder={placeholder}
          className="input w-full"
          autoFocus={autoFocus}
        />
      </div>
      <div>
        <label htmlFor="save-location-input" className="text-xs text-muted-foreground block mb-1">Save location</label>
        <div className="flex gap-2">
          <input
            id="save-location-input"
            value={parentPath}
            onChange={(e) => onParentPathChange(e.target.value)}
            className="input flex-1 text-xs"
          />
          <Button onClick={() => setPickerOpen(true)} variant="outline" size="sm">Browse…</Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground font-mono truncate" title={fullPath}>
        {fullPath}
      </p>

      <FolderPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={(p) => { onParentPathChange(p); setPickerOpen(false); }}
        initialPath={parentPath}
      />
    </>
  );
}
