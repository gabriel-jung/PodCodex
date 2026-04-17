import { Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type {
  McpPrompt,
  McpPromptCreate,
  McpPromptUpdate,
  SlotDef,
  SlotType,
} from "@/api/mcpPrompts";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const SLUG_RE = /^[a-z][a-z0-9_-]{1,39}$/;
const SLOT_RE = /\{(\w+)\}/g;

const SLOT_TYPES: SlotType[] = ["string", "enum", "int", "bool"];


interface Props {
  open: boolean;
  editing: McpPrompt | null;
  onClose: () => void;
  onSubmit: (payload: McpPromptCreate | McpPromptUpdate) => Promise<void>;
  existingIds: string[];
}

export default function PromptEditorModal({
  open,
  editing,
  onClose,
  onSubmit,
  existingIds,
}: Props) {
  const isEdit = editing !== null;
  const isBuiltin = editing?.is_builtin ?? false;

  const [id, setId] = useState("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [template, setTemplate] = useState("");
  const [slots, setSlots] = useState<SlotDef[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setId(editing?.id ?? "");
    setTitle(editing?.title ?? "");
    setDescription(editing?.description ?? "");
    setTemplate(editing?.template ?? "");
    setSlots(editing?.slots ?? []);
    setError(null);
  }, [open, editing]);

  const usedSlotNames = useMemo(() => {
    const names = new Set<string>();
    for (const m of template.matchAll(SLOT_RE)) names.add(m[1]);
    return names;
  }, [template]);

  const declaredNames = useMemo(
    () => new Set(slots.map((s) => s.name)),
    [slots],
  );

  const undeclared = useMemo(
    () => [...usedSlotNames].filter((n) => !declaredNames.has(n)),
    [usedSlotNames, declaredNames],
  );

  const idError = (() => {
    if (isEdit) return null;
    if (!id) return "Required.";
    if (!SLUG_RE.test(id))
      return "Lowercase letters, digits, hyphens or underscores; start with a letter; 2-40 chars.";
    if (existingIds.includes(id)) return "An entry with this id already exists.";
    return null;
  })();

  const canSubmit =
    !submitting &&
    title.trim() !== "" &&
    template.trim() !== "" &&
    undeclared.length === 0 &&
    (isEdit || idError === null);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      if (isEdit) {
        const payload: McpPromptUpdate = isBuiltin
          ? { template, slots }
          : { title, description, template, slots };
        await onSubmit(payload);
      } else {
        const payload: McpPromptCreate = {
          id,
          name: id,
          title,
          description,
          template,
          slots,
        };
        await onSubmit(payload);
      }
      onClose();
    } catch (exc: unknown) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setSubmitting(false);
    }
  }

  function addSlot() {
    const base = "slot";
    let i = slots.length + 1;
    let name = `${base}${i}`;
    while (declaredNames.has(name)) {
      i += 1;
      name = `${base}${i}`;
    }
    setSlots([...slots, { name, type: "string", required: true }]);
  }

  function updateSlot(idx: number, patch: Partial<SlotDef>) {
    setSlots(slots.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  }

  function removeSlot(idx: number) {
    setSlots(slots.filter((_, i) => i !== idx));
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{isEdit ? `Edit ${editing?.id}` : "New prompt"}</DialogTitle>
          <DialogDescription>
            {isBuiltin
              ? "Built-in prompt. Template and slots are editable; name stays fixed."
              : "Claude Desktop will show this in its slash menu once you restart it."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {!isEdit && (
            <Field label="id" error={idError}>
              <input
                className="w-full px-2 py-1.5 rounded border border-border bg-background text-sm font-mono"
                value={id}
                onChange={(e) => setId(e.target.value)}
                placeholder="my_prompt"
                autoFocus
              />
            </Field>
          )}

          <Field label="Title" error={title.trim() === "" ? "Required." : null}>
            <input
              className="w-full px-2 py-1.5 rounded border border-border bg-background text-sm"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={isBuiltin}
              placeholder="Podcast brief on a topic"
            />
          </Field>

          <Field label="Description">
            <textarea
              className="w-full px-2 py-1.5 rounded border border-border bg-background text-sm"
              rows={2}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={isBuiltin}
              placeholder="Short hint shown in Claude's slash menu."
            />
          </Field>

          <Field
            label="Template"
            help="Use {slot_name} for fields the user fills. Undeclared slots are rejected."
            error={undeclared.length ? `Undeclared slots: ${undeclared.join(", ")}` : null}
          >
            <textarea
              className="w-full px-2 py-1.5 rounded border border-border bg-background text-sm font-mono"
              rows={6}
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              placeholder="Using podcodex, compile a brief on {topic}. Cite inline as [Show • Episode @ MM:SS]."
            />
          </Field>

          <Field label="Slots" help="Fields the user fills in Claude's slash menu.">
            <div className="space-y-2">
              {slots.length === 0 && (
                <p className="text-xs text-muted-foreground">No slots yet.</p>
              )}
              {slots.map((slot, i) => (
                <div
                  key={i}
                  className="grid grid-cols-[1fr_auto_auto_auto] gap-2 items-center"
                >
                  <input
                    className="px-2 py-1 rounded border border-border bg-background text-xs font-mono"
                    value={slot.name}
                    onChange={(e) => updateSlot(i, { name: e.target.value })}
                    placeholder="topic"
                  />
                  <select
                    className="px-2 py-1 rounded border border-border bg-background text-xs"
                    value={slot.type ?? "string"}
                    onChange={(e) =>
                      updateSlot(i, { type: e.target.value as SlotType })
                    }
                  >
                    {SLOT_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                  <label className="text-xs flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={slot.required ?? true}
                      onChange={(e) =>
                        updateSlot(i, { required: e.target.checked })
                      }
                    />
                    required
                  </label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                    onClick={() => removeSlot(i)}
                    title="Remove slot"
                  >
                    <Trash2 size={14} />
                  </Button>
                </div>
              ))}
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addSlot}
                className="h-7 text-xs gap-1"
              >
                <Plus size={12} /> Add slot
              </Button>
            </div>
          </Field>
        </div>

        {error && (
          <p className="text-xs text-destructive border border-destructive/40 rounded bg-destructive/10 px-2 py-1.5">
            {error}
          </p>
        )}

        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!canSubmit}>
            {isEdit ? "Save" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}


function Field({
  label,
  help,
  error,
  children,
}: {
  label: string;
  help?: string;
  error?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-medium">{label}</span>
        {help && <span className="text-xs text-muted-foreground">{help}</span>}
      </div>
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
