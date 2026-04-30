import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, KeyRound, Loader2, RefreshCcw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import {
  deleteShowPassword,
  getShowAccess,
  setShowPassword,
} from "@/api/botAccess";
import type { ShowAccess, ShowPasswordSet } from "@/api/botAccess";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";


interface Props {
  show: string;
}

export default function ShowAccessSection({ show }: Props) {
  const qc = useQueryClient();
  const { data: access, isLoading } = useQuery<ShowAccess>({
    queryKey: queryKeys.showAccess(show),
    queryFn: () => getShowAccess(show),
    retry: false,
  });

  const [manualOpen, setManualOpen] = useState(false);
  const [reveal, setReveal] = useState<ShowPasswordSet | null>(null);
  const [error, setError] = useState<string | null>(null);

  function invalidate() {
    qc.invalidateQueries({ queryKey: queryKeys.showAccess(show) });
  }

  const generateMut = useMutation({
    mutationFn: () => setShowPassword(show),
    onSuccess: (r) => {
      invalidate();
      setReveal(r);
    },
    onError: (e: Error) => setError(e.message),
  });

  const removeMut = useMutation({
    mutationFn: () => deleteShowPassword(show),
    onSuccess: invalidate,
    onError: (e: Error) => setError(e.message),
  });

  const setManualMut = useMutation({
    mutationFn: (password: string) => setShowPassword(show, password),
    onSuccess: (r) => {
      invalidate();
      setReveal(r);
      setManualOpen(false);
    },
    onError: (e: Error) => setError(e.message),
  });

  const busy =
    isLoading ||
    generateMut.isPending ||
    removeMut.isPending ||
    setManualMut.isPending;

  return (
    <>
      <SettingSection
        title="Discord bot access"
        description="Restrict this show to Discord servers that know the password. Leave public for anyone who can reach your bot."
      >
        <SettingRow
          label="Access"
          help={
            access?.is_protected
              ? "Discord servers must run /unlock <password> before querying this show."
              : "Anyone with access to your bot can search this show."
          }
        >
          <AccessChip access={access} isLoading={isLoading} />
        </SettingRow>

        <div className="py-3 flex flex-wrap gap-2">
          <Button
            size="sm"
            variant={access?.is_protected ? "outline" : "default"}
            disabled={busy}
            onClick={() => generateMut.mutate()}
            className="gap-1.5"
          >
            {access?.is_protected ? (
              <RefreshCcw className="w-3.5 h-3.5" />
            ) : (
              <KeyRound className="w-3.5 h-3.5" />
            )}
            {access?.is_protected ? "Rotate password" : "Generate password"}
          </Button>

          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => {
              setError(null);
              setManualOpen(true);
            }}
          >
            Set manually
          </Button>

          {access?.is_protected && (
            <Button
              size="sm"
              variant="destructive"
              disabled={busy}
              onClick={() => {
                confirmDialog.open({
                  title: `Remove password protection for "${show}"?`,
                  description: "It will become publicly searchable via the bot.",
                  confirmLabel: "Remove",
                  variant: "destructive",
                  onConfirm: () => removeMut.mutate(),
                });
              }}
              className="gap-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Remove password
            </Button>
          )}
        </div>

        {error && <p className="text-xs text-destructive">{error}</p>}
      </SettingSection>

      <RevealDialog
        reveal={reveal}
        onClose={() => setReveal(null)}
      />

      <ManualDialog
        open={manualOpen}
        onClose={() => setManualOpen(false)}
        onSubmit={(pw) => setManualMut.mutate(pw)}
        submitting={setManualMut.isPending}
      />
    </>
  );
}


function AccessChip({
  access,
  isLoading,
}: {
  access: ShowAccess | undefined;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <Loader2 size={12} className="animate-spin" /> Loading
      </span>
    );
  }
  if (!access) return null;
  if (access.is_protected) {
    return (
      <span className="inline-flex items-center text-xs border rounded-full px-2 py-0.5 bg-success/15 text-success border-success/30">
        Password protected
      </span>
    );
  }
  return (
    <span className="inline-flex items-center text-xs border rounded-full px-2 py-0.5 bg-muted text-muted-foreground border-border">
      Public
    </span>
  );
}


function RevealDialog({
  reveal,
  onClose,
}: {
  reveal: ShowPasswordSet | null;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);

  useEffect(() => {
    if (reveal) {
      setCopied(false);
      setAcknowledged(false);
    }
  }, [reveal]);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);

  async function doCopy() {
    if (!reveal) return;
    try {
      await navigator.clipboard.writeText(reveal.password);
      setCopied(true);
    } catch {
      /* ignore */
    }
  }

  return (
    <Dialog
      open={reveal !== null}
      onOpenChange={(o) => {
        if (!o && acknowledged) onClose();
      }}
    >
      <DialogContent className="max-w-md" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>
            {reveal?.generated ? "New password generated" : "Password saved"}
          </DialogTitle>
          <DialogDescription>
            This is shown once. If you lose it, generate a new one — the
            previous password stops working immediately.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2">
          <code className="flex-1 text-sm font-mono px-3 py-2 rounded bg-muted break-all select-all">
            {reveal?.password}
          </code>
          <Button
            size="sm"
            variant="outline"
            onClick={doCopy}
            className="shrink-0 gap-1.5"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>

        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(e) => setAcknowledged(e.target.checked)}
          />
          I saved this password somewhere safe.
        </label>

        <DialogFooter>
          <Button onClick={onClose} disabled={!acknowledged}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}


function ManualDialog({
  open,
  onClose,
  onSubmit,
  submitting,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (password: string) => void;
  submitting: boolean;
}) {
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");

  useEffect(() => {
    if (open) {
      setPassword("");
      setConfirm("");
    }
  }, [open]);

  const tooShort = password.length > 0 && password.length < 16;
  const mismatch = confirm.length > 0 && password !== confirm;
  const canSubmit = password.length >= 16 && password === confirm && !submitting;

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Set password manually</DialogTitle>
          <DialogDescription>
            Minimum 16 characters — short passwords are trivially brute-forced.
            Prefer Generate password for a random one.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-xs font-medium">Password</label>
            <input
              type="password"
              className="w-full px-2 py-1.5 rounded border border-border bg-background text-sm font-mono"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoFocus
            />
            {tooShort && (
              <p className="text-xs text-destructive">Must be at least 16 characters.</p>
            )}
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium">Confirm</label>
            <input
              type="password"
              className="w-full px-2 py-1.5 rounded border border-border bg-background text-sm font-mono"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
            {mismatch && (
              <p className="text-xs text-destructive">Passwords don't match.</p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button onClick={() => onSubmit(password)} disabled={!canSubmit}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
