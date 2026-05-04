/** First-launch onboarding: dismissible walkthrough of sources, pipeline, and search. */

import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  FileAudio,
  Languages,
  Mic,
  Plug,
  Plus,
  Radio,
  Search,
  Sparkles,
  Volume2,
  Wand2,
  Video,
} from "lucide-react";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { VisuallyHidden } from "radix-ui";
import { Button } from "@/components/ui/button";
import { useOnboardingStore } from "@/stores";

interface OnboardingModalProps {
  onAddShow: () => void;
}

export default function OnboardingModal({ onAddShow }: OnboardingModalProps) {
  const seen = useOnboardingStore((s) => s.seen);
  const setSeen = useOnboardingStore((s) => s.setSeen);
  const [step, setStep] = useState(0);

  if (seen) return null;

  const finish = () => setSeen(true);
  const handleAddShow = () => {
    setSeen(true);
    onAddShow();
  };

  const stepTitle = step === 0 ? "Welcome to PodCodex"
    : step === 1 ? "Three ways to add a show"
    : "From audio to answers";

  return (
    <Dialog open onOpenChange={(o) => !o && finish()}>
      <DialogContent className="sm:max-w-lg h-[440px] flex flex-col">
        <VisuallyHidden.Root>
          <DialogTitle>{stepTitle}</DialogTitle>
          <DialogDescription>First-launch walkthrough</DialogDescription>
        </VisuallyHidden.Root>
        <div className="flex items-center justify-center gap-2 pt-2 pb-4">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={`h-1.5 rounded-full transition-all ${
                i === step ? "w-8 bg-primary" : i < step ? "w-6 bg-primary/60" : "w-6 bg-muted"
              }`}
            />
          ))}
        </div>

        <div className="flex-1 overflow-y-auto">
          {step === 0 && <StepWelcome />}
          {step === 1 && <StepSources />}
          {step === 2 && <StepPipeline />}
        </div>

        <div className="flex items-center justify-between pt-4 border-t border-border mt-2">
          <Button onClick={finish} variant="ghost" size="sm">Skip</Button>
          <div className="flex items-center gap-2">
            {step > 0 && (
              <Button onClick={() => setStep((s) => s - 1)} variant="outline" size="sm">
                <ArrowLeft className="w-3.5 h-3.5 mr-1" /> Back
              </Button>
            )}
            {step < 2 && (
              <Button onClick={() => setStep((s) => s + 1)} size="sm">
                Next <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Button>
            )}
            {step === 2 && (
              <Button onClick={handleAddShow} size="sm">
                <Plus className="w-3.5 h-3.5 mr-1" /> Add your first show
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function StepWelcome() {
  return (
    <div className="text-center space-y-3 py-2">
      <div className="w-14 h-14 mx-auto rounded-full bg-primary/10 flex items-center justify-center">
        <Sparkles className="w-7 h-7 text-primary" />
      </div>
      <h2 className="text-xl font-semibold">Welcome to PodCodex</h2>
      <p className="text-sm text-muted-foreground max-w-sm mx-auto">
        Turn any audio source into a searchable knowledge base, fully on your machine.
      </p>
      <ul className="text-sm text-muted-foreground space-y-2 pt-2 text-left max-w-sm mx-auto">
        {[
          "Transcribe with WhisperX: accurate, diarized, timestamped",
          "Correct in the editor or with any local or cloud LLM",
          "Search across your library, semantic and exact",
          "Plug into a Discord bot or MCP chat for conversational access",
        ].map((line) => (
          <li key={line} className="flex items-start gap-2">
            <Check className="w-4 h-4 text-success shrink-0 mt-0.5" />
            <span>{line}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function StepSources() {
  const sources = [
    { icon: Radio, label: "RSS feed", detail: "Any podcast feed URL" },
    { icon: Video, label: "YouTube", detail: "Channel or playlist, with subtitles" },
    { icon: FileAudio, label: "Local folder", detail: "Audio files you already have" },
  ];
  return (
    <div className="space-y-4">
      <div className="text-center space-y-1.5">
        <h2 className="text-xl font-semibold">Three ways to add a show</h2>
        <p className="text-sm text-muted-foreground">Mix and match. Every show lives in its own folder.</p>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {sources.map(({ icon: Icon, label, detail }) => (
          <div key={label} className="border border-border rounded-lg p-3 text-center space-y-1.5">
            <div className="w-9 h-9 mx-auto rounded-lg bg-muted flex items-center justify-center">
              <Icon className="w-4 h-4 text-muted-foreground" />
            </div>
            <p className="text-sm font-medium">{label}</p>
            <p className="text-2xs text-muted-foreground leading-snug">{detail}</p>
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground text-center pt-1">
        Drop audio files on the window for one-off transcription, or import existing subtitles (.srt, .vtt) to skip it entirely.
      </p>
    </div>
  );
}

function StepPipeline() {
  const steps = [
    { icon: Mic, label: "Transcribe", detail: "WhisperX with optional diarization" },
    { icon: Wand2, label: "Correct", detail: "Hand-edit in the editor, or run an LLM cleanup pass" },
    { icon: Search, label: "Index & search", detail: "Semantic and exact, across every episode" },
  ];
  const sideSteps = [
    { icon: Languages, label: "Translate to any language" },
    { icon: Volume2, label: "Dub with voice cloning" },
    { icon: Plug, label: "Share via Discord or MCP" },
  ];
  return (
    <div className="space-y-3">
      <div className="text-center space-y-1.5">
        <h2 className="text-xl font-semibold">From audio to answers</h2>
        <p className="text-sm text-muted-foreground">The goal is high-quality transcripts. Run the pipeline one episode at a time, or batch a whole feed.</p>
      </div>
      <ol className="space-y-2">
        {steps.map(({ icon: Icon, label, detail }, i) => (
          <li key={label} className="flex items-center gap-3 p-2.5 rounded-md bg-muted/40">
            <span className="w-6 h-6 rounded-full bg-background border border-border flex items-center justify-center text-xs font-medium shrink-0">
              {i + 1}
            </span>
            <Icon className="w-4 h-4 text-muted-foreground shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">{label}</p>
              <p className="text-xs text-muted-foreground">{detail}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="pt-1">
        <p className="text-2xs text-muted-foreground mb-1.5">Optional side branches</p>
        <ul className="flex flex-wrap gap-x-3 gap-y-1">
          {sideSteps.map(({ icon: Icon, label }) => (
            <li key={label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Icon className="w-3 h-3 shrink-0" />
              <span>{label}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
