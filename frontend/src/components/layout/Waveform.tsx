import { useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";

interface WaveformProps {
  /** Existing <audio> element to attach to (no double-loading). */
  audioRef: React.RefObject<HTMLAudioElement | null>;
  /** 0-1 progress fraction — for external seeking. */
  seekFraction?: number;
  onSeekConsumed?: () => void;
}

function getCSSVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function resolveColor(cssVar: string): string {
  const raw = getCSSVar(cssVar);
  if (!raw) return "#666";
  // oklch values need wrapping
  if (raw.startsWith("oklch")) return raw;
  return `oklch(${raw})`;
}

export default function Waveform({ audioRef, seekFraction, onSeekConsumed }: WaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  // Create WaveSurfer once, attach to existing <audio> element
  useEffect(() => {
    if (!containerRef.current || !audioRef.current) return;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      media: audioRef.current,
      height: 48,
      barWidth: 2,
      barRadius: 2,
      barGap: 1,
      normalize: true,
      dragToSeek: true,
      cursorWidth: 2,
      waveColor: resolveColor("--muted"),
      progressColor: resolveColor("--primary"),
      cursorColor: resolveColor("--primary"),
    });

    wsRef.current = ws;

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [audioRef.current?.src]);

  // Update colors when theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      if (wsRef.current) {
        wsRef.current.setOptions({
          waveColor: resolveColor("--muted"),
          progressColor: resolveColor("--primary"),
          cursorColor: resolveColor("--primary"),
        });
      }
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  // Handle external seek
  useEffect(() => {
    if (seekFraction != null && wsRef.current) {
      wsRef.current.seekTo(Math.max(0, Math.min(1, seekFraction)));
      onSeekConsumed?.();
    }
  }, [seekFraction, onSeekConsumed]);

  return (
    <div ref={containerRef} className="flex-1 min-w-0 h-12" />
  );
}
