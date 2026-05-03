import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useMutation } from "@tanstack/react-query";
import { useAudioStore } from "@/stores";
import type { AudioSegment } from "@/stores";
import { audioFileUrl } from "@/api/client";
import { getBestSegments, toAudioSegments } from "@/api/segments";
import { Button } from "@/components/ui/button";
import { Play, Pause, X, Volume2, VolumeX, MessageSquareText, Loader2 } from "lucide-react";
import { formatTime } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";

function findActiveSegment(segments: AudioSegment[] | null, time: number): AudioSegment | null {
  if (!segments || segments.length === 0) return null;
  // Binary search for the segment containing `time`
  let lo = 0, hi = segments.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const seg = segments[mid];
    if (time < seg.start) hi = mid - 1;
    else if (time > seg.end) lo = mid + 1;
    else return seg;
  }
  return null;
}

export default function AudioBar() {
  // Narrow selectors so AudioBar only re-renders when a consumed field changes.
  // Using the destructured `useAudioStore()` shape re-rendered on every audio
  // tick (currentTime, isPlaying) because the returned object ref changes.
  const audioPath = useAudioStore((s) => s.audioPath);
  const audioTitle = useAudioStore((s) => s.audioTitle);
  const audioArtwork = useAudioStore((s) => s.audioArtwork);
  const audioShowName = useAudioStore((s) => s.audioShowName);
  const audioFolder = useAudioStore((s) => s.audioFolder);
  const audioStem = useAudioStore((s) => s.audioStem);
  const audioSegments = useAudioStore((s) => s.audioSegments);
  const setAudioSegments = useAudioStore((s) => s.setAudioSegments);
  const pendingSeek = useAudioStore((s) => s.pendingSeek);
  const consumeSeek = useAudioStore((s) => s.consumeSeek);
  const stopAudio = useAudioStore((s) => s.stopAudio);

  const loadSegmentsMutation = useMutation({
    mutationFn: getBestSegments,
    onSuccess: (data, path) => {
      setAudioSegments(path, toAudioSegments(data));
      setShowSegment(true);
    },
    onError: (err) => {
      console.error("Failed to load transcript segments", err);
    },
  });
  const navigate = useNavigate();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState<number>(() => {
    const saved = parseFloat(localStorage.getItem("audioVolume") ?? "");
    return Number.isFinite(saved) && saved >= 0 && saved <= 1 ? saved : 0.8;
  });
  const [muted, setMuted] = useState(false);

  useEffect(() => {
    const id = setTimeout(() => localStorage.setItem("audioVolume", String(volume)), 250);
    return () => clearTimeout(id);
  }, [volume]);
  const [speed, setSpeed] = useState(1);
  const [showSegment, setShowSegment] = useState(false);
  const [hoverTime, setHoverTime] = useState<{ time: number; pct: number } | null>(null);
  const activeSeg = useMemo(() => findActiveSegment(audioSegments, currentTime), [audioSegments, currentTime]);

  const segmentLoading = loadSegmentsMutation.isPending;
  const segmentButtonTitle = segmentLoading
    ? "Loading transcript…"
    : audioSegments
      ? (showSegment ? "Hide segment text" : "Show current segment text")
      : "Load transcript";

  // Save position to localStorage periodically while playing.
  // Direct-assign is the idiomatic "latest value" ref pattern; an effect here
  // would introduce a post-paint window where the ref holds stale values.
  const currentTimeRef = useRef(currentTime);
  const durationRef = useRef(duration);
  // eslint-disable-next-line react-hooks/refs
  currentTimeRef.current = currentTime;
  // eslint-disable-next-line react-hooks/refs
  durationRef.current = duration;

  useEffect(() => {
    if (!audioPath || !playing) return;
    const save = () => {
      if (currentTimeRef.current > 5 && durationRef.current > 0) {
        localStorage.setItem(`pos:${audioPath}`, JSON.stringify({ time: currentTimeRef.current, duration: durationRef.current }));
      }
    };
    const interval = setInterval(save, 5000);
    return () => { save(); clearInterval(interval); };
  }, [audioPath, playing]);

  // Reset when track changes — restore saved speed, default 1× if none.
  useEffect(() => {
    setCurrentTime(0);
    setDuration(0);
    setPlaying(false);
    let newSpeed = 1;
    if (audioPath) {
      const savedSpeed = localStorage.getItem(`speed:${audioPath}`);
      if (savedSpeed) {
        const s = parseFloat(savedSpeed);
        if (s >= 0.5 && s <= 3) newSpeed = s;
      }
    }
    setSpeed(newSpeed);
    if (audioRef.current) audioRef.current.playbackRate = newSpeed;
  }, [audioPath]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle pending seek (from segment play buttons)
  useEffect(() => {
    if (pendingSeek == null) return;
    const audio = audioRef.current;
    if (!audio) return;

    // pendingSeek === -1 means pause
    if (pendingSeek < 0) {
      audio.pause();
      consumeSeek();
      return;
    }

    // Seek slightly before the requested time. HTML5 audio seeking snaps to
    // the nearest frame (~26ms for MP3) and WhisperX segment.start can run a
    // bit late relative to the first phoneme — without this preroll, very
    // short segments miss their opening.
    const doSeek = () => {
      audio.currentTime = Math.max(0, pendingSeek - 0.15);
      audio.play().catch(() => { /* autoplay block or interrupted */ });
      consumeSeek();
    };

    if (audio.readyState >= 1) {
      doSeek();
    } else {
      audio.addEventListener("loadedmetadata", doSeek, { once: true });
      return () => audio.removeEventListener("loadedmetadata", doSeek);
    }
  }, [pendingSeek, consumeSeek]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = muted ? 0 : volume;
    }
  }, [volume, muted]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = speed;
    if (audioPath) localStorage.setItem(`speed:${audioPath}`, String(speed));
  }, [speed, audioPath]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => { /* autoplay block or interrupted by pause */ });
    } else {
      audio.pause();
    }
  };

  const skip = (delta: number) => {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(audio.duration)) return;
    audio.currentTime = Math.max(0, Math.min(audio.duration, audio.currentTime + delta));
  };

  if (!audioPath) return null;

  // Right time always shows `−H:MM:SS` (or `−MM:SS`); the Unicode minus is
  // wider than a hyphen, so size for the worst case at each duration tier.
  const timeColW = duration >= 36000 ? "w-[4.75rem]"
    : duration >= 3600 ? "w-[4.25rem]"
    : "w-14";

  return (
    <div className="border-t border-border bg-card">
      {/* Current segment text — collapsible */}
      {showSegment && activeSeg && (
        <div className="px-4 py-2 border-b border-border/50 text-sm">
          <span className="text-xs mr-2 font-medium" style={{ color: speakerColor(activeSeg.speaker) }}>{activeSeg.speaker}</span>
          <span className="text-foreground/80">{activeSeg.text}</span>
        </div>
      )}

    <div className="px-4 py-2">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={audioFileUrl(audioPath)}
        onPlay={() => { setPlaying(true); useAudioStore.setState({ isPlaying: true }); }}
        onPause={() => { setPlaying(false); useAudioStore.setState({ isPlaying: false }); }}
        onEnded={() => { setPlaying(false); useAudioStore.setState({ isPlaying: false }); }}
        onLoadedMetadata={(e) => {
          setDuration(e.currentTarget.duration);
          e.currentTarget.playbackRate = speed;
          if (pendingSeek == null && audioPath) {
            try {
              const saved = JSON.parse(localStorage.getItem(`pos:${audioPath}`) || "null");
              if (saved?.time > 5 && saved.time < e.currentTarget.duration - 1) {
                e.currentTarget.currentTime = saved.time;
              }
            } catch { /* ignore */ }
          }
        }}
        onTimeUpdate={(e) => {
          const t = e.currentTarget.duration > 0 ? e.currentTarget.currentTime : 0;
          setCurrentTime(t);
          useAudioStore.setState({ currentTime: t });
        }}
      />

      {/* Row 1: Artwork + info | centered transport | right controls */}
      <div className="flex items-center gap-3 mb-1">
        {/* Artwork — click to open episode */}
        <button
          onClick={() => {
            if (audioFolder && audioStem) {
              navigate({ to: "/show/$folder/episode/$stem", params: { folder: encodeURIComponent(audioFolder), stem: encodeURIComponent(audioStem) } });
            } else if (audioPath) {
              navigate({ to: "/file/$path", params: { path: encodeURIComponent(audioPath) } });
            }
          }}
          className="w-10 h-10 rounded-md bg-muted shrink-0 overflow-hidden flex items-center justify-center hover:ring-2 hover:ring-primary/50 transition cursor-pointer"
          title="Go to episode"
        >
          {audioArtwork ? (
            <img src={audioArtwork} alt={audioTitle || "Now playing"} className="w-full h-full object-cover" />
          ) : (
            <Play className="w-4 h-4 text-muted-foreground" />
          )}
        </button>

        {/* Title above, show name below — vertical stack matches artwork height */}
        <div className="flex-1 min-w-0 flex flex-col justify-center leading-tight">
          <span className="text-sm font-medium truncate" title={audioTitle || "Playing"}>
            {audioTitle || "Playing"}
          </span>
          {audioShowName && (
            <span className="text-xs text-muted-foreground truncate">
              {audioShowName}
            </span>
          )}
        </div>

        {/* Transport — sits at the geometric center thanks to the equal-flex
            wrappers on the left (title) and right (controls). */}
        <div className="shrink-0 flex items-center justify-center gap-0.5">
          <SkipLabelButton label="−15" onClick={() => skip(-15)} title="Back 15s" />
          <SkipLabelButton label="−5" onClick={() => skip(-5)} title="Back 5s" />
          <SkipLabelButton label="−1" onClick={() => skip(-1)} title="Back 1s" />
          <Button onClick={togglePlay} variant="ghost" size="icon" className="h-7 w-7 mx-0.5" aria-label={playing ? "Pause" : "Play"}>
            {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </Button>
          <SkipLabelButton label="+1" onClick={() => skip(1)} title="Forward 1s" />
          <SkipLabelButton label="+5" onClick={() => skip(5)} title="Forward 5s" />
          <SkipLabelButton label="+15" onClick={() => skip(15)} title="Forward 15s" />
        </div>

        <div className="flex-1 flex items-center justify-end gap-3 min-w-0">
        {/* Speed — grouped in a pill */}
        <div className="flex items-center shrink-0 bg-muted/40 rounded-full h-7">
          <button
            onClick={() => setSpeed((s) => Math.max(0.5, +(s - 0.25).toFixed(2)))}
            className="text-muted-foreground hover:text-foreground text-sm w-6 h-7 flex items-center justify-center rounded-l-full hover:bg-accent transition"
            title="Slower"
            aria-label="Slower"
          >
            −
          </button>
          <button
            onClick={() => setSpeed(1)}
            className="text-[11px] font-medium font-mono w-10 text-center text-foreground/80 hover:text-foreground h-7 transition"
            title="Reset speed"
            aria-label="Reset speed"
          >
            {speed.toFixed(2)}×
          </button>
          <button
            onClick={() => setSpeed((s) => Math.min(3, +(s + 0.25).toFixed(2)))}
            className="text-muted-foreground hover:text-foreground text-sm w-6 h-7 flex items-center justify-center rounded-r-full hover:bg-accent transition"
            title="Faster"
            aria-label="Faster"
          >
            +
          </button>
        </div>

        {/* Volume — mute toggle; hover reveals a vertical slider above. */}
        <div className="group/vol relative shrink-0">
          <button
            onClick={() => {
              if (muted) {
                setMuted(false);
                if (volume === 0) setVolume(0.2);
              } else {
                setMuted(true);
              }
            }}
            className="h-7 w-7 flex items-center justify-center rounded-full bg-muted/40 text-muted-foreground hover:text-foreground hover:bg-muted/60 transition"
            title={muted ? "Unmute" : "Mute"}
            aria-label={muted ? "Unmute" : "Mute"}
          >
            {muted || volume === 0 ? (
              <VolumeX className="w-3.5 h-3.5" />
            ) : (
              <Volume2 className="w-3.5 h-3.5" />
            )}
          </button>
          <div
            className="absolute bottom-full left-1/2 -translate-x-1/2 pb-1 opacity-0 pointer-events-none group-hover/vol:opacity-100 group-hover/vol:pointer-events-auto transition-opacity"
          >
            <div className="px-2 py-3 rounded-md bg-popover border border-border shadow-md">
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={muted ? 0 : volume}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setVolume(v);
                  if (v > 0 && muted) setMuted(false);
                  if (v === 0 && !muted) setMuted(true);
                }}
                aria-label="Volume"
                className="accent-primary cursor-pointer h-24 w-1.5 [writing-mode:vertical-lr] [direction:rtl] [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3"
              />
            </div>
          </div>
        </div>

        {/* Lazy-loads transcript on first click; won't re-fetch once in store */}
        <button
          onClick={() => {
            if (audioSegments) {
              setShowSegment(!showSegment);
            } else if (audioPath && !segmentLoading) {
              loadSegmentsMutation.mutate(audioPath);
            }
          }}
          disabled={segmentLoading}
          className={`shrink-0 h-7 w-7 flex items-center justify-center rounded-full transition ${
            showSegment && audioSegments
              ? "bg-primary/15 text-primary hover:bg-primary/20"
              : "bg-muted/40 text-muted-foreground hover:text-foreground hover:bg-muted/60"
          } disabled:opacity-60`}
          title={segmentButtonTitle}
          aria-label="Segment text"
        >
          {segmentLoading ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <MessageSquareText className="w-3.5 h-3.5" />
          )}
        </button>

        {/* Close */}
        <Button onClick={stopAudio} variant="ghost" size="icon" className="h-7 w-7 shrink-0 text-muted-foreground hover:text-foreground" aria-label="Close player">
          <X className="w-3.5 h-3.5" />
        </Button>
        </div>
      </div>

      {/* Row 2: Seek bar + time */}
      <div className="flex items-center gap-2">
        <span className={`text-[11px] font-light text-muted-foreground ${timeColW} text-right shrink-0 font-mono tabular-nums`}>
          {formatTime(currentTime, false)}
        </span>
        <div
          className="flex-1 h-4 flex items-center cursor-pointer relative group"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            if (audioRef.current && duration > 0) {
              audioRef.current.currentTime = frac * duration;
            }
          }}
          onMouseMove={(e) => {
            if (duration <= 0) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            setHoverTime({ time: pct * duration, pct: pct * 100 });
          }}
          onMouseLeave={() => setHoverTime(null)}
        >
          <div className="w-full h-1.5 bg-muted rounded-full relative">
            <div
              className="absolute inset-y-0 left-0 bg-primary rounded-full transition-all"
              style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%" }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-primary rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
              style={{ left: duration > 0 ? `calc(${(currentTime / duration) * 100}% - 6px)` : "0" }}
            />
          </div>
          {hoverTime && (
            <div
              className="absolute bottom-full mb-1.5 -translate-x-1/2 bg-popover text-popover-foreground text-2xs font-mono px-1.5 py-0.5 rounded border border-border shadow-sm pointer-events-none"
              style={{ left: `${Math.max(3, Math.min(97, hoverTime.pct))}%` }}
            >
              {formatTime(hoverTime.time, false)}
            </div>
          )}
        </div>
        <span className={`text-[11px] font-light text-muted-foreground ${timeColW} shrink-0 font-mono tabular-nums text-left`}>
          {duration > 0 ? `−${formatTime(Math.max(0, duration - currentTime), false)}` : "−0:00"}
        </span>
      </div>
    </div>
    </div>
  );
}

// Text-only skip button — clean tabular label, reads instantly, no fiddly icon+number overlay.
function SkipLabelButton({
  label,
  onClick,
  title,
}: {
  label: string;
  onClick: () => void;
  title: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="h-7 px-1.5 flex items-center justify-center rounded-md text-[11px] font-medium font-mono text-muted-foreground hover:text-foreground hover:bg-accent transition"
    >
      {label}
    </button>
  );
}
