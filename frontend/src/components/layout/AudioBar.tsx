import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useAudioStore } from "@/stores";
import type { AudioSegment } from "@/stores";
import { audioFileUrl } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Play, Pause, X, Volume2, VolumeX, MessageSquareText } from "lucide-react";
import { formatTime } from "@/lib/utils";

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
  const { audioPath, audioTitle, audioArtwork, audioShowName, audioFolder, audioStem, audioSegments, pendingSeek, consumeSeek, stopAudio } = useAudioStore();
  const navigate = useNavigate();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [muted, setMuted] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showSegment, setShowSegment] = useState(false);
  const [hoverTime, setHoverTime] = useState<{ time: number; pct: number } | null>(null);
  const [timeMode, setTimeMode] = useState<"remaining" | "elapsed" | "total">("remaining");
  const activeSeg = useMemo(() => findActiveSegment(audioSegments, currentTime), [audioSegments, currentTime]);

  // Save position to localStorage on pause/time update
  useEffect(() => {
    if (!audioPath || !playing) return;
    const save = () => {
      if (currentTime > 5 && duration > 0) {
        localStorage.setItem(`pos:${audioPath}`, JSON.stringify({ time: currentTime, duration }));
      }
    };
    const interval = setInterval(save, 5000);
    return () => { save(); clearInterval(interval); };
  }, [audioPath, playing, currentTime, duration]);

  // Reset when track changes — restore saved speed
  useEffect(() => {
    setCurrentTime(0);
    setDuration(0);
    setPlaying(false);
    if (audioPath) {
      const savedSpeed = localStorage.getItem(`speed:${audioPath}`);
      if (savedSpeed) {
        const s = parseFloat(savedSpeed);
        if (s >= 0.5 && s <= 3) setSpeed(s);
      }
    }
    if (audioRef.current) audioRef.current.playbackRate = speed;
  }, [audioPath]);

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
      audio.play();
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
      audio.play();
    } else {
      audio.pause();
    }
  };

  const skip = (delta: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(audio.duration, audio.currentTime + delta));
  };

  if (!audioPath) return null;

  // Time-label width adapts to the longest possible display for the current
  // duration: `-MM:SS` fits in w-11, `-H:MM:SS` needs w-14, and anything from
  // 10h upward (including day-long audiobooks) needs w-16.
  const timeColW = duration >= 36000 ? "w-16"
    : duration >= 3600 ? "w-14"
    : "w-11";

  return (
    <div className="border-t border-border bg-card">
      {/* Current segment text — collapsible */}
      {showSegment && activeSeg && (
        <div className="px-4 py-2 border-b border-border/50 text-sm">
          <span className="text-xs text-muted-foreground mr-2">{activeSeg.speaker}</span>
          <span className="text-foreground/80">{activeSeg.text}</span>
        </div>
      )}

    <div className="px-4 pt-3 pb-1.5">
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
              if (saved?.time > 5 && saved.time < e.currentTarget.duration - 5) {
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
      <div className="flex items-center gap-4 mb-1.5">
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

        {/* Title + show name */}
        <div className="flex-1 min-w-0 max-w-sm">
          <p className="text-sm font-medium truncate" title={audioTitle || "Playing"}>{audioTitle || "Playing"}</p>
          {audioShowName && (
            <p className="text-xs text-muted-foreground truncate">{audioShowName}</p>
          )}
        </div>

        {/* Centered transport */}
        <div className="flex-1 flex items-center justify-center gap-0.5">
          <SkipLabelButton label="−15s" onClick={() => skip(-15)} title="Back 15s" />
          <SkipLabelButton label="−5s" onClick={() => skip(-5)} title="Back 5s" />
          <Button onClick={togglePlay} variant="ghost" size="icon" className="h-9 w-9 mx-1">
            {playing ? <Pause className="w-4.5 h-4.5" /> : <Play className="w-4.5 h-4.5 ml-0.5" />}
          </Button>
          <SkipLabelButton label="+5s" onClick={() => skip(5)} title="Forward 5s" />
          <SkipLabelButton label="+15s" onClick={() => skip(15)} title="Forward 15s" />
        </div>

        {/* Speed — grouped in a pill */}
        <div className="flex items-center shrink-0 bg-muted/40 rounded-full h-7">
          <button
            onClick={() => setSpeed((s) => Math.max(0.5, +(s - 0.25).toFixed(2)))}
            className="text-muted-foreground hover:text-foreground text-sm w-6 h-7 flex items-center justify-center rounded-l-full hover:bg-accent transition"
            title="Slower"
          >
            −
          </button>
          <button
            onClick={() => setSpeed(1)}
            className="text-[11px] font-medium tabular-nums w-10 text-center text-foreground/80 hover:text-foreground h-7 transition"
            title="Reset speed"
          >
            {speed.toFixed(2)}×
          </button>
          <button
            onClick={() => setSpeed((s) => Math.min(3, +(s + 0.25).toFixed(2)))}
            className="text-muted-foreground hover:text-foreground text-sm w-6 h-7 flex items-center justify-center rounded-r-full hover:bg-accent transition"
            title="Faster"
          >
            +
          </button>
        </div>

        {/* Volume — icon + slider in a pill */}
        <div className="flex items-center gap-1.5 shrink-0 bg-muted/40 rounded-full h-7 pl-1 pr-2.5">
          <button
            onClick={() => setMuted(!muted)}
            className="text-muted-foreground hover:text-foreground h-7 w-6 flex items-center justify-center transition"
            title={muted ? "Unmute" : "Mute"}
          >
            {muted || volume === 0 ? (
              <VolumeX className="w-3.5 h-3.5" />
            ) : (
              <Volume2 className="w-3.5 h-3.5" />
            )}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={muted ? 0 : volume}
            onChange={(e) => {
              setVolume(Number(e.target.value));
              if (muted) setMuted(false);
            }}
            className="w-16 accent-primary h-1"
          />
        </div>

        {/* Segment text toggle — matches pill styling so it reads as a real control */}
        {audioSegments && (
          <button
            onClick={() => setShowSegment(!showSegment)}
            className={`shrink-0 h-7 w-7 flex items-center justify-center rounded-full transition ${
              showSegment
                ? "bg-primary/15 text-primary hover:bg-primary/20"
                : "bg-muted/40 text-muted-foreground hover:text-foreground hover:bg-muted/60"
            }`}
            title={showSegment ? "Hide segment text" : "Show current segment text"}
          >
            <MessageSquareText className="w-3.5 h-3.5" />
          </button>
        )}

        {/* Close */}
        <Button onClick={stopAudio} variant="ghost" size="icon" className="h-7 w-7 shrink-0 text-muted-foreground hover:text-foreground">
          <X className="w-3.5 h-3.5" />
        </Button>
      </div>

      {/* Row 2: Seek bar + time */}
      <div className="flex items-center gap-2">
        <span className={`text-2xs text-muted-foreground ${timeColW} text-right shrink-0 tabular-nums`}>
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
              className="absolute bottom-full mb-1.5 -translate-x-1/2 bg-popover text-popover-foreground text-2xs tabular-nums px-1.5 py-0.5 rounded border border-border shadow-sm pointer-events-none"
              style={{ left: `${hoverTime.pct}%` }}
            >
              {formatTime(hoverTime.time, false)}
            </div>
          )}
        </div>
        <button
          onClick={() => setTimeMode((m) => m === "remaining" ? "elapsed" : m === "elapsed" ? "total" : "remaining")}
          className={`text-2xs text-muted-foreground ${timeColW} shrink-0 tabular-nums text-left hover:text-foreground transition`}
          title="Click to toggle time display"
        >
          {timeMode === "remaining" && duration > 0
            ? `-${formatTime(duration - currentTime, false)}`
            : timeMode === "elapsed"
              ? formatTime(currentTime, false)
              : formatTime(duration, false)}
        </button>
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
      className="h-7 px-2 flex items-center justify-center rounded-md text-[11px] font-medium tabular-nums text-muted-foreground hover:text-foreground hover:bg-accent transition"
    >
      {label}
    </button>
  );
}
