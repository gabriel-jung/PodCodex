import { useEffect, useRef, useState } from "react";
import { useAppStore } from "@/store";
import { audioFileUrl } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Play, Pause, X, SkipBack, SkipForward, Volume2, VolumeX } from "lucide-react";

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "--:--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function AudioBar() {
  const { audioPath, audioTitle, audioArtwork, audioShowName, stopAudio } = useAppStore();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [muted, setMuted] = useState(false);
  const [seeking, setSeeking] = useState(false);

  // Reset when track changes
  useEffect(() => {
    setCurrentTime(0);
    setDuration(0);
    setPlaying(false);
  }, [audioPath]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = muted ? 0 : volume;
    }
  }, [volume, muted]);

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

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = Number(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  if (!audioPath) {
    return (
      <div className="border-t border-border bg-card px-6 py-3 h-16 flex items-center">
        <span className="text-xs text-muted-foreground">No audio playing</span>
      </div>
    );
  }

  return (
    <div className="border-t border-border bg-card px-4 flex items-center gap-3 h-[72px]">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={audioFileUrl(audioPath)}
        autoPlay
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => {
          if (!seeking) setCurrentTime(e.currentTarget.duration > 0 ? e.currentTarget.currentTime : 0);
        }}
      />

      {/* Artwork + info */}
      <div className="flex items-center gap-3 w-56 shrink-0 min-w-0">
        {audioArtwork && (
          <img src={audioArtwork} alt="" className="w-11 h-11 rounded shrink-0" />
        )}
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{audioTitle || "Playing"}</p>
          {audioShowName && (
            <p className="text-xs text-muted-foreground truncate">{audioShowName}</p>
          )}
        </div>
      </div>

      {/* Controls + seek bar */}
      <div className="flex-1 flex flex-col gap-1 min-w-0">
        {/* Transport */}
        <div className="flex items-center justify-center gap-2">
          <Button onClick={() => skip(-15)} variant="ghost" size="icon" className="h-7 w-7">
            <SkipBack className="w-3.5 h-3.5" />
          </Button>
          <Button onClick={togglePlay} variant="ghost" size="icon" className="h-8 w-8">
            {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </Button>
          <Button onClick={() => skip(15)} variant="ghost" size="icon" className="h-7 w-7">
            <SkipForward className="w-3.5 h-3.5" />
          </Button>
        </div>
        {/* Seek bar + time */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground w-10 text-right shrink-0">
            {formatTime(currentTime)}
          </span>
          <div className="flex-1 relative h-1.5 group">
            <div className="absolute inset-0 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-foreground/50 rounded-full transition-[width] duration-100"
                style={{ width: `${progress}%` }}
              />
            </div>
            <input
              type="range"
              min={0}
              max={duration || 1}
              step={0.1}
              value={currentTime}
              onChange={handleSeek}
              onMouseDown={() => setSeeking(true)}
              onMouseUp={() => setSeeking(false)}
              className="absolute inset-0 w-full opacity-0 cursor-pointer"
            />
          </div>
          <span className="text-[10px] text-muted-foreground w-10 shrink-0">
            {formatTime(duration)}
          </span>
        </div>
      </div>

      {/* Volume */}
      <div className="flex items-center gap-1.5 shrink-0">
        <Button
          onClick={() => setMuted(!muted)}
          variant="ghost"
          size="icon"
          className="h-7 w-7"
        >
          {muted || volume === 0 ? (
            <VolumeX className="w-3.5 h-3.5" />
          ) : (
            <Volume2 className="w-3.5 h-3.5" />
          )}
        </Button>
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
          className="w-20 accent-primary h-1"
        />
      </div>

      {/* Close */}
      <Button onClick={stopAudio} variant="ghost" size="icon" className="h-7 w-7 shrink-0">
        <X className="w-3.5 h-3.5" />
      </Button>
    </div>
  );
}
