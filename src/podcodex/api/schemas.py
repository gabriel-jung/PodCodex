"""Pydantic request/response models for the PodCodex API."""

from __future__ import annotations

from pydantic import BaseModel


class PipelineDefaultsSchema(BaseModel):
    """Per-show pipeline default settings (mirrors ``show.toml [pipeline]``)."""

    model_size: str = ""
    diarize: bool = True
    llm_mode: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    target_lang: str = ""


class ShowMeta(BaseModel):
    """Editable metadata for a show (read/write via the settings panel)."""

    name: str
    rss_url: str = ""
    youtube_url: str = ""
    language: str = ""
    speakers: list[str] = []
    artwork_url: str = ""
    pipeline: PipelineDefaultsSchema = PipelineDefaultsSchema()


class EpisodeOut(BaseModel):
    stem: str
    title: str
    audio_path: str | None
    output_dir: str
    # pipeline status
    segments_ready: bool
    diarized: bool
    assigned: bool
    mapped: bool
    transcribed: bool
    corrected: bool
    indexed: bool
    synthesized: bool
    translations: list[str]


class RSSEpisodeOut(BaseModel):
    guid: str
    title: str
    pub_date: str
    description: str = ""
    audio_url: str = ""
    duration: float = 0.0
    episode_number: int | None = None
    season_number: int | None = None
    artwork_url: str = ""
    # local status (filled when matching against local episodes)
    local_stem: str | None = None
    downloaded: bool = False


class Segment(BaseModel):
    speaker: str = ""
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    flagged: bool = False


class UnifiedEpisodeOut(BaseModel):
    id: str  # guid or stem
    title: str
    stem: str | None = None
    pub_date: str | None = None
    description: str = ""
    audio_url: str | None = None
    duration: float = 0.0
    episode_number: int | None = None
    audio_path: str | None = None
    output_dir: str | None = (
        None  # episode directory (always set when episode dir exists)
    )
    downloaded: bool = False
    transcribed: bool = False
    corrected: bool = False
    indexed: bool = False
    synthesized: bool = False
    has_subtitles: bool = False
    translations: list[str] = []
    artwork_url: str = ""
    provenance: dict = {}
    # Key files present in episode directory
    files: list[str] = []
    # Step status: "none" | "outdated" | "done"
    transcribe_status: str = "none"
    correct_status: str = "none"
    translate_status: str = "none"


class CreateFromRSSRequest(BaseModel):
    rss_url: str
    save_path: str  # absolute path where the show folder will be created
    folder_name: str = ""  # optional subfolder name (auto-generated if empty)
    name: str = ""  # display name from search/RSS (falls back to folder_name)
    artwork_url: str = ""  # optional, passed from search result
    language: str = ""  # ISO language code (e.g. "en", "fr")


class RegisterShowRequest(BaseModel):
    path: str  # absolute path to existing show folder


class CreateFromRSSResponse(BaseModel):
    folder: str
    name: str
    episode_count: int


class CreateFromYouTubeRequest(BaseModel):
    youtube_url: str  # channel, playlist, or single video URL
    save_path: str  # absolute path where the show folder will be created
    folder_name: str = ""  # optional subfolder name (auto-generated if empty)
    name: str = ""  # display name (falls back to channel/playlist title)
    artwork_url: str = ""  # optional, passed from preview
    language: str = ""  # ISO language code (e.g. "en", "fr")


class CreateFromYouTubeResponse(BaseModel):
    folder: str
    name: str
    episode_count: int


class TaskResponse(BaseModel):
    task_id: str
