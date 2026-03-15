"""
podcodex.core.translate — Translation pipeline for podcast transcripts.

Translates source text to a target language without correcting the source.
(For source correction only, use podcodex.core.polish.)

Modes:
    - manual  : user provides the translated JSON directly (e.g. via a LLM UI)
    - ollama  : local LLM via Ollama
    - api     : external API (OpenAI, Anthropic, etc.)

Output files:
    {stem}.{lang_norm}.json  — translated segments, e.g. {stem}.english.json
                               (text field = translation, no text_trad)
"""

import json
import os
import re
import shutil
from pathlib import Path

from loguru import logger

from podcodex.core._paths import episode_output_dir
from podcodex.core._utils import simplify_transcript


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

_INTERNAL_SUFFIXES = frozenset(
    {
        "transcript",
        "transcript.raw",
        "polished",
        "polished.raw",
        "words",
        "diar",
        "assigned",
        "speaker_map",
        "imported",
        "segments.meta",
        "diarization.meta",
    }
)


def _lang_norm(lang: str) -> str:
    return lang.lower().strip().replace(" ", "_")


def _translation_json(
    audio_path: Path, lang: str, output_dir: str | Path | None = None
) -> Path:
    return (
        episode_output_dir(audio_path, output_dir)
        / f"{audio_path.stem}.{_lang_norm(lang)}.json"
    )


def _translation_raw_json(
    audio_path: Path, lang: str, output_dir: str | Path | None = None
) -> Path:
    return (
        episode_output_dir(audio_path, output_dir)
        / f"{audio_path.stem}.{_lang_norm(lang)}.raw.json"
    )


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


def _build_translate_prompt(
    context: str = "", source_lang: str = "French", target_lang: str = "English"
) -> str:
    context_block = (
        f"Context about this podcast: {context}\n"
        f"Any names, titles, brands, or terms mentioned in the context above are the CORRECT spellings.\n"
        if context
        else ""
    )
    return f"""You are translating a transcript from a podcast in {source_lang}.
{context_block}
Your task: translation only.
- Translate into natural, conversational {target_lang}
- Preserve the oral tone and style of the podcast
- Do not translate proper nouns (people, films, places)
- Translate the full text — never truncate or summarize
- Segments with speaker "[BREAK]" are music or jingle breaks — copy them to the output exactly as-is, do not translate their text

Output format:
Return a JSON array with one entry per segment, containing only the index and translated text.
Reply ONLY with valid JSON, no surrounding text, no markdown.
Format: [{{"index": 0, "text": "translated text..."}}]"""


# ──────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────


def _translate_batch(
    batch: list[dict],
    system_prompt: str,
    call_fn,
) -> list[dict]:
    """
    Translate a single batch of segments using the provided call function.
    call_fn(messages) -> raw response string

    LLM returns [{"index": i, "text": "translated"}]; result stored in text_trad.
    Source text is kept unchanged in the entry.
    """

    user_content = "\n\n".join(f"[{i}] {seg['text']}" for i, seg in enumerate(batch))
    user_content += f"\n\nTranslate all {len(batch)} numbered segments above."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = call_fn(messages)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        by_index = {item.get("index", i): item for i, item in enumerate(parsed)}
    except Exception as e:
        logger.warning(
            f"Parse error: {e} — batch will keep original text, empty translation"
        )
        by_index = {}

    results = []
    for i, seg in enumerate(batch):
        item = by_index.get(i, {})
        entry = {**seg}
        entry.pop("index", None)
        entry["text_trad"] = item.get("text", "")
        results.append(entry)

    return results


def _translate_ollama(
    segments: list[dict],
    context: str,
    source_lang: str,
    target_lang: str,
    model: str,
    batch_size: int,
) -> list[dict]:
    from ollama import Client

    client = Client()
    system_prompt = _build_translate_prompt(
        context, source_lang=source_lang, target_lang=target_lang
    )
    results = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        logger.info(
            f"Translate batch {i // batch_size + 1}/{-(-len(segments) // batch_size)} via Ollama ({model})"
        )

        def call_fn(messages):
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0},
                format="json",
            )
            return response.message.content.strip()

        results.extend(_translate_batch(batch, system_prompt, call_fn))

    return results


def _translate_api(
    segments: list[dict],
    context: str,
    source_lang: str,
    target_lang: str,
    model: str,
    api_base_url: str,
    api_key: str | None,
    batch_size: int,
) -> list[dict]:
    from openai import OpenAI

    key = api_key or os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "No API key found. Set API_KEY in your .env file, or pass api_key=."
        )

    client = OpenAI(api_key=key, base_url=api_base_url)
    system_prompt = _build_translate_prompt(
        context, source_lang=source_lang, target_lang=target_lang
    )
    results = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        logger.info(
            f"Translate batch {i // batch_size + 1}/{-(-len(segments) // batch_size)} via API ({model})"
        )

        def call_fn(messages):
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            return response.choices[0].message.content.strip()

        results.extend(_translate_batch(batch, system_prompt, call_fn))

    return results


def _validate_manual_translate(
    translations: list[dict], original_segments: list[dict]
) -> list[dict]:
    """
    Merge LLM-returned translations with original source segments.

    Args:
        translations     : list of {"index": i, "text": "translated text"} from LLM
        original_segments: source segments (speaker, start, end, text, ...)
    """
    if not isinstance(translations, list) or not translations:
        raise ValueError("Expected a non-empty JSON array from the LLM.")
    if "text" not in translations[0]:
        raise ValueError(
            f"Expected 'text' field in each translation entry. "
            f"Fields found: {sorted(translations[0].keys())}"
        )

    by_index = {item.get("index", i): item for i, item in enumerate(translations)}

    results = []
    for i, seg in enumerate(original_segments):
        item = by_index.get(i, {})
        translated = item.get("text", "")
        if not translated:
            logger.warning(f"Segment [{i}] has no translation — field will be empty")
        entry = {**seg, "text_trad": translated}
        entry.pop("index", None)
        results.append(entry)

    logger.info(f"Manual translation validated — {len(results)} segments")
    return results


def translate_segments(
    segments: list[dict],
    mode: str = "ollama",
    context: str = "",
    source_lang: str = "French",
    target_lang: str = "English",
    model: str = "",
    api_base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    batch_size: int = 10,
    original_segments: list[dict] | None = None,
    simplify: bool = True,
) -> list[dict]:
    """
    Translate transcript segments to the target language.

    The source text field is kept unchanged; text_trad receives the translation.

    Args:
        segments          : for manual mode — LLM output [{"index": i, "text": "..."}];
                            for ollama/api mode — source segments to translate
        mode              : "manual", "ollama", or "api"
        context           : podcast context to guide the LLM
        source_lang       : source language (e.g. "French")
        target_lang       : target language (e.g. "English")
        model             : LLM model name
        api_base_url      : base URL for OpenAI-compatible API
        api_key           : API key (None reads from API_KEY env variable)
        batch_size        : number of segments per LLM call
        original_segments : required for manual mode — the source segments being translated
        simplify          : merge consecutive same-speaker segments before processing (default True)

    Returns:
        List of segments with text (unchanged) and text_trad (translation) fields.
    """
    if mode == "manual":
        orig = original_segments if original_segments is not None else segments
        return _validate_manual_translate(segments, orig)

    if simplify:
        segments = simplify_transcript(segments)

    if mode == "ollama":
        return _translate_ollama(
            segments,
            context=context,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model or "qwen3:4b",
            batch_size=batch_size,
        )
    elif mode == "api":
        return _translate_api(
            segments,
            context=context,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model or "gpt-4o",
            api_base_url=api_base_url,
            api_key=api_key,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from 'manual', 'ollama', 'api'."
        )


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────


def save_translation_raw(
    audio_path: Path | str,
    segments: list[dict],
    lang: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Save pipeline-generated translation to {stem}.{lang_norm}.raw.json.

    Use this for LLM/pipeline output. The user can then review and promote
    to the validated {stem}.{lang_norm}.json via promote_translation().
    """
    audio_path = Path(audio_path)
    trans = [
        {
            **{k: v for k, v in s.items() if k != "text_trad"},
            "text": s.get("text_trad", s["text"]),
        }
        for s in segments
    ]
    out = _translation_raw_json(audio_path, lang, output_dir=output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trans, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Translation ({lang}) saved — {len(trans)} segments → {out.name}")
    return out


def save_translation(
    audio_path: Path | str,
    segments: list[dict],
    lang: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Save validated/edited translation to {stem}.{lang_norm}.json."""
    audio_path = Path(audio_path)
    trans = [
        {
            **{k: v for k, v in s.items() if k != "text_trad"},
            "text": s.get("text_trad", s["text"]),
        }
        for s in segments
    ]
    out = _translation_json(audio_path, lang, output_dir=output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trans, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Translation ({lang}) saved — {len(trans)} segments → {out.name}")
    return out


def load_translation(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load translated segments. Prefers validated .json, falls back to .raw.json."""
    audio_path = Path(audio_path)
    validated = _translation_json(audio_path, lang, output_dir=output_dir)
    raw = _translation_raw_json(audio_path, lang, output_dir=output_dir)
    path = validated if validated.exists() else raw
    return json.loads(path.read_text(encoding="utf-8"))


def load_translation_raw(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load specifically from .{lang}.raw.json (pipeline output)."""
    return json.loads(
        _translation_raw_json(Path(audio_path), lang, output_dir=output_dir).read_text(
            encoding="utf-8"
        )
    )


def load_translation_validated(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load specifically from .{lang}.json (user-validated)."""
    return json.loads(
        _translation_json(Path(audio_path), lang, output_dir=output_dir).read_text(
            encoding="utf-8"
        )
    )


def translation_exists(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> bool:
    """True if either validated or raw translation file exists."""
    audio_path = Path(audio_path)
    return (
        _translation_json(audio_path, lang, output_dir=output_dir).exists()
        or _translation_raw_json(audio_path, lang, output_dir=output_dir).exists()
    )


def list_translations(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> list[str]:
    """Return sorted list of available translation language names for this episode.

    Scans for {stem}.*.json files in output_dir, excluding internal suffixes.
    Merges raw and validated: english.json and english.raw.json both contribute
    "english" once. Returns normalised language names, e.g. ["english", "spanish"].
    """
    audio_path = Path(audio_path)
    root = episode_output_dir(audio_path, output_dir)
    langs: set[str] = set()
    for f in sorted(root.glob(f"{audio_path.stem}.*.json")):
        suffix = f.stem[len(audio_path.stem) + 1 :]
        if suffix in _INTERNAL_SUFFIXES:
            continue
        if suffix.endswith(".raw"):
            base = suffix[:-4]  # strip ".raw"
            if base not in _INTERNAL_SUFFIXES:
                langs.add(base)
        else:
            langs.add(suffix)
    return sorted(langs)


def translation_raw_exists(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> bool:
    """True if {lang}.raw.json exists (regardless of validated state)."""
    return _translation_raw_json(Path(audio_path), lang, output_dir=output_dir).exists()


def has_raw_translation(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> bool:
    """True if {lang}.raw.json exists but {lang}.json (validated) does not."""
    audio_path = Path(audio_path)
    return (
        _translation_raw_json(audio_path, lang, output_dir=output_dir).exists()
        and not _translation_json(audio_path, lang, output_dir=output_dir).exists()
    )


def is_validated_translation(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> bool:
    """True if the validated {lang}.json exists."""
    return _translation_json(Path(audio_path), lang, output_dir=output_dir).exists()


def promote_translation(
    audio_path: Path | str, lang: str, output_dir: str | Path | None = None
) -> Path:
    """Copy {lang}.raw.json → {lang}.json (promote to validated).

    Raises FileNotFoundError if no raw file exists.
    Returns the path of the validated file.
    """

    audio_path = Path(audio_path)
    raw = _translation_raw_json(audio_path, lang, output_dir=output_dir)
    validated = _translation_json(audio_path, lang, output_dir=output_dir)
    if not raw.exists():
        raise FileNotFoundError(f"No raw translation ({lang}): {raw}")
    shutil.copy2(raw, validated)
    logger.info(f"Translation ({lang}) promoted: {raw.name} → {validated.name}")
    return validated


# ──────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────


def translation_to_text(segments: list[dict], lang: str = "source") -> str:
    """
    Format segments as plain readable text.

    Args:
        segments : segments list (text field = translation when loaded from translation file)
        lang     : "source" (text field as-is) or "trad" (text_trad field, for in-memory segments)
    """
    lines = []
    for seg in segments:
        header = f"[{seg['start']:.3f}s - {seg['end']:.3f}s] {seg['speaker']}"
        if lang == "trad":
            lines.append(f"{header}\n{seg.get('text_trad', '[not translated]')}")
        else:
            lines.append(f"{header}\n{seg['text']}")
    return "\n\n".join(lines)


# ──────────────────────────────────────────────
# Manual translation helper
# ──────────────────────────────────────────────


def build_manual_translate_prompt(
    segments: list[dict],
    context: str = "",
    source_lang: str = "French",
    target_lang: str = "English",
) -> str:
    """Generate a prompt to paste into a LLM UI for manual translation."""
    segments_text = "\n\n".join(
        f"[{i}] {seg['text']}" for i, seg in enumerate(segments)
    )
    instruction = (
        f"Translate the {len(segments)} segments below from {source_lang} to {target_lang}.\n"
        "Return a JSON array with only the index and translated text — no other fields.\n"
        "Segments marked [BREAK] are music breaks — return them with an empty text string.\n"
        "Reply ONLY with valid JSON, no surrounding text, no markdown.\n"
        'Format: [{"index": 0, "text": "translated text..."}]'
    )
    if context:
        context_block = (
            f"Context about this podcast: {context}\n"
            f"Any names, titles, brands, or terms mentioned in the context above are the CORRECT spellings.\n\n"
        )
    else:
        context_block = ""
    return f"{context_block}{instruction}\n\n{segments_text}"


def build_manual_translate_prompts_batched(
    segments: list[dict],
    batch_minutes: float = 15.0,
    context: str = "",
    source_lang: str = "French",
    target_lang: str = "English",
) -> list[tuple[list[dict], str]]:
    """Split segments into batches and return one prompt per batch."""
    max_seconds = batch_minutes * 60
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_duration = 0.0

    for seg in segments:
        seg_duration = seg.get("end", 0) - seg.get("start", 0)
        if current and current_duration + seg_duration > max_seconds:
            batches.append(current)
            current = []
            current_duration = 0.0
        current.append(seg)
        current_duration += seg_duration

    if current:
        batches.append(current)

    return [
        (
            batch,
            build_manual_translate_prompt(
                batch,
                context=context,
                source_lang=source_lang,
                target_lang=target_lang,
            ),
        )
        for batch in batches
    ]
