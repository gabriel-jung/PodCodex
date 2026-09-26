"""The LLM batch pipeline behind correct and translate.

Segments are batched by duration, sent to Ollama or an OpenAI-compatible
API, parsed back by position, and applied with a length guard; every batch's
outcome is recorded for ``llm_failures.json``. Split out of ``core/_utils``,
whose callers already used this part and the rest separately.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from loguru import logger

from podcodex.core._utils import (
    BREAK_SPEAKER,
    DEFAULT_BATCH_MINUTES,
    DEFAULT_MAX_GAP,
    merge_consecutive_segments,
)


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Knobs for the schema-constrained correction call. See `run_ollama`.
OLLAMA_READ_TIMEOUT_S = 600.0
OLLAMA_CONNECT_TIMEOUT_S = 10.0
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT_MAX = 8192
OLLAMA_NUM_CTX_MAX = 16384
# Largest batch whose `_ollama_token_budget` estimate still fits both caps
# (about 60 items overflows num_ctx). Batches are split by duration first;
# a dense stretch of short turns can exceed this, so it is split again.
OLLAMA_MAX_BATCH_ITEMS = 55
OLLAMA_PROBE_TIMEOUT_S = 5.0

# Bounded retry for the hosted-API path, matching `run_ollama`'s shape. A
# long episode is many batches and every one already completed is lost when
# a call raises out of the loop, so a single provider hiccup used to throw
# away a whole paid run.
API_MAX_ATTEMPTS = 3
API_BACKOFF_BASE_S = 2.0
# Ceiling on a provider-supplied Retry-After. Some return minutes, which is
# longer than a user will sit in front of a progress bar.
API_RETRY_AFTER_MAX_S = 60.0
# One request's ceiling on the hosted-API path. The OpenAI client runs with
# max_retries=0 so `_api_call_with_retry` is the only retry ladder; the SDK's
# own 2 retries inside our 3 attempts used to make 9 requests per batch.
API_READ_TIMEOUT_S = 300.0
API_CONNECT_TIMEOUT_S = 10.0


def ollama_host() -> str:
    """Return the Ollama daemon URL, honoring the ``OLLAMA_HOST`` env var.

    Centralizes host resolution so both the pipeline (``run_ollama``) and
    the API health check route resolve the same target.
    """
    return _normalize_ollama_host(os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST)


def _normalize_ollama_host(host: str) -> str:
    """Give *host* a scheme and port the way the ollama client does.

    Ollama's own convention allows ``127.0.0.1:11434`` or a bare host; the
    client fills in the rest, but the plain httpx probes need a full URL.
    """
    host = host.strip().rstrip("/")
    explicit_scheme = "://" in host
    if not explicit_scheme:
        host = f"http://{host}"
    scheme, _, rest = host.partition("://")
    netloc, sep, path = rest.partition("/")
    if ":" not in netloc:
        # Same defaults as ollama._client._parse_host: a bare host gets the
        # daemon port, an explicit scheme gets that scheme's port.
        port = 443 if scheme == "https" else 80 if explicit_scheme else 11434
        netloc = f"{netloc}:{port}"
    return f"{scheme}://{netloc}{sep}{path}"


def _ollama_model_key(name: str) -> str:
    """Model name as the daemon lists it: an untagged name means ``:latest``."""
    return name if ":" in name else f"{name}:latest"


def list_pulled_ollama_models(host: str | None = None) -> list[str]:
    """Return sorted list of model tags pulled into the local Ollama daemon.

    Shared by the API health route and the pipeline's pre-flight check so
    both stay in lock-step on how the daemon's model list is fetched. Bounded
    by a short timeout: the client's default is none, and the health route
    calls this on a worker thread.
    """
    import httpx
    from ollama import Client

    client = Client(
        host=host or ollama_host(),
        timeout=httpx.Timeout(OLLAMA_PROBE_TIMEOUT_S, connect=OLLAMA_CONNECT_TIMEOUT_S),
    )
    resp = client.list()
    return sorted(m.model for m in resp.models if m.model)


def probe_ollama(host: str | None = None) -> dict:
    """Pulled models at *host*, or why they could not be listed.

    Returns ``{"reachable", "models", "problem", "error"}``. ``problem`` is
    None when reachable, ``"port_taken"`` when something answered on the
    port but not the Ollama API (another program holds it, so Ollama's own
    server cannot bind there), else ``"not_running"``. The one probe behind
    the settings status and the pre-run check.
    """
    from ollama import ResponseError

    host = host or ollama_host()
    try:
        models = list_pulled_ollama_models(host)
    except ResponseError as exc:
        return {
            "reachable": False,
            "models": [],
            "problem": "port_taken",
            "error": f"HTTP {exc.status_code} from {host}/api/tags: not the Ollama API",
        }
    except Exception as exc:  # noqa: BLE001 - nothing answering, or a bad host
        return {
            "reachable": False,
            "models": [],
            "problem": "not_running",
            "error": str(exc)[:300],
        }
    return {"reachable": True, "models": models, "problem": None, "error": None}


def correction_schema(n_items: int) -> dict:
    """JSON Schema for one batch of N correction items.

    Used by ``run_ollama`` and the ``scripts/debug_ollama_output.py`` probe.
    Constrains output to a JSON array of exactly N objects, each with a
    single ``text`` field, which is what stops small models from emitting
    a wrapping object or looping garbage when given ``format=<schema>``.
    """
    return {
        "type": "array",
        "minItems": n_items,
        "maxItems": n_items,
        "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    }


def _ollama_token_budget(n_items: int) -> tuple[int, int]:
    """Pick ``num_predict`` and ``num_ctx`` for a batch of N correction items.

    Schema-constrained sampling can spin forever without a num_predict cap.
    Ollama's default ``num_ctx`` is 4096 regardless of the model card, so
    we round the (prompt + output) budget up to the next power of two and
    cap to keep KV cache within typical consumer VRAM.
    """
    num_predict = min(OLLAMA_NUM_PREDICT_MAX, n_items * 120 + 64)
    rough_budget = 800 + n_items * 140 + num_predict
    num_ctx = max(4096, 1 << max(0, rough_budget - 1).bit_length())
    return num_predict, min(num_ctx, OLLAMA_NUM_CTX_MAX)


def _warn_if_ollama_too_old(host: str) -> None:
    """Structured-output ``format=<schema>`` requires Ollama >= 0.5; older
    daemons silently ignore it and the model emits whatever shape it likes."""
    import httpx

    try:
        version = (
            httpx.get(f"{host}/api/version", timeout=OLLAMA_PROBE_TIMEOUT_S)
            .json()
            .get("version", "")
        )
        # Strip prerelease suffix (e.g. "0.5.0-rc1") before int-parsing.
        major, minor = (int(p.split("-")[0]) for p in version.split(".")[:2])
        if (major, minor) < (0, 5):
            logger.warning(
                f"Ollama daemon version {version} predates structured-output "
                "support (>=0.5). Schema-constrained decoding will be ignored."
            )
    except Exception as e:
        logger.debug(f"Could not read Ollama version: {e}")


def _warn_if_model_unpulled(client, model: str) -> None:
    """Pre-flight check so a typo'd model name fails before the first batch
    instead of mid-pipeline with a confusing 404."""
    try:
        pulled = {m.model for m in client.list().models if m.model}
        if _ollama_model_key(model) not in pulled:
            logger.warning(
                f"Model {model!r} not in pulled list {sorted(pulled)}; "
                "first chat call will likely 404."
            )
    except Exception as e:
        logger.debug(f"Could not list pulled models: {e}")


# LLM temperature for deterministic output in correct / translate.
DEFAULT_TEMPERATURE = 0


def build_batched_manual_prompts(
    segments: list[dict],
    build_prompt_fn,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    batch_count: int | None = None,
) -> list[tuple[list[dict], str]]:
    """Split *segments* into batches and build one prompt per batch.

    *build_prompt_fn* receives ``(batch, start_index)`` and returns the
    prompt string. ``start_index`` is the absolute count of real segments
    consumed by prior batches, so each batch's [N] markers are unique
    across the whole transcript — concatenated LLM responses then keep
    distinct positions.

    When *batch_count* is given it overrides *batch_minutes* and produces
    exactly that many batches (see batch_segments_by_duration).
    """
    batches = batch_segments_by_duration(segments, batch_minutes, batch_count)
    out: list[tuple[list[dict], str]] = []
    offset = 0
    for batch in batches:
        out.append((batch, build_prompt_fn(batch, offset)))
        _, real = _separate_breaks(batch)
        offset += len(real)
    return out


def _segment_end(seg: dict) -> float:
    """Best-effort end timestamp for a segment (falls back to start, then 0)."""
    return float(seg.get("end", seg.get("start", 0)) or 0)


def batch_segments_by_duration(
    segments: list[dict],
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    batch_count: int | None = None,
) -> list[list[dict]]:
    """Split segments into time-based batches.

    Splits by absolute segment start timestamp against fixed cutoffs, so
    the batch count tracks the audio's elapsed duration (not the sum of
    per-segment durations, which can under-count silence and produce fewer
    batches than the user requested).

    Args:
        segments      : transcript segments to batch
        batch_minutes : maximum duration per batch in minutes (default 15)
        batch_count   : when set, overrides batch_minutes and sizes batches
                        off the transcript's real span so the count tracks
                        the request without overshooting (a large silence
                        gap straddling a cutoff can still yield one fewer).

    Returns:
        List of non-empty segment batches (each batch is a list of segment dicts).
    """
    if not segments:
        return []
    if batch_count is not None and batch_count >= 1:
        if batch_count == 1:
            return [list(segments)]
        max_seconds = max(_segment_end(s) for s in segments) / batch_count
    else:
        max_seconds = batch_minutes * 60
    if max_seconds <= 0:
        return [list(segments)]

    batches: list[list[dict]] = []
    current: list[dict] = []
    cutoff = max_seconds

    for seg in segments:
        start = float(seg.get("start", 0))
        while start >= cutoff:
            if current:
                batches.append(current)
                current = []
            cutoff += max_seconds
        current.append(seg)

    if current:
        batches.append(current)

    # Merge a tiny overshoot tail into the previous batch. When segments
    # extend just past the final cutoff (e.g. episode.duration under-reports
    # the real transcript span), the user's chosen batch count would otherwise
    # gain a spurious extra batch containing only the last few seconds.
    if len(batches) >= 2:
        last_batch_start = cutoff - max_seconds
        span = _segment_end(batches[-1][-1]) - last_batch_start
        if 0 <= span < max_seconds * 0.15:
            batches[-2].extend(batches.pop())

    return batches


# ──────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────


def output_format_rules(when_unchanged: str) -> str:
    """The JSON output contract every correct / translate prompt ends with.

    Positional mapping is what ``parse_llm_response`` and ``call_and_parse``
    rely on, so the contract lives next to them rather than in each prompt.
    *when_unchanged* names the case where the model must copy a segment
    verbatim ("you cannot improve it", "untranslatable").
    """
    return f"""\
Output format — CRITICAL RULES:
1. Return a JSON array with EXACTLY the same number of elements as the input, in the SAME ORDER. Never merge, split, drop, add, or reorder segments.
2. Each element is a plain object with a single `text` field — no index, no other fields.
3. The Nth output element corresponds to the Nth input segment. Position is the only mapping.
4. If a segment is empty, trivial, or {when_unchanged}, copy the original text verbatim — never omit the entry.
5. Reply ONLY with valid JSON. No surrounding text, no markdown fences, no commentary.
6. Format: `[{{"text": "..."}}, {{"text": "..."}}, ...]`"""


def build_llm_prompt(
    role: str,
    task: str,
    output: str,
    context: str = "",
    context_extra: str = "",
) -> str:
    """Assemble a system prompt from standard sections.

    Args:
        role          : opening role sentence
        task          : bullet-list of task instructions
        output        : output format instructions
        context       : optional podcast context; omitted when empty
        context_extra : additional sentence appended to the context block
    """
    context_section = (
        f"Context about this podcast: {context}\n"
        "Any names, titles, brands, or terms mentioned in the context above are the CORRECT spellings."
        + (f" {context_extra}" if context_extra else "")
        if context
        else ""
    )
    sections = [role, context_section, task, output]
    return "\n\n".join(s for s in sections if s)


# ──────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────


def _is_break(seg: dict) -> bool:
    """Return True for [BREAK] segments (music/jingle markers)."""
    return seg.get("speaker") == BREAK_SPEAKER


def _separate_breaks(
    segments: list[dict],
) -> tuple[list[int], list[dict]]:
    """Split segments into real content and [BREAK] markers.

    Returns:
        (real_indices, real_segments) — positions and segments that are
        not ``[BREAK]`` markers.
    """
    real_indices: list[int] = []
    real_segs: list[dict] = []
    for i, seg in enumerate(segments):
        if not _is_break(seg):
            real_indices.append(i)
            real_segs.append(seg)
    return real_indices, real_segs


def _reassemble_breaks(
    segments: list[dict],
    real_indices: list[int],
    processed: list[dict],
) -> list[dict]:
    """Merge processed results back with [BREAK] segments in original order."""
    real_set = set(real_indices)
    results: list[dict] = []
    proc_iter = iter(processed)
    for i, seg in enumerate(segments):
        if i in real_set:
            results.append(next(proc_iter))
        else:
            results.append(seg)
    return results


def format_segments(
    segments: list[dict],
    instruction: str = "Process",
    start_index: int = 0,
) -> str:
    """Format segments as a numbered user message for the LLM.

    Produces the same ``[i] text`` format used by all three modes
    (ollama, api, manual).  ``[BREAK]`` segments are excluded.

    Args:
        segments    : transcript segments (breaks are filtered out)
        instruction : verb for the closing instruction line
        start_index : first absolute index for numbering (used by manual mode
                      so concatenated batch responses keep unique indices)
    """
    _, real = _separate_breaks(segments)
    n = len(real)
    lines = [f"[{start_index + i}] {seg['text']}" for i, seg in enumerate(real)]
    first = start_index
    last = start_index + n - 1 if n > 0 else start_index
    lines.append(
        f"\n{instruction} all {n} segments above. "
        f"Output MUST contain exactly {n} entries with indices {first}..{last}, "
        "no gaps, no extras, no renumbering. Verify the count before responding."
    )
    return "\n\n".join(lines)


def parse_llm_response(raw: str) -> dict[int, dict]:
    """Parse a raw LLM response string into a dict keyed by segment position.

    Keys are positional (0..N-1) — the LLM's own ``index`` field is treated
    as advisory only. Position-based mapping is safer because callers verify
    ``len(parsed) == len(input)`` before applying, so position equals the
    intended target index regardless of any renumbering by the LLM.

    Strips ``<think>`` tags and markdown fences before parsing JSON. Falls
    back through tiered repair (trim trailing junk, fix invalid ``\\'``
    escape, regex-extract orphan ``"text": "..."`` pairs) so a single
    schema slip from a small model doesn't drop the whole batch.

    Returns:
        ``{position: {"text": "...", ...}}`` dict.  Empty dict on parse failure.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    def _to_index(parsed: object) -> dict[int, dict]:
        # Only an array maps to positions. Enumerating anything else turned a
        # single `{"text": "Oui."}` into the item "text" (a dict iterates its
        # keys) and a bare string into one item per character. A lone object
        # with a text field is one item; an object wrapping one array is that
        # array; everything else is a parse failure.
        if isinstance(parsed, dict):
            if "text" in parsed:
                parsed = [parsed]
            else:
                arrays = [v for v in parsed.values() if isinstance(v, list)]
                if len(arrays) != 1:
                    raise ValueError("top-level JSON object is not a segment list")
                parsed = arrays[0]
        if not isinstance(parsed, list):
            raise ValueError(f"top-level JSON {type(parsed).__name__}, not a list")
        out: dict[int, dict] = {}
        for i, item in enumerate(parsed):
            if isinstance(item, dict):
                out[i] = item
            elif isinstance(item, str):
                out[i] = {"text": item}
        return out

    try:
        return _to_index(json.loads(raw))
    except Exception as first_err:
        # Cross-model repair: trailing junk after the array close, and the
        # `\'` escape (invalid in JSON, valid in Python/JS). Both seen on
        # multiple model sizes, not just one run.
        repaired = raw
        last_bracket = repaired.rfind("]")
        if last_bracket != -1:
            repaired = repaired[: last_bracket + 1]
        repaired = repaired.replace("\\'", "'")
        try:
            return _to_index(json.loads(repaired))
        except Exception:
            pass

        logger.warning(f"Parse error: {first_err}, batch will keep original text")
        logger.warning(f"Raw response (first 600 chars): {raw[:600]}")
        return {}


# The "[12] " position markers format_segments puts before each segment.
# Small models copy them into their answers, sometimes more than once.
_ECHOED_MARKERS = re.compile(r"^\s*(?:\[\d+\]\s*)+")


def _strip_echoed_markers(text: str, original: str) -> str:
    """*text* without leading ``[N]`` markers the prompt added.

    Left alone when the source itself starts with one, so real text such as
    a footnote reference survives.
    """
    if not isinstance(text, str) or _ECHOED_MARKERS.match(original):
        return text
    return _ECHOED_MARKERS.sub("", text, count=1)


def apply_corrections(
    batch: list[dict],
    by_index: dict[int, dict],
    min_length_ratio: float = 0.7,
    reverted: list[int] | None = None,
) -> list[dict]:
    """Apply LLM corrections to a batch of segments.

    Merges corrected text from *by_index* into the original segments.
    ``[BREAK]`` segments are passed through unchanged.  Segments whose
    corrected text is suspiciously short (below *min_length_ratio* of the
    original) keep their original text.

    Args:
        batch            : original segments (may include ``[BREAK]``s)
        by_index         : ``{index: {"text": "..."}}`` from the LLM
        min_length_ratio : minimum corrected/original length ratio (0 to
                           disable). A correction guard only: translation
                           changes length legitimately (English to Chinese
                           is often a third of the characters).
        reverted         : when given, receives the batch position of every
                           segment that kept its original text because the
                           LLM output was empty or too short.

    Returns:
        List of segments with text field updated.
    """
    real_indices, real_segs = _separate_breaks(batch)

    corrected_segs: list[dict] = []
    changed = 0
    for i, seg in enumerate(real_segs):
        item = by_index.get(i, {})
        original_text = seg["text"]
        corrected_text = _strip_echoed_markers(
            item.get("text", original_text), original_text
        )

        if not corrected_text:
            logger.warning(f"Segment [{i}] has no corrected text, keeping original")
            corrected_text = original_text
            if original_text and reverted is not None and i in by_index:
                reverted.append(i)

        if (
            min_length_ratio
            and original_text
            and len(corrected_text) < len(original_text) * min_length_ratio
        ):
            logger.warning(
                f"Segment [{i}] truncated by LLM "
                f"({len(corrected_text)} vs {len(original_text)} chars), keeping original. "
                f"original={original_text!r:.120} corrected={corrected_text!r:.120}"
            )
            corrected_text = original_text
            if reverted is not None:
                reverted.append(i)

        if corrected_text != original_text:
            changed += 1
        entry = {**seg, "text": corrected_text}
        entry.pop("index", None)
        corrected_segs.append(entry)

    logger.debug(f"Batch: {changed}/{len(real_segs)} segments modified")
    return _reassemble_breaks(batch, real_indices, corrected_segs)


# Short segments ("OK.", a name) legitimately survive translation unchanged,
# so only segments at least this long count as evidence.
_UNCHANGED_MIN_CHARS = 20


def _mostly_unchanged(
    inputs: list[dict], outputs: list[dict]
) -> tuple[int, int] | None:
    """``(unchanged, eligible)`` when most substantial segments came back as
    they went in, else None.

    A small model asked to translate sometimes returns the source verbatim
    for a whole batch; the count and length checks both pass, so without
    this the batch saves as a finished translation. Language-agnostic: it
    compares output to input rather than guessing the script.
    """
    eligible = unchanged = 0
    for src, out in zip(inputs, outputs):
        text = src["text"].strip()
        if len(text) < _UNCHANGED_MIN_CHARS:
            continue
        eligible += 1
        if out["text"].strip() == text:
            unchanged += 1
    return (unchanged, eligible) if eligible and unchanged * 2 > eligible else None


def call_and_parse(
    batch: list[dict],
    system_prompt: str,
    call_fn,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    flag_unchanged: bool = False,
    start_index: int = 0,
    on_outcome: Callable[[str, int, int, str, str], None] | None = None,
) -> list[dict]:
    """Call the LLM for one batch and parse the response.

    Uses :func:`format_segments`, :func:`parse_llm_response`, and
    :func:`apply_corrections` — the same pipeline that manual mode uses.
    ``[BREAK]`` segments are passed through unchanged.

    ``start_index`` shifts the displayed ``[N]`` markers in the prompt so
    log lines and the LLM see absolute positions across batches.

    ``on_outcome``, when given, is called once with
    ``(raw, expected, got, status, reason)`` — ``status`` is ``"ok"`` or
    ``"rejected"`` — so a caller can record per-batch results.

    ``flag_unchanged`` rejects a batch whose output is mostly its input
    (see :func:`_mostly_unchanged`). The text is kept as returned: the flag
    only marks the batch for a retry, since a short or proper-noun segment
    can legitimately stay the same.
    """
    _, real_segs = _separate_breaks(batch)
    if not real_segs:
        return list(batch)

    user_content = format_segments(
        batch, instruction=instruction, start_index=start_index
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = call_fn(messages)
    logger.debug(f"LLM response: {len(raw)} chars")
    by_index = parse_llm_response(raw)

    expected = len(real_segs)
    got = len(by_index)
    status, reason = "ok", ""

    if not by_index:
        # parse_llm_response already logged the parse error; flag the batch
        # so the run's failure record shows it produced no corrections.
        status, reason = "rejected", "parse failure"
    elif got != expected:
        # LLM count drift: a response with fewer/more items than the input
        # batch means indices got renumbered, which would silently misalign
        # corrections. Reject the whole batch and keep originals.
        logger.warning(
            f"LLM returned {got} items for {expected} segments, "
            "rejecting batch to avoid index drift; keeping original text."
        )
        # Surface enough of the raw response to diagnose the shape mismatch
        # (top-level object vs array, wrapping key, single concatenated
        # string). Without this the rejection is opaque.
        logger.warning(f"Rejected raw response (first 600 chars): {raw[:600]}")
        sample = next(iter(by_index.values()), None)
        logger.warning(
            f"Rejected first item shape: {type(sample).__name__}={sample!r:.300}"
        )
        status, reason = "rejected", "count drift"
        by_index = {}

    reverted: list[int] = []
    results = apply_corrections(
        batch, by_index, min_length_ratio=min_length_ratio, reverted=reverted
    )
    if status == "ok" and reverted:
        # A segment that fell back to its original is a partial rejection:
        # the batch saves as done while that text was never processed.
        status = "rejected"
        reason = (
            f"{len(reverted)} of {expected} segments kept their original text "
            "(LLM output empty or much shorter than the input)"
        )
    if status == "ok" and flag_unchanged:
        unchanged = _mostly_unchanged(real_segs, _separate_breaks(results)[1])
        if unchanged:
            status = "rejected"
            reason = (
                f"{unchanged[0]} of {unchanged[1]} segments came back "
                "unchanged (not translated)"
            )

    if on_outcome is not None:
        on_outcome(raw, expected, got, status, reason)

    return results


def _append_batch_record(
    batch_sink: list[dict] | None,
    *,
    batch_num: int,
    real_segs: list[dict],
    offset: int,
    raw: str,
    expected: int,
    got: int,
    status: str,
    reason: str,
) -> None:
    """Append one batch's LLM outcome to *batch_sink* (no-op when None)."""
    if batch_sink is None:
        return
    batch_sink.append(
        {
            "batch": batch_num,
            "status": status,
            "reason": reason,
            "expected": expected,
            "got": got,
            "raw": raw,
            "input": [
                {"index": offset + i, "text": s.get("text", "")}
                for i, s in enumerate(real_segs)
            ],
        }
    )


def _outcome_recorder(
    batch_sink: list[dict] | None,
    batch_num: int,
    real_segs: list[dict],
    offset: int,
) -> Callable[[str, int, int, str, str], None]:
    """Build a ``call_and_parse`` on_outcome callback bound to one batch."""

    def record(raw: str, expected: int, got: int, status: str, reason: str) -> None:
        _append_batch_record(
            batch_sink,
            batch_num=batch_num,
            real_segs=real_segs,
            offset=offset,
            raw=raw,
            expected=expected,
            got=got,
            status=status,
            reason=reason,
        )

    return record


class LLMBatchError(RuntimeError):
    """A provider call failed for good; carries whether retrying can help.

    ``permanent`` errors (bad key, unknown model, bad request) fail every
    batch the same way, so the run stops at the first one. Transient ones
    that outlived the retry ladder reject only their own batch.
    """

    def __init__(self, message: str, *, permanent: bool) -> None:
        super().__init__(message)
        self.permanent = permanent


def _run_batches(
    segments: list[dict],
    system_prompt: str,
    make_call_fn: Callable[[int], Callable[[list[dict]], str]],
    *,
    batches: list[list[dict]],
    via: str,
    instruction: str,
    min_length_ratio: float,
    flag_unchanged: bool,
    label: str,
    on_batch: Callable[[int, int], None] | None,
    batch_sink: list[dict] | None,
) -> list[dict]:
    """The batch loop shared by every provider.

    ``make_call_fn(n_items)`` returns the provider call for one batch; it
    raises :class:`LLMBatchError` when the provider gave up. A transient
    failure rejects that batch (originals kept, recorded in *batch_sink*
    for the manual-fix flow) and the run continues, so one bad batch no
    longer throws away every batch already paid for. A permanent failure,
    or every batch failing, stops the run: saving untouched text as a
    finished step would be worse than an error.
    """
    results: list[dict] = []
    n_batches = len(batches)
    offset = 0
    errors: list[LLMBatchError] = []

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"{label} batch {batch_num}/{n_batches} via {via}")
        _, real_segs = _separate_breaks(batch)
        record = _outcome_recorder(batch_sink, batch_num, real_segs, offset)
        try:
            results.extend(
                call_and_parse(
                    batch,
                    system_prompt,
                    make_call_fn(len(real_segs)),
                    instruction=instruction,
                    min_length_ratio=min_length_ratio,
                    flag_unchanged=flag_unchanged,
                    start_index=offset,
                    on_outcome=record,
                )
            )
        except LLMBatchError as exc:
            record("", len(real_segs), 0, "rejected", f"provider error: {exc}")
            if exc.permanent:
                raise
            logger.warning(f"{label} batch {batch_num} failed, keeping original: {exc}")
            errors.append(exc)
            results.extend(batch)
        offset += len(real_segs)
        if on_batch:
            on_batch(batch_num, n_batches)

    if errors and len(errors) == n_batches:
        raise LLMBatchError(
            f"Every batch failed; last error: {errors[-1]}", permanent=False
        )
    return results


def _ollama_batches(segments: list[dict], batch_minutes: float) -> list[list[dict]]:
    """Duration batches, split again where one holds too many items.

    `_ollama_token_budget` caps num_ctx and num_predict; past
    OLLAMA_MAX_BATCH_ITEMS the estimate no longer fits and Ollama silently
    truncates the prompt or the JSON. [BREAK] markers do not count.
    """
    out: list[list[dict]] = []
    for batch in batch_segments_by_duration(segments, batch_minutes):
        current: list[dict] = []
        n_real = 0
        for seg in batch:
            is_real = not _is_break(seg)
            if is_real and n_real == OLLAMA_MAX_BATCH_ITEMS:
                out.append(current)
                current, n_real = [], 0
            current.append(seg)
            n_real += is_real
        if current:
            out.append(current)
    return out


# Client errors that fail every batch the same way: bad credentials, no
# access, unknown model. Any other 4xx (context length, a content filter, a
# malformed batch) is about this batch's content: reject it, keep going.
_RUN_STOPPING_STATUSES = frozenset({401, 403, 404})


def _ollama_error(exc: Exception, model: str = "") -> LLMBatchError | None:
    """Classify an Ollama call failure; None means worth another attempt.

    Connection drops and 5xx are transient. A 4xx will not change on retry:
    it stops the run when it would fail every batch (see
    ``_RUN_STOPPING_STATUSES``), else rejects this batch. A read timeout
    already waited OLLAMA_READ_TIMEOUT_S; retrying it only multiplies that
    wait, so it rejects the batch instead.
    """
    import httpx
    from ollama import ResponseError

    if isinstance(exc, ResponseError):
        status = getattr(exc, "status_code", 0) or 0
        if 400 <= status < 500:
            hint = (
                f" (is {model!r} pulled? `ollama pull {model}`)"
                if status == 404
                else ""
            )
            return LLMBatchError(
                f"Ollama {status}: {exc}{hint}",
                permanent=status in _RUN_STOPPING_STATUSES,
            )
        return None
    if isinstance(exc, httpx.TimeoutException) and not isinstance(
        exc, httpx.ConnectTimeout
    ):
        return LLMBatchError(
            f"Ollama did not answer within {OLLAMA_READ_TIMEOUT_S:g}s", permanent=False
        )
    return None


def run_ollama(
    segments: list[dict],
    system_prompt: str,
    model: str,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    flag_unchanged: bool = False,
    label: str = "",
    on_batch: Callable[[int, int], None] | None = None,
    batch_sink: list[dict] | None = None,
) -> list[dict]:
    """Run segments through a local Ollama model.

    Args:
        segments: source segments to process.
        system_prompt: system prompt for the LLM.
        model: Ollama model name.
        batch_minutes: max audio duration per batch in minutes.
        instruction: verb for user-message formatting (e.g. "Correct", "Translate").
        min_length_ratio: minimum output/input length ratio before a segment
            keeps its original text (0 disables; see apply_corrections).
        flag_unchanged: reject a batch whose output is mostly its input
            (see call_and_parse); translation sets it.
        label: human-readable label for log messages.
        on_batch: optional callback(batch_num, total_batches) for progress.

    Returns:
        Processed segments with updated text fields.
    """
    import time

    import httpx
    from ollama import Client, ResponseError

    host = ollama_host()
    client = Client(
        host=host,
        timeout=httpx.Timeout(OLLAMA_READ_TIMEOUT_S, connect=OLLAMA_CONNECT_TIMEOUT_S),
    )
    _warn_if_ollama_too_old(host)
    _warn_if_model_unpulled(client, model)

    def make_call_fn(n_items: int) -> Callable[[list[dict]], str]:
        schema = correction_schema(n_items)
        num_predict, num_ctx = _ollama_token_budget(n_items)

        def call_fn(messages):
            for attempt in range(3):
                try:
                    response = client.chat(
                        model=model,
                        messages=messages,
                        options={
                            "temperature": OLLAMA_TEMPERATURE,
                            "num_predict": num_predict,
                            "num_ctx": num_ctx,
                        },
                        format=schema,
                        # Reasoning models (Qwen3, DeepSeek-R1) emit
                        # `<think>...</think>` before the answer, which
                        # conflicts with schema-constrained decoding and
                        # burns the whole num_predict budget producing zero
                        # JSON output.
                        think=False,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                    break
                except (httpx.HTTPError, ResponseError, ConnectionError) as e:
                    final = _ollama_error(e, model)
                    if final is not None:
                        raise final from e
                    if attempt == 2:
                        raise LLMBatchError(
                            f"Ollama call failed 3 times: {e}", permanent=False
                        ) from e
                    backoff = 2**attempt
                    logger.warning(
                        f"Ollama call failed (attempt {attempt + 1}/3): {e}. "
                        f"Retrying in {backoff}s."
                    )
                    time.sleep(backoff)

            content = (response.message.content or "").strip()
            pec = response.prompt_eval_count or 0
            # Ollama silently chops the prompt when it exceeds num_ctx; the
            # system prompt is what gets cut first, so the model ends up
            # following a partial instruction.
            if pec > num_ctx * 0.9:
                logger.warning(
                    f"Prompt used {pec}/{num_ctx} tokens (>=90%); Ollama may "
                    "have truncated the system prompt. Reduce batch size."
                )
            if not content:
                logger.warning(
                    f"Empty response from {model}. done_reason="
                    f"{response.done_reason!r} eval_count={response.eval_count} "
                    f"prompt_eval_count={pec} done={response.done}"
                )
            return content

        return call_fn

    return _run_batches(
        segments,
        system_prompt,
        make_call_fn,
        batches=_ollama_batches(segments, batch_minutes),
        via=f"Ollama ({model})",
        instruction=instruction,
        min_length_ratio=min_length_ratio,
        flag_unchanged=flag_unchanged,
        label=label,
        on_batch=on_batch,
        batch_sink=batch_sink,
    )


def _api_retry_delay(exc, attempt: int) -> float:
    """Seconds to wait before the next attempt.

    Honours a provider ``Retry-After`` when the exception carries one (429s
    usually do), capped, and otherwise falls back to the same 2s/4s ladder
    ``run_ollama`` uses.
    """
    fallback = API_BACKOFF_BASE_S**attempt
    try:
        raw = exc.response.headers.get("retry-after")
    except Exception:
        return fallback
    if not raw:
        return fallback
    try:
        # Seconds is the common form; an HTTP-date is legal but rare, and
        # the fallback is a fine answer for it.
        return min(max(float(raw), 0.0), API_RETRY_AFTER_MAX_S)
    except (TypeError, ValueError):
        return fallback


def _api_is_retryable(exc) -> bool:
    """Whether this OpenAI-client error is worth another attempt.

    Rate limits, connection drops, timeouts and 5xx are transient. Every
    other 4xx (bad key, unknown model, context length) will fail the same
    way three times, so retrying only delays the error the user needs.
    """
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return (getattr(exc, "status_code", 0) or 0) >= 500
    return False


def _api_call_with_retry(client, model: str, messages: list[dict], label: str):
    """One chat completion, retried on transient provider failures."""
    import time

    last: Exception | None = None
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=DEFAULT_TEMPERATURE
            )
        except Exception as exc:
            if not _api_is_retryable(exc):
                status = getattr(exc, "status_code", None)
                raise LLMBatchError(
                    f"{type(exc).__name__}: {exc}",
                    # Unknown shapes (no status) stop the run: safer than
                    # sending every remaining batch into the same failure.
                    permanent=status is None or status in _RUN_STOPPING_STATUSES,
                ) from exc
            if attempt == API_MAX_ATTEMPTS - 1:
                raise LLMBatchError(
                    f"failed {API_MAX_ATTEMPTS} times: {type(exc).__name__}: {exc}",
                    permanent=False,
                ) from exc
            last = exc
            delay = _api_retry_delay(exc, attempt)
            logger.warning(
                f"{label} API call failed (attempt {attempt + 1}/"
                f"{API_MAX_ATTEMPTS}): {exc}. Retrying in {delay:g}s."
            )
            time.sleep(delay)
    raise last  # unreachable: the last attempt re-raises above


def _api_response_text(response, model: str) -> str:
    """Assistant text from a completion, or ``""`` for an unusable one.

    OpenAI-compatible endpoints (gemini, groq, openrouter, anthropic's compat
    route) answer a refusal or a reasoning length-stop with ``content: null``,
    and can return an empty ``choices`` array. Indexing and stripping those
    blindly raised out of the batch loop and lost every completed batch;
    returning "" lets ``call_and_parse`` record one rejected batch instead.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        logger.warning(f"Empty choices from {model}; recording batch as rejected.")
        return ""
    choice = choices[0]
    content = getattr(choice.message, "content", None)
    if not content:
        logger.warning(
            f"Empty content from {model}. finish_reason="
            f"{getattr(choice, 'finish_reason', None)!r}"
        )
        return ""
    return content.strip()


def run_api(
    segments: list[dict],
    system_prompt: str,
    model: str,
    api_base_url: str,
    api_key: str | None,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    provider: str | None = None,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    flag_unchanged: bool = False,
    label: str = "",
    on_batch: Callable[[int, int], None] | None = None,
    batch_sink: list[dict] | None = None,
) -> list[dict]:
    """Run segments through an OpenAI-compatible API.

    Args:
        segments: source segments to process.
        system_prompt: system prompt for the LLM.
        model: model name (the provider default from `effective_llm_model`
            when empty).
        api_base_url: base URL of the OpenAI-compatible endpoint.
        api_key: API key from the key pool (``resolve_llm``).
        batch_minutes: max audio duration per batch in minutes.
        provider: provider shorthand ("openai", "anthropic", "mistral").
        instruction: verb for user-message formatting.
        min_length_ratio: minimum output/input length ratio before a segment
            keeps its original text (0 disables; see apply_corrections).
        flag_unchanged: reject a batch whose output is mostly its input
            (see call_and_parse); translation sets it.
        label: human-readable label for log messages.
        on_batch: optional callback(batch_num, total_batches) for progress.

    Returns:
        Processed segments with updated text fields.
    """
    import httpx
    from openai import OpenAI

    model = effective_llm_model("api", model, provider)
    if not api_key:
        raise ValueError("No API key found. Add one in Settings and pick it.")

    logger.debug(
        f"API config: base_url={api_base_url}, model={model}, provider={provider}"
    )
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url,
        max_retries=0,
        timeout=httpx.Timeout(API_READ_TIMEOUT_S, connect=API_CONNECT_TIMEOUT_S),
    )

    def make_call_fn(_n_items: int) -> Callable[[list[dict]], str]:
        def call_fn(messages):
            response = _api_call_with_retry(client, model, messages, label)
            return _api_response_text(response, model)

        return call_fn

    return _run_batches(
        segments,
        system_prompt,
        make_call_fn,
        batches=batch_segments_by_duration(segments, batch_minutes),
        via=f"API ({model})",
        instruction=instruction,
        min_length_ratio=min_length_ratio,
        flag_unchanged=flag_unchanged,
        label=label,
        on_batch=on_batch,
        batch_sink=batch_sink,
    )


def effective_llm_model(mode: str, model: str, provider: str | None = None) -> str:
    """The model an auto run will actually call.

    An empty pick means the default: DEFAULT_OLLAMA_MODEL for Ollama, the
    provider's entry in LLM_PROVIDER_DEFAULT_MODEL for the hosted API (empty for a
    custom endpoint, which has no default). Provenance and llm_failures.json
    record this value, not the empty string the request carried.
    """
    if model:
        return model
    from podcodex.core.constants import DEFAULT_OLLAMA_MODEL, LLM_PROVIDER_DEFAULT_MODEL

    if mode == "ollama":
        return DEFAULT_OLLAMA_MODEL
    if mode == "api" and provider in LLM_PROVIDER_DEFAULT_MODEL:
        return LLM_PROVIDER_DEFAULT_MODEL[provider]
    return model


def validate_manual(
    corrections: list[dict], original_segments: list[dict]
) -> list[dict]:
    """Merge LLM-returned corrections with original source segments.

    Uses position-based mapping — corrections must be in the same order as
    the (non-[BREAK]) source segments. The LLM-supplied ``index`` field, if
    present, is ignored.

    Args:
        corrections       : list of {"text": "..."} entries from the LLM, in order
        original_segments : source segments (speaker, start, end, text, ...)

    Returns:
        List of segments with text field updated from corrections.

    Raises:
        ValueError: the response is empty, has no ``text`` field, or its entry
            count does not match the source segments. Saving a count-mismatched
            response would persist untouched source text as a finished step.
    """
    if not isinstance(corrections, list) or not corrections:
        raise ValueError("Expected a non-empty JSON array from the LLM.")
    if "text" not in corrections[0]:
        raise ValueError(
            f"Expected 'text' field in each entry. "
            f"Fields found: {sorted(corrections[0].keys())}"
        )

    _, real_segs = _separate_breaks(original_segments)
    if len(corrections) != len(real_segs):
        # Rejecting silently used to keep the originals and let the caller save
        # them as a "translation" or "correction" that had never been touched.
        raise ValueError(
            f"Count mismatch: {len(corrections)} entries from the LLM "
            f"vs {len(real_segs)} source segments (excluding "
            f"{len(original_segments) - len(real_segs)} breaks). "
            "Paste the response for this exact source version, or re-generate "
            "the prompts."
        )
    # Position-based mapping (LLM's index field is advisory only).
    by_index = {i: item for i, item in enumerate(corrections)}
    results = apply_corrections(original_segments, by_index, min_length_ratio=0)

    logger.info(f"Manual corrections validated — {len(results)} segments")
    return results


def run_llm_step(
    step: str,
    segments: list[dict],
    system_prompt: str,
    *,
    audio_path: str | None,
    output_dir: str | None,
    records_out: list[dict] | None = None,
    **pipeline_kwargs,
) -> list[dict]:
    """Run an auto correct / translate pass and record its batch outcomes.

    *step* is the ``llm_failures.json`` section (``corrected`` or a language
    key). The record is written only after a successful run: a run that
    raises saves no version, so the section must keep describing the version
    on disk (the batch-fix flow patches it by the recorded indices). Its
    rejected batches are logged instead.

    With *records_out* the batch records are handed to the caller instead of
    written, so it can record them once, with the id of the version it then
    saves (``provenance.save_llm_run``).
    """
    from podcodex.core.llm_failures import record_run

    batch_sink: list[dict] = []
    try:
        result = run_llm_pipeline(
            segments, system_prompt, batch_sink=batch_sink, **pipeline_kwargs
        )
    except Exception:
        for rec in batch_sink:
            if rec.get("status") == "rejected":
                logger.warning(
                    "Batch {} rejected: {}", rec.get("batch"), rec.get("reason")
                )
        raise
    if records_out is not None:
        records_out.extend(batch_sink)
        return result
    mode = pipeline_kwargs.get("mode", "ollama")
    record_run(
        audio_path,
        output_dir,
        step,
        model=effective_llm_model(
            mode, pipeline_kwargs.get("model", ""), pipeline_kwargs.get("provider")
        ),
        mode=mode,
        records=batch_sink,
    )
    return result


def run_llm_pipeline(
    segments: list[dict],
    system_prompt: str,
    *,
    mode: str = "ollama",
    model: str = "",
    api_base_url: str = "",
    api_key: str | None = None,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    provider: str | None = None,
    instruction: str = "Process",
    label: str = "",
    original_segments: list[dict] | None = None,
    merge: bool = True,
    max_gap: float = DEFAULT_MAX_GAP,
    on_batch: Callable[[int, int], None] | None = None,
    batch_sink: list[dict] | None = None,
    min_length_ratio: float = 0.7,
    flag_unchanged: bool = False,
) -> list[dict]:
    """Run an LLM pipeline (correct or translate) on segments.

    Handles manual/ollama/api modes, optional merge, and progress callbacks.
    When *batch_sink* is given, each batch's LLM outcome is appended to it
    (ollama/api modes only). *min_length_ratio* is the correction length
    guard; translation passes 0. *flag_unchanged* rejects batches that
    come back mostly untouched; translation sets it.
    """
    if mode == "manual":
        orig = original_segments if original_segments is not None else segments
        return validate_manual(segments, orig)

    if merge:
        segments = merge_consecutive_segments(segments, max_gap=max_gap)
        logger.info(f"After merge: {len(segments)} segments")

    if mode == "ollama":
        return run_ollama(
            segments,
            system_prompt,
            model=effective_llm_model(mode, model),
            batch_minutes=batch_minutes,
            instruction=instruction,
            min_length_ratio=min_length_ratio,
            flag_unchanged=flag_unchanged,
            label=label,
            on_batch=on_batch,
            batch_sink=batch_sink,
        )
    elif mode == "api":
        return run_api(
            segments,
            system_prompt,
            model=model,
            api_base_url=api_base_url,
            api_key=api_key,
            batch_minutes=batch_minutes,
            provider=provider,
            instruction=instruction,
            min_length_ratio=min_length_ratio,
            flag_unchanged=flag_unchanged,
            label=label,
            on_batch=on_batch,
            batch_sink=batch_sink,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from 'manual', 'ollama', 'api'."
        )
