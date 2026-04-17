"""CRUD for user-managed MCP prompts.

Each mutation persists to ``~/.config/podcodex/mcp_prompts.json`` and
calls ``reregister_all`` so the live FastMCP instance picks up changes
without a Python restart. Claude Desktop still needs its own restart to
resync the catalog — we surface this via ``needs_restart_hint`` on the
response.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from podcodex.mcp.prompts import (
    PromptDef,
    PromptValidationError,
    SlotDef,
    load_prompts,
    reregister_all,
    save_prompts,
    validate_prompt,
)

router = APIRouter()


SlotType = Literal["string", "enum", "int", "bool"]


# ── Wire models ─────────────────────────────────────────────────────────


class SlotIn(BaseModel):
    name: str
    type: SlotType = "string"
    required: bool = True
    default: str | None = None
    options: list[str] = Field(default_factory=list)


class PromptOut(BaseModel):
    id: str
    name: str
    title: str
    description: str
    template: str
    slots: list[SlotIn]
    enabled: bool
    is_builtin: bool


class PromptCreate(BaseModel):
    id: str
    name: str = ""
    title: str
    description: str = ""
    template: str
    slots: list[SlotIn] = Field(default_factory=list)
    enabled: bool = True


class PromptUpdate(BaseModel):
    name: str | None = None
    title: str | None = None
    description: str | None = None
    template: str | None = None
    slots: list[SlotIn] | None = None
    enabled: bool | None = None


def _to_out(p: PromptDef) -> PromptOut:
    return PromptOut(
        id=p.id,
        name=p.name,
        title=p.title,
        description=p.description,
        template=p.template,
        slots=[SlotIn(**s.__dict__) for s in p.slots],
        enabled=p.enabled,
        is_builtin=p.is_builtin,
    )


def _from_slot_in(s: SlotIn) -> SlotDef:
    return SlotDef(
        name=s.name,
        type=s.type,
        required=s.required,
        default=s.default,
        options=list(s.options),
    )


def _apply_and_persist(prompts: list[PromptDef]) -> None:
    save_prompts(prompts)
    from podcodex.mcp.server import mcp

    reregister_all(mcp, prompts)


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/prompts", response_model=list[PromptOut])
async def list_prompts() -> list[PromptOut]:
    return [_to_out(p) for p in load_prompts()]


@router.post("/prompts", response_model=PromptOut, status_code=201)
async def create_prompt(payload: PromptCreate) -> PromptOut:
    prompts = load_prompts()
    if any(p.id == payload.id for p in prompts):
        raise HTTPException(409, f"Prompt id {payload.id!r} already exists.")
    new = PromptDef(
        id=payload.id,
        name=payload.name or payload.id,
        title=payload.title,
        description=payload.description,
        template=payload.template,
        slots=[_from_slot_in(s) for s in payload.slots],
        enabled=payload.enabled,
        is_builtin=False,
    )
    try:
        validate_prompt(new)
    except PromptValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    prompts.append(new)
    _apply_and_persist(prompts)
    return _to_out(new)


@router.put("/prompts/{prompt_id}", response_model=PromptOut)
async def update_prompt(prompt_id: str, payload: PromptUpdate) -> PromptOut:
    prompts = load_prompts()
    idx = next((i for i, p in enumerate(prompts) if p.id == prompt_id), -1)
    if idx < 0:
        raise HTTPException(404, f"Prompt {prompt_id!r} not found.")
    current = prompts[idx]

    if current.is_builtin:
        attempted = sorted(
            k
            for k in ("name", "title", "description")
            if getattr(payload, k) is not None
            and getattr(payload, k) != getattr(current, k)
        )
        if attempted:
            raise HTTPException(
                422,
                f"Cannot edit {attempted} on a built-in prompt. "
                "Only template, slots, and enabled are editable.",
            )

    updated = PromptDef(
        id=current.id,
        name=payload.name if payload.name is not None else current.name,
        title=payload.title if payload.title is not None else current.title,
        description=payload.description
        if payload.description is not None
        else current.description,
        template=payload.template if payload.template is not None else current.template,
        slots=[_from_slot_in(s) for s in payload.slots]
        if payload.slots is not None
        else current.slots,
        enabled=payload.enabled if payload.enabled is not None else current.enabled,
        is_builtin=current.is_builtin,
    )
    try:
        validate_prompt(updated)
    except PromptValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    prompts[idx] = updated
    _apply_and_persist(prompts)
    return _to_out(updated)


@router.delete("/prompts/{prompt_id}", status_code=204)
async def delete_prompt(prompt_id: str) -> None:
    prompts = load_prompts()
    idx = next((i for i, p in enumerate(prompts) if p.id == prompt_id), -1)
    if idx < 0:
        raise HTTPException(404, f"Prompt {prompt_id!r} not found.")
    if prompts[idx].is_builtin:
        raise HTTPException(
            409,
            "Built-in prompts cannot be deleted. Toggle them off instead.",
        )
    del prompts[idx]
    _apply_and_persist(prompts)


@router.post("/prompts/{prompt_id}/toggle", response_model=PromptOut)
async def toggle_prompt(prompt_id: str) -> PromptOut:
    prompts = load_prompts()
    idx = next((i for i, p in enumerate(prompts) if p.id == prompt_id), -1)
    if idx < 0:
        raise HTTPException(404, f"Prompt {prompt_id!r} not found.")
    prompts[idx].enabled = not prompts[idx].enabled
    _apply_and_persist(prompts)
    return _to_out(prompts[idx])
