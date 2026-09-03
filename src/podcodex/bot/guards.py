"""Command preconditions shared by the bot's mixins.

One definition per rule. The DM guard in particular was copied verbatim
into four handlers across two mixins, which is how three of them ended up
with the check and the fourth without it, and how the wording drifted from
the comment explaining why it exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import discord


async def require_guild(interaction: "discord.Interaction") -> int | None:
    """Return the guild id, or None after telling the user this needs a server.

    Per-guild state (settings, unlocked shows, announce channels) is keyed
    by guild id. discord.py exposes global commands in DMs unless a command
    opts out, and a DM has no guild: writing under ``None`` produced the
    literal key ``"None"`` in ``server_config.json``, which the next bot
    start could not parse back.

    Callers treat ``None`` as "already answered, stop here".
    """
    if interaction.guild_id is None:
        await interaction.response.send_message(
            "Use this command in a server.", ephemeral=True
        )
        return None
    return interaction.guild_id
