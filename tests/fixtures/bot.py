"""A real ``PodCodexBot`` over a tmp index, for handler and access tests.

Hand-built bots (``PodCodexBot.__new__`` plus whichever private attributes a
test needed) broke on every rename and, where they stubbed the label
resolver or the access rule, tested a copy instead of the bot. This builds
the real object; only Discord is absent. Seed its index with
``tests/fixtures/index.add_show``.
"""

from __future__ import annotations

from pathlib import Path


def make_bot(tmp_path: Path, server_cfg: dict | None = None):
    """``(bot, saves)``: the bot over the process-wide index (point it at a
    tmp dir with the ``isolated_index`` fixture first, so the test and the bot
    share one store) with its server config at ``tmp_path /
    "server_config.json"``, and a list that grows by one on every settings
    save (the real save still writes the file)."""
    from podcodex.bot.bot import BotConfig, PodCodexBot

    bot = PodCodexBot(BotConfig(), server_config_path=tmp_path / "server_config.json")
    bot._server_cfg.update(server_cfg or {})
    bot._reload_shows()

    saves: list[bool] = []
    real_save = bot._save_server_config

    def _recorded_save() -> None:
        saves.append(True)
        real_save()

    bot._save_server_config = _recorded_save
    return bot, saves
