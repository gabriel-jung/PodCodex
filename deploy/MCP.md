# PodCodex MCP server

Expose your PodCodex index to **Claude Desktop** (or any MCP-capable client) so Claude can search your transcripts directly during a conversation.

The server does retrieval only. The client LLM reads your chunks and synthesises the answer — no API key, no billing on the PodCodex side.

Two install paths — pick whichever fits your setup:

- [Path A — desktop app (recommended)](#path-a--desktop-app-recommended) — one toggle in Settings, no JSON editing.
- [Path B — stdio (Claude Code / advanced)](#path-b--stdio-claude-code--advanced) — hand-configured, useful when you don't run the desktop app.

## Tools

Both paths expose the same four tools:

- `search(query, show?, top_k?, episode?, speaker?)` — hybrid semantic + FTS.
- `exact(query, show?, episode?, speaker?)` — literal phrase, every match (no cap).
- `list_shows()` — shows indexed with the default model + chunker.
- `get_context(show, episode, chunk_index, window?)` — expand a hit with neighbouring chunks.

Plus user-managed **prompts** that appear in Claude Desktop's `/` slash menu — see [Prompts](#prompts) below.

---

## Path A — desktop app (recommended)

PodCodex's desktop app writes the Claude Desktop config for you and points it at the bundled `podcodex-mcp` stdio binary.

### Enable

1. Start PodCodex (`make dev-no-tauri` or the `.app` bundle).
2. Go to **Settings → Claude Desktop**.
3. Flip the toggle **Enable Claude Desktop integration**.
4. Fully quit Claude Desktop (Cmd+Q on macOS, Alt+F4 elsewhere) and reopen it. Claude Desktop reads its MCP config only at startup.
5. Inside a Claude conversation, click the 🔌 plug icon — `podcodex` should appear with all four tools.

### Lifecycle

The toggle writes a stdio `mcpServers.podcodex` entry into `claude_desktop_config.json` that points at the absolute path of `.venv/bin/podcodex-mcp`. Claude Desktop spawns that binary on each startup and keeps it alive for the session.

PodCodex's desktop app does not need to be running for Claude Desktop to use the MCP — the entry persists and the subprocess reads the shared LanceDB index at `~/.local/share/podcodex/index/` directly. Toggling off removes the entry.

### Why the toggle

- No hand-editing of JSON.
- Absolute path to the binary resolved correctly for your venv.
- Merge-safe write — every other key in the config is preserved.

### WSL (PodCodex in WSL, Claude Desktop on Windows)

Supported out of the box. When the toggle runs inside WSL it:

1. Detects WSL via `/proc/version` and resolves `%APPDATA%` on the Windows host through `cmd.exe` + `wslpath`, so it writes to the Windows-side config the Windows Claude Desktop actually reads (`\\wsl$\...` is not involved on the hot path — the config lives on the Windows filesystem, the subprocess lives in WSL).
2. Emits a `wsl.exe`-wrapped entry so Windows Claude Desktop can launch the Linux binary:

   ```json
   {
     "mcpServers": {
       "podcodex": {
         "command": "wsl.exe",
         "args": ["-d", "Ubuntu", "-e", "/home/you/PodCodex/.venv/bin/podcodex-mcp"]
       }
     }
   }
   ```

   `-d <distro>` is set from `WSL_DISTRO_NAME` so Claude always lands in the distro where you installed PodCodex, even if it isn't your WSL default. The Linux path in `args[-1]` is the same binary the Linux path resolver returns.

If Win32 interop isn't available (headless WSL, unusual configs), the toggle falls back to the in-distro Linux config path — harmless, but Windows Claude Desktop won't pick it up. Check `cmd.exe /c "echo %APPDATA%"` runs from your WSL shell.

### HTTP endpoint (other clients)

When the desktop app is running, the same MCP tools are also served over HTTP at `http://127.0.0.1:18811/mcp`. That's for Claude Code and other MCP clients that support HTTP transport — Claude Desktop itself only accepts stdio entries in its config, which is why the toggle writes stdio.

---

## Path B — stdio (Claude Code / advanced)

Use this when you run Claude Code, or Claude Desktop without PodCodex's desktop app (bot-only deployments, remote VPS, CLI scripts).

### Requirements

- Python 3.12, same venv as the desktop app / bot.
- An existing index at `~/.local/share/podcodex/index/` (or wherever you point `PODCODEX_INDEX`).

### Install

From the project root:

```bash
uv sync --extra mcp
```

This installs the `mcp` SDK and the `podcodex-mcp` entry point at `.venv/bin/podcodex-mcp`.

Verify it runs (it blocks on stdin — `Ctrl+C` to quit):

```bash
.venv/bin/podcodex-mcp
```

### Register with Claude Desktop

Edit Claude Desktop's config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add (or merge into) the `mcpServers` block:

```json
{
  "mcpServers": {
    "podcodex": {
      "command": "/absolute/path/to/PodCodex/.venv/bin/podcodex-mcp",
      "env": {
        "PODCODEX_INDEX": "/Users/you/.local/share/podcodex/index"
      }
    }
  }
}
```

Replace `/absolute/path/to/PodCodex` with your actual checkout. `PODCODEX_INDEX` can be omitted — it defaults to `~/.local/share/podcodex/index`. Same env var as the desktop app and Discord bot.

Restart Claude Desktop. You should see the 🔌 plug icon next to the composer and the four tools in its menu.

### Register with Claude Code

`claude mcp add podcodex /path/to/PodCodex/.venv/bin/podcodex-mcp`

### WSL + manual register

If Path A can't reach interop and you want to register manually from WSL, edit the Windows-side config (`\\wsl.localhost\...` from Windows, or `/mnt/c/Users/<you>/AppData/Roaming/Claude/claude_desktop_config.json` from WSL) with the `wsl.exe` shape shown in Path A's WSL section.

---

## Prompts

Both paths expose user-editable **prompts**. These are canned templates that appear in Claude Desktop's `/` slash menu — unlike tools, the user invokes them explicitly.

PodCodex ships five built-ins, all of which can be toggled off:

- `/brief {topic}` — cited overview from the indexed transcripts.
- `/speaker {name}` — aggregate a speaker's stances and recurring points.
- `/quote {phrase}` — verify a quote; flags wording variance when the exact string isn't found.
- `/compare {topic}` — per-show comparison on a topic.
- `/timeline {topic}` — chronological mentions across episodes.

Custom prompts are added from **Settings → Claude Desktop → Add prompt** in the desktop app. After any prompt change, restart Claude Desktop so it re-handshakes the catalog — PodCodex itself applies changes live without a process restart.

---

## Try it

Prompts to test the tools directly:

- *"List all my podcast shows via podcodex."*
- *"Search Passion Modernistes for Bauhaus."*
- *"Find every time Anaïs literally says 'art déco'."*
- *"Quote more of that passage"* (after a hit) — triggers `get_context`.

Multi-step is fine: *"Compare how each of my shows talks about minimalism"* will make Claude call `list_shows`, loop `search` per show, and compile.

## Limits

- Only collections built with the **default model** (`bge-m3`) and **default chunker** (`semantic`) are visible. Reindex if you used a different combo.
- No write access — Claude cannot mutate your index.
- HTTP endpoint is bound to 127.0.0.1 only. Remote/VPS exposure is not supported.

## Troubleshooting

**Toggle in Settings refuses to enable**

- Check the status chip: "MCP extra not installed" means the backend needs `uv sync --extra desktop`.
- "Claude Desktop not detected" means PodCodex couldn't find the config directory. Install Claude Desktop or verify its default config directory exists.

**Claude Desktop says "server failed to start" (stdio path)**

- Run `.venv/bin/podcodex-mcp` manually; any import error prints to stderr.
- Confirm the `command` path is absolute and the binary is executable.

**Tools appear but return empty results**

- Verify `PODCODEX_INDEX` (or the default path) is a real LanceDB folder (contains `_collections.lance/` and per-show directories).
- Ask Claude to call `list_shows` first — empty list confirms the server can open the index but finds nothing indexed under the default model + chunker. Reindex via the desktop app if needed.

**Claude sees the tools but never calls them**

- The tool docstrings restrict use to podcast-related questions. Mention podcodex explicitly in your prompt (e.g. *"Using podcodex, search for..."*) to push it over the threshold. Once Claude uses them successfully, it keeps doing so.
