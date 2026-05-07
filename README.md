# rupi

Minimalistic RPC coding agent. OpenAI-compatible API only. No bloat.

## Setup

```json
// ~/.config/rupi.json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model_tag": "mistralai/mistral-small-2603"
}
```

That's it. Everything else can be overridden via CLI flags (`--base-url`, `--api-key`, `--model`).

## Modes

### Default (interactive) — `./rupi`

REPL prompt. Type your request, get streaming response. Supports `/model provider/model_id` to switch models on the fly. Exit with `Ctrl+D`, `/exit`, `/quit`, or `exit`.

### RPC — `./rupi --rpc`

JSONL protocol over stdin/stdout. Headless, designed for programmatic use. Commands are JSON lines on stdin, responses and events are JSON lines on stdout.

```json
{"type":"prompt","id":"1","message":"hello"}
{"id":"1","type":"response","command":"prompt","success":true}
{"type":"message_update","assistant_message_event":{"type":"text_delta","delta":"Hello"},"timestamp":...}
...
{"type":"agent_end","timestamp":...}
```

Events include: `generation_id`, `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, `agent_end`, `tool_execution_start`, `tool_execution_end`.

### Raw — `./rupi --raw`

Same as interactive but each SSE delta is printed as a separate JSON line to stdout. Useful for piping or debugging.

## How it works

- **Tools**: bash, read, write, edit, grep, find, ls — the agent decides when to use them
- **YOLO mode**: always active. The agent never asks for permission — it just runs commands and reports back
- **Skills**: place `.md` files with frontmatter in `~/.config/rupi/skills/` — injected into the system prompt on startup
- **Context files**: `CLAUDE.md` and `AGENTS.md` from cwd and all ancestor directories are loaded automatically
- **Compaction**: auto-triggers when context approaches the window limit. Summarizes old messages via LLM
- **Session persistence**: conversations are saved as JSONL in `~/.config/rupi_sessions/`
- **Generation ID**: `X-Generation-Id` from response headers is emitted as an early event — can be used to query OpenRouter stats
- **Cost**: usage and cost data from the API is included in the `message_end` event
