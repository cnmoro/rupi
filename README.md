# rupi

A Rust coding agent that borrows concepts from [pi](https://pi.dev/), [little-coder](https://github.com/itayinbarr/little-coder), [RTK](https://github.com/rtk-ai/rtk), and [semble](https://github.com/MinishLab/semble). Just `./rupi`. A single ~12MB binary. No npm install, no pip install, no node_modules, no Python runtime, no JVM. Download it and run it.

Works as an interactive coding agent for humans, or as a headless RPC backend for automation scripts.

Output compression: bash command output is automatically filtered to reduce token consumption. Git status/diff/log/add/commit/push, cargo test/build/check, ls, find — each has a specialized filter that strips noise and keeps only what the agent needs. Generic fallback handles ANSI stripping, deduplication, and line capping. Typical savings: 60-90% on common dev commands.

## Setup

```json
// ~/.config/rupi.json — standard OpenAI-compatible provider
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model_tag": "mistralai/mistral-small-2603"
}
```

All values can be overridden via CLI flags (`--base-url`, `--api-key`, `--model`, `--context-window`, `--timeout`, `--memory`, `--disable-yolo`, `--session`).

### Opencode Go / Zen (zero-config alternative)

Instead of finding an API key and base URL, set `opencode_api_key` in your config:

```json
// ~/.config/rupi.json
{
  "opencode_api_key": "oc_...",
  "opencode_provider": "go"
}
```

This uses opencode.ai's API directly — no separate base URL or model tag needed. Two tiers:

- **Opencode Go** (`"opencode_provider": "go"`, default) — `https://opencode.ai/zen/go/v1`. Models: deepseek-v4-flash, deepseek-v4-pro, kimi-k2.5, minimax-m2.7, glm-5, mimo-v2.5-pro.
- **Opencode Zen** (`"opencode_provider": "zen"`) — `https://opencode.ai/zen/v1`. Models: gpt-5.1-codex-max, claude-sonnet-4-6, gemini-3.1-pro, gpt-5-nano, claude-haiku-4-5.

List available models with:

```
./rupi --list-opencode-models
./rupi --list-opencode-models --opencode-provider zen
```

The list is fetched from `https://models.dev/api.json` (same source opencode uses), with a bundled snapshot as fallback.

## Modes

### Interactive — `./rupi` (default)

REPL prompt for humans. While the agent is generating, you can still type:

- **Press Enter** → queues as **follow-up**: the message is saved and processed after the current response finishes.
- **`/steer <message>`** → **interrupts immediately**: the agent receives your message right away and pivots.

Commands: `/goal <desc>`, `/model <name>`, `/compact`, `/steer <message>`, `/loop <prompt>`, `/stop`. Exit with `Ctrl+D`, `/exit`, `/quit`, or `exit`.

### Loop mode — `/loop <prompt>`

Sends the prompt, waits for the agent to finish, then sends it again — repeats forever until cancelled. Useful for:
- Continuous code review
- Ongoing monitoring tasks
- Creative generation sprints

Cancel with **double-Esc** (press Esc twice in rapid succession), or type `/stop`.

### RPC — `./rupi --rpc`

JSONL protocol over stdin/stdout. Designed for programmatic use — send JSON commands on stdin, receive events on stdout.

```json
{"type":"prompt","id":"1","message":"hello"}
{"id":"1","type":"response","command":"prompt","success":true}
{"type":"message_update","assistant_message_event":{"type":"text_delta","delta":"Hello"},"timestamp":...}
...
```

Events: `generation_id`, `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, `agent_end`, `tool_execution_start`, `tool_execution_end`.

**Steer / follow-up in RPC**: add `"streamingBehavior"` to the prompt command:

```json
{"type":"prompt","id":"2","message":"fix the indentation","streamingBehavior":"steer"}
{"type":"prompt","id":"3","message":"add error handling","streamingBehavior":"followUp"}
```

- `"steer"` — interrupts the current generation immediately, like `/steer` in interactive.
- `"followUp"` — queues the message; the agent processes it after the current turn finishes.

**Loop mode in RPC** — start and stop with `SetLoop` and `StopLoop`:

```json
{"type":"set_loop","id":"1","message":"review this file"}
{"type":"stop_loop","id":"2"}
```

### Raw — `./rupi --raw`

Same event stream as RPC but reads user input interactively. Useful for debugging or piping. Supports the same interactive steer/follow-up behavior: **Enter queues as follow-up**, **`/steer <message>` interrupts**.

## How it works

- **Tools**: bash, read, write, edit, grep, find, ls, search_code — the agent decides when to use them. `search_code` uses a local Model2Vec semantic code search model (potion-code-16M) to find code by natural language description — no grep patterns needed. YOLO mode (default): no approval needed. Add `--disable-yolo` to require user confirmation per execution.
- **Write guard**: `write` refuses if the file already exists, returning an error with the exact `edit` call-shape. This prevents accidental whole-file rewrites of existing code. Use `edit` for any change to an existing file.
- **Multi-edit**: `edit` accepts an `edits` array for batch changes in a single call. Each edit's `old_text` is matched against the **original** file content (not after other edits). Edits must not overlap.
- **Output parser**: when the model emits tool calls inside text (fenced ` ```tool ``` blocks, `<tool_call>` tags, or bare JSON), the parser extracts and executes them as if they were native tool calls.
- **Quality monitor**: detects empty responses, hallucinated tool names, and repeated identical tool calls (loops). Queues correction messages to nudge the model back on track (capped at 2 per session to avoid correction loops).
- **Skills**: place `.md` files in `~/.config/rupi/skills/` — injected into the system prompt on startup
- **Context files**: `CLAUDE.md` and `AGENTS.md` from cwd and ancestor directories are loaded automatically
- **Compaction**: two-layer context management. First, **snip** truncates long tool-role messages older than the last 6 turns (rule-based, no API cost). Then, if still over threshold, **auto-compact** calls the LLM to summarize old messages. Set with `--context-window` (default 128000, fires at `window - 16384` tokens).
- **No hard limits**: the agent runs indefinitely until the task is done. When context approaches the window limit, snip + auto-compact keeps the agent going. Optionally set `--timeout <secs>` to cap execution time.
- **Steer / follow-up**: type while the agent generates — normal Enter queues as follow-up (processed after the current turn). Use `/steer <message>` to interrupt immediately. In RPC mode, set `"streamingBehavior": "steer"` or `"followUp"` on the prompt command.
- **Memory**: add `--memory` to persist key facts across sessions. The agent reads/writes `~/.config/rupi/MEMORY.md` — reads on startup, overwrites with bullet points during execution.
- **Session persistence**: conversations saved as JSONL in `~/.config/rupi_sessions/` with UUID filenames
- **Session resumption**: use `--session <id>` to resume a previous conversation from where you left off. The agent remembers all prior messages. Works in interactive, raw, and RPC modes.
- **Error reporting**: `message_end` includes `stop_reason` (`"stop"`, `"error"`, `"tool_calls"`, `"timeout"`) and error text in `content` when applicable.
- **Generation ID**: `X-Generation-Id` from response headers emitted as an early event
- **Cost**: usage and cost data from the API included in the `message_end` event

## Calling via code (Python / Node.js / Java)

### Python

```python
import subprocess, json

proc = subprocess.Popen(
    ["./rupi", "--rpc"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

def send(cmd):
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()

def read_until(target_type, timeout=60):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        obj = json.loads(line)
        if obj.get("type") == target_type:
            return obj
    return None

send({"type": "prompt", "id": "1", "message": "What is 2+2?"})
read_until("response")  # prompt accepted
while (ev := read_until("agent_end", timeout=30)) is None:
    ev = read_until("message_update")
    if ev and "delta" in ev.get("assistant_message_event", {}):
        print(ev["assistant_message_event"]["delta"], end="", flush=True)
```

### Node.js

```javascript
import { spawn } from "child_process";

const rupi = spawn("./rupi", ["--rpc"]);
const pending = new Map();
let reqId = 0;

function send(cmd) {
  const id = `req_${++reqId}`;
  rupi.stdin.write(JSON.stringify({ ...cmd, id }) + "\n");
  return new Promise((resolve) => pending.set(id, resolve));
}

function readLoop() {
  let buffer = "";
  rupi.stdout.on("data", (chunk) => {
    buffer += chunk.toString();
    for (const line of buffer.split("\n").slice(0, -1)) {
      if (!line.trim()) continue;
      const msg = JSON.parse(line);
      if (msg.type === "response" && msg.id && pending.has(msg.id)) {
        pending.get(msg.id)(msg);
        pending.delete(msg.id);
      }
      if (msg.type === "message_update") {
        process.stdout.write(msg.assistant_message_event?.delta ?? "");
      }
      if (msg.type === "agent_end") process.exit();
    }
    buffer = buffer.split("\n").pop();
  });
}

readLoop();
await send({ type: "prompt", message: "Say hello" });
```

### Java (Gson)

```java
import java.io.*;
import com.google.gson.*;

public class RupiClient {
    public static void main(String[] args) throws Exception {
        Process rupi = new ProcessBuilder("./rupi", "--rpc").start();
        var stdin = new BufferedWriter(new OutputStreamWriter(rupi.getOutputStream()));
        var stdout = new BufferedReader(new InputStreamReader(rupi.getInputStream()));
        var gson = new Gson();

        var cmd = new JsonObject();
        cmd.addProperty("type", "prompt");
        cmd.addProperty("id", "1");
        cmd.addProperty("message", "Say hello");
        stdin.write(gson.toJson(cmd) + "\n");
        stdin.flush();

        String line;
        while ((line = stdout.readLine()) != null) {
            var event = JsonParser.parseString(line).getAsJsonObject();
            if (event.get("type").getAsString().equals("agent_end")) break;
            if (event.get("type").getAsString().equals("message_update")) {
                var delta = event.getAsJsonObject("assistant_message_event");
                System.out.print(delta.get("delta").getAsString());
                System.out.flush();
            }
        }
        rupi.destroy();
    }
}
```
