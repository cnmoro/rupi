# rupi

A Rust clone-ish of [pi](https://pi.dev/). Just `./rupi`. A single 7MB binary. No npm install, no pip install, no node_modules, no Python runtime, no JVM. Download it and run it.

Works as an interactive coding agent for humans, or as a headless RPC backend for automation scripts.

## Setup

```json
// ~/.config/rupi.json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model_tag": "mistralai/mistral-small-2603"
}
```

All values can be overridden via CLI flags (`--base-url`, `--api-key`, `--model`, `--context-window`, `--timeout`, `--memory`, `--disable-yolo`).

## Modes

### Interactive — `./rupi` (default)

REPL prompt for humans. Supports `/goal <desc>` (durable sessions — loops until goal verified), `/model <name>` (switch models), `/compact` (trigger compaction). Multi-line paste supported (lines arriving within 100ms are joined). Exit with `Ctrl+D`, `/exit`, `/quit`, or `exit`.

### RPC — `./rupi --rpc`

JSONL protocol over stdin/stdout. Designed for programmatic use — send JSON commands on stdin, receive events on stdout.

```json
{"type":"prompt","id":"1","message":"hello"}
{"id":"1","type":"response","command":"prompt","success":true}
{"type":"message_update","assistant_message_event":{"type":"text_delta","delta":"Hello"},"timestamp":...}
...
```

Events: `generation_id`, `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, `agent_end`, `tool_execution_start`, `tool_execution_end`.

### Raw — `./rupi --raw`

Same event stream as RPC but reads user input interactively. Useful for debugging or piping.

## How it works

- **Tools**: bash, read, write, edit, grep, find, ls — the agent decides when to use them. YOLO mode (default): no approval needed. Add `--disable-yolo` to require user confirmation per execution.
- **Skills**: place `.md` files in `~/.config/rupi/skills/` — injected into the system prompt on startup
- **Context files**: `CLAUDE.md` and `AGENTS.md` from cwd and ancestor directories are loaded automatically
- **Compaction**: auto-triggers when context approaches the window. Set with `--context-window` (default 128000, fires at `window - 16384` tokens)
- **No hard limits**: the agent runs indefinitely until the task is done. When context approaches the window limit, auto-compaction summarizes old messages and the agent keeps going. Optionally set `--timeout <secs>` to cap execution time.
- **Memory**: add `--memory` to persist key facts across sessions. The agent reads/writes `~/.config/rupi/MEMORY.md` — reads on startup, overwrites with bullet points during execution.
- **Session persistence**: conversations saved as JSONL in `~/.config/rupi_sessions/`
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
