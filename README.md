# rupi

Minimalistic RPC coding agent. Designed for programmatic use — call it from Python, Node.js, Java, or any language that can spawn a subprocess and read JSON from stdout. Optionally usable by humans via the interactive REPL.

## Setup

```json
// ~/.config/rupi.json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model_tag": "mistralai/mistral-small-2603"
}
```

All values can be overridden via CLI flags (`--base-url`, `--api-key`, `--model`, `--context-window`, `--timeout`).

## Modes

### RPC — `./rupi --rpc` (recommended for programmatic use)

JSONL protocol over stdin/stdout. Send a JSON command on stdin, receive responses and streaming events on stdout.

```json
{"type":"prompt","id":"1","message":"hello"}
{"id":"1","type":"response","command":"prompt","success":true}
{"type":"message_update","assistant_message_event":{"type":"text_delta","delta":"Hello"},"timestamp":...}
...
{"type":"agent_end","timestamp":...}
```

Events: `generation_id`, `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, `agent_end`, `tool_execution_start`, `tool_execution_end`.

### Raw — `./rupi --raw`

Same event stream as RPC but reads user input from an interactive prompt instead of stdin JSON. Useful for debugging or piping to other tools.

### Interactive — `./rupi` (default)

REPL prompt for humans. Supports `/model <name>` to switch models. Supports `/goal <description>` for durable sessions — the agent loops until an internal LLM verification confirms the goal is met. Events are held (no `agent_end`) until completion. Exit with `Ctrl+D`, `/exit`, `/quit`, or `exit`.

## Headless usage

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

### Java

```java
import java.io.*;
import java.util.concurrent.*;

public class RupiClient {
    public static void main(String[] args) throws Exception {
        Process rupi = new ProcessBuilder("./rupi", "--rpc").start();
        var stdin = new BufferedWriter(new OutputStreamWriter(rupi.getOutputStream()));
        var stdout = new BufferedReader(new InputStreamReader(rupi.getInputStream()));

        stdin.write("{\"type\":\"prompt\",\"id\":\"1\",\"message\":\"Say hello\"}\n");
        stdin.flush();

        String line;
        while ((line = stdout.readLine()) != null) {
            if (line.contains("\"agent_end\"")) break;
            if (line.contains("\"text_delta\"")) {
                System.out.print(line.replaceAll(".*\"delta\":\"([^\"]+)\".*", "$1"));
                System.out.flush();
            }
        }
        rupi.destroy();
    }
}
```

## How it works

- **Tools**: bash, read, write, edit, grep, find, ls — the agent decides when to use them. YOLO mode (default): no approval needed. Add `--disable-yolo` to require user confirmation per execution.
- **Skills**: place `.md` files in `~/.config/rupi/skills/` — injected into the system prompt on startup
- **Context files**: `CLAUDE.md` and `AGENTS.md` from cwd and ancestor directories are loaded automatically
- **Compaction**: auto-triggers when context approaches the window. Set with `--context-window` (default 128000, fires at `window - 16384` tokens)
- **Session persistence**: conversations saved as JSONL in `~/.config/rupi_sessions/`
- **Generation ID**: `X-Generation-Id` from response headers emitted as an early event
- **Cost**: usage and cost data from the API included in the `message_end` event
