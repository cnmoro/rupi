# rupi

A Rust coding agent that borrows concepts from [pi](https://pi.dev/), [little-coder](https://github.com/itayinbarr/little-coder), [RTK](https://github.com/rtk-ai/rtk), and [semble](https://github.com/MinishLab/semble). A single ~28MB binary. No npm install, no pip install, no node_modules, no Python runtime, no JVM. Download it and run it.

**This project targets headless use.** rupi is built to be driven by another program over a JSONL protocol on stdin and stdout, not typed at by a person in a terminal. Every design decision answers to that: one `agent_end` per prompt so a caller always knows when a turn is over, a session id the caller chooses and can resume, events for every step of a turn, and refusals that are reported as errors instead of silently changing what the agent does.

An interactive terminal mode exists, and it works, but it is there to debug the engine by hand. It is not the audience. See **Modes** below, and **Calling via code** for a Python, Node.js or Java client.

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

All values can be overridden via CLI flags (`--base-url`, `--api-key`, `--model`, `--context-window`, `--timeout`, `--memory`, `--disable-yolo`, `--session`, `--sessions-dir`, `--bash-timeout-max`, `--bash-timeout-default`, `--list-sessions`), and each also reads a `RUPI_*` environment variable (`RUPI_BASE_URL`, `RUPI_API_KEY`, `RUPI_MODEL`, `RUPI_SESSIONS_DIR`, `RUPI_BASH_TIMEOUT_MAX`, `RUPI_BASH_TIMEOUT_DEFAULT`).

The config file is optional when `--base-url`, `--api-key` and `--model` are all supplied — useful when running rupi somewhere without a writable `$HOME`.

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

Default model is `deepseek-v4-flash` for both providers. Override with `--model` or set `model_tag` in config.

List available models with:

```
./rupi --list-opencode-models
./rupi --list-opencode-models --opencode-provider zen
```

The list is fetched from `https://models.dev/api.json` (same source opencode uses), with a bundled snapshot as fallback.

## Modes

RPC is the mode the project is built around. One-shot mode answers a single prompt on stdout. Raw mode is the same event stream with keyboard input. Interactive mode is a convenience for driving the engine by hand. For working code that speaks the protocol, see **Calling via code** below.

### RPC — `./rupi --rpc` (the mode this project is for)

JSONL protocol over stdin and stdout. Send one JSON command per line on stdin, read one JSON event per line on stdout. This is how rupi is meant to run: a parent program owns the session, decides what to prompt, and reacts to the events.

```json
{"type":"prompt","id":"1","message":"hello"}
{"id":"1","type":"response","command":"prompt","success":true}
{"type":"message_update","assistant_message_event":{"type":"text_delta","delta":"Hello"},"timestamp":...}
...
```

Events: `generation_id`, `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, `agent_end`, `tool_execution_start`, `tool_execution_end`.

Commands: `ping`, `prompt`, `steer`, `follow_up`, `abort`, `new_session`, `get_state`, `set_model`, `cycle_model`, `get_available_models`, `set_thinking_level`, `cycle_thinking_level`, `compact`, `set_auto_compaction`, `get_messages`, `list_sessions`, `set_loop`, `stop_loop`. Every command takes an optional `id`, and the `response` line for that command carries it back with `success`. A command that is refused answers with `success: false` and a reason.

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

A steer redirects the loop rather than ending it: the round it interrupts is cancelled, the new instruction opens the next round, and the loop keeps cycling. A loop that stops for any other reason is cleared, so it is never reported as running when it will not run again.

While a loop runs, a plain `prompt` is refused with an error that says to send `stop_loop` first. Such a prompt used to be swallowed into the loop's own turn, where it produced no `agent_end` of its own and left a caller waiting for one forever. A `prompt` that names a `streamingBehavior` is a steer or a follow-up, and those still go through: interjecting in a running turn is what they are for.

### One shot — `./rupi -p "<task>"`

Runs one prompt to completion, prints the agent's final answer to stdout, and exits. Nothing else reaches stdout, so the answer can be read straight out of a pipe:

```sh
./rupi -p "list every file that calls run_tool_loop, and say what each call does"
```

Exit code 0 with the answer on stdout, or a non-zero code with the reason on stderr: 1 for a run that produced no answer, 2 for an empty prompt, 3 when the live-agent limit refused the start. A run that fails never prints an answer, so a caller cannot mistake a failure for an empty report, and a turn that did not finish — a stream that went silent, one the provider cut off, a token limit, a cancellation — fails rather than printing the fragment it had. Progress and errors from the agent and its tools go to stderr.

A prompt given this way starts a fresh session unless `--session <id>` names one.

This is also how one agent starts another. See **Subagents** below.

### Raw — `./rupi --raw`

Same event stream as RPC but reads user input interactively. Useful for debugging or piping. Supports the same interactive steer/follow-up behavior: **Enter queues as follow-up**, **`/steer <message>` interrupts**.

### Interactive — `./rupi` (the default when no mode flag is given, for debugging)

A REPL for a person at a keyboard. It runs the same engine as RPC mode, so it is a good way to watch the agent work by hand. It is not what the project optimizes for. While the agent is generating, you can still type:

- **Press Enter** → queues as **follow-up**: the message is saved and processed after the current response finishes.
- **`/steer <message>`** → **interrupts immediately**: the agent receives your message right away and pivots.

Commands: `/goal <desc>`, `/model <name>`, `/compact`, `/steer <message>`, `/loop <prompt>`, `/stop`, `/session`. With `--disable-yolo`, typing while the agent works is unavailable: the approval prompt needs the terminal, and two readers on one terminal means the prompt never receives your answer. Exit with `Ctrl+D`, `/exit`, `/quit`, or `exit`. Use `/session` to show the current session ID.

### Loop mode — `set_loop` over RPC, `/loop <prompt>` interactively

Sends the prompt, waits for the agent to finish, then sends it again — repeats forever until cancelled. Useful for:
- Continuous code review
- Ongoing monitoring tasks
- Creative generation sprints

Over RPC, start it with `set_loop` and end it with `stop_loop`. Interactively, cancel with **double-Esc** (press Esc twice in rapid succession), or type `/stop`.

### Goal mode — `/goal <description>` (interactive and raw modes only)

Sets a durable objective and drives the agent in rounds until the objective is met. Each round injects a `<goal_round>` block that carries the objective, the round number, the round budget, and an instruction to treat the workspace and the tool results as authoritative rather than earlier narration.

The agent ends the run itself with the `goal` tool:

- `{"operation": "complete", "round": N}` — the whole objective is achieved.
- `{"operation": "block", "round": N, "reason": "..."}` — the agent cannot proceed.
- `{"operation": "read"}` — report the objective, the open round, and the status.

A decision is accepted only from inside the round the driver opened. `N` must match that round, so the agent cannot declare the goal done from a stray turn. A model that never calls the tool falls back to an out-of-band check, and the run stops after 5 rounds either way. However the driver stops, the goal is finished — an undecided goal does not keep driving later, unrelated prompts.

`/goal` with no argument shows the objective and its status.

RPC has no goal command. A headless caller drives rounds itself — it owns the prompt loop already — or uses `set_loop`, which repeats one prompt until it is stopped. The goal tool the model calls to end a run is only wired to the two keyboard modes.

## How it works

- **Tools**: bash, read, write, edit, grep, find, ls, search_code, todo_write, goal — the agent decides when to use them. A command that times out or is cancelled has its descendants collected before the kill, so a process that left the group with `setsid`, `nohup` or a daemonizing double fork is still stopped. Every process started by one command also inherits a marker in its environment, unique to that command, so a descendant whose own parent already exited is found by the marker wherever the kernel reparented it. Each collected process carries its start time, checked again immediately before the signal, so a number recycled in between is never signalled — killing an unrelated process would be far worse than missing a descendant. A command that clears its own environment before it detaches escapes the marker, which is deliberate evasion rather than a case the walk can cover. The harness writes its own notices about a command — "timed out", "cancelled" — inside a fence that carries a token generated once per process and never given to any child. The phrase that opens a fence is also rewritten wherever it appears in command output, because a command that had seen one real notice could otherwise reprint its token and report a timeout for a command that in fact succeeded. A kill reads the process table twice, which costs about 70ms per thousand live processes on the machine, and only on a timeout or a cancellation. `edit` and `write` also accept `then_run` (see action fusion below). A bash command times out after 30s unless the model asks for longer, capped at 120s; `--bash-timeout-default` and `--bash-timeout-max` move both, and the tool schema tells the model what the current limits are. `search_code` uses a local Model2Vec semantic code search model (potion-code-16M) to find code by natural language description — no grep patterns needed. Its index decides in two steps. The modification time, length, status change time and inode decide whether to LOOK at a file — mtime and length alone compared two revisions equal whenever a checkout, `rsync -a` or `cp -p` restored the timestamp, and search then served content that no longer existed. A content digest then decides whether to re-embed it, because status change time also moves on `chmod`, `chown` and `touch`, and re-embedding a repository whose bytes never changed is a worse problem than the one the stricter check fixes. Windows has no status change time, so a same-size edit made in place with the timestamp restored is not detected there. YOLO mode (default): no approval needed. Add `--disable-yolo` to require user confirmation per execution.
- **Subagents**: the system prompt tells the agent how to start other agents, with the bash tool and the path of the running binary: `<binary> -p "<task>"`. Each one is a separate process with an empty context, its own session file, and the same tools, so the task string has to carry everything it needs. Several can run at once with `&` and `wait`, each redirected to its own file. Every process the bash tool starts carries `RUPI_SUBAGENT=1` in its environment, and a starting agent also looks for a copy of itself among its parent processes, so a command that scrubs its environment with `env -i` is recognised anyway. A subagent is told that it is one and is not given these instructions. `--max-agents` (default 12, also `RUPI_MAX_AGENTS`) is the hard bound: a start beyond that many live processes of the same binary is refused with exit code 3, so a model that loops on starting agents cannot fan out until the machine gives up. The limit is stated in the system prompt, so the agent knows it rather than discovering it as a failure. The bash ceiling applies to the call that starts one, so a long subagent run either needs a larger `--bash-timeout-max` or has to be started in the background and collected on a later turn. Ask for a subagent when parts of the work are genuinely independent — a review of several areas at once, or a search that would fill the parent's context. Sequential work is cheaper in one agent.
- **Action fusion**: `edit` and `write` take an optional `then_run: {"command": "...", "timeout": N}`. The tool applies the change and runs that command in the same call, returning one observation. A file change is nearly always followed by the command that checks it, and splitting that pair across two turns costs a whole round trip for a decision the model has already made. The command is skipped if the change fails, and a non-zero exit is reported without undoing the change. Under `--disable-yolo` the fused command gets its own approval prompt, labelled as `bash` — declining it applies the change, drops only the command, and tells the agent plainly that the check was refused. Approval prompts show their arguments in full and never elide the middle, because a prompt that hides part of a command is not a gate.
- **Write guard**: `write` refuses if the file already exists, returning an error with the exact `edit` call-shape. This prevents accidental whole-file rewrites of existing code. Use `edit` for any change to an existing file.
- **Multi-edit**: `edit` accepts an `edits` array for batch changes in a single call. A malformed entry rejects the whole call and applies nothing, rather than quietly dropping that entry and reporting success for a half-applied change. Each edit's `old_text` is matched against the **original** file content (not after other edits). Edits must not overlap.
- **Secret redaction**: a compaction checkpoint is scanned for credential-shaped values before it is stored, and they are replaced with `[redacted]`. The summarizer is told to preserve commands verbatim, which is right for resuming work and wrong for credentials: compaction deletes the region that held the secret, so an unredacted checkpoint would carry it into every later request and onto disk. Matching is on the shape of a value, never a keyword, so prose about tokens or passwords is untouched and the command itself stays readable.
- **Harness notices are fenced**: a timeout or a cancellation is reported inside explicit markers, so a command that prints the same words cannot be mistaken for the harness reporting what it did.
- **Fail-open output compression**: the per-command filters drop noise, and a check afterwards makes sure they did not drop signal. If the raw output carried a failure marker and the compressed output no longer does, the original is returned unchanged.
- **Quoted tool calls are not run**: when a provider does not surface a native tool call, rupi falls back to reading one out of the model's text. A block that covers only part of a message is treated as something the model quoted or explained, not something it asked to run — otherwise content the agent merely read could become a command it executes.
- **Output parser**: the fallback runs only for a provider that has never surfaced a tool call natively in this session. Once one arrives natively, a block in the text is a quotation, and the fallback stays off for the rest of the session. When the model emits tool calls inside text (fenced ` ```tool ``` blocks, `<tool_call>` tags, or bare JSON), the parser extracts and executes them as if they were native tool calls. A model that means to call a tool stops there, so the call must end the message: a block followed by more than a short closing remark is treated as a quotation and is not run. This matters because a model often quotes a block out of a file it just read, and running that turns content the agent merely read into a command it executes. The scan makes one pass over both marker styles, so a nested block is never counted or executed twice, and an unterminated block earns no credit at all.
- **Quality monitor**: detects empty responses, hallucinated tool names, and repeated identical tool calls. Empty and hallucinated responses queue a correction, capped at 2 per session to avoid correction loops.
- **Repeat ladder**: a run of identical tool calls earns a reminder at 3, 5, and 8 consecutive calls. The first is a gentle note. The later two name the tool, the run length, and the arguments, capped at 500 characters. Arguments are compared with object keys sorted, so a model cannot hide a loop by shuffling JSON keys. The ladder is exempt from the correction cap, because it fires at most three times per run and cannot loop.
- **Task anchor**: the request that started the session is held as session state and re-stated verbatim below every compaction checkpoint. Compaction deletes everything outside the recent tail, so after a few hundred tool calls the original prompt used to survive only if the summarizer chose to restate it. The anchor makes that structural instead. It is also re-stated at the tail after 40 tool results with no user message. Both positions are free: a compaction has already invalidated the cached prefix, and a tail append never touches it. The anchor is persisted below the compaction record, so a resumed session recovers the exact request.

  A new prompt sent while the agent is idle replaces the anchor, because it starts a new task. A `/steer` or a follow-up sent while the agent works joins the anchor instead, because it refines the task in flight. The stored anchor keeps the head and the tail when it grows, so the request that started the work and the most recent instruction both survive.
- **Todo list**: `todo_write` records a plan. Every call replaces the whole list, and at most one task can be `in_progress`. The newest copy sits at the tail of the context, where attention is strongest, and it rides along with the anchor reminder.
- **Spill**: a tool result larger than 4000 characters is written in full to `<sessions-dir>/spill/<session-id>/` before the stored copy is truncated. The truncation notice carries the path, so the agent can `read` or `grep` the part that was cut. The context stays small and no output is destroyed. A storage failure is not fatal: the plain truncated result is kept.

  Artifacts are content addressed by digest, so an agent that runs the same failing command ten times archives it once. An existing archive is compared byte for byte before it is reused, and anything that does not match exactly is written to a fresh address instead. Files are created with mode `0600` and directories with `0700`, set at creation rather than afterwards, and a symlink at an artifact path is refused — tool output carries environment dumps and credentials, and these files outlive the session.
- **Line-aligned truncation**: the copy kept in the conversation is cut on whole lines, so a stack trace or a compiler error never ends in a fragment that reads as a shorter message than it is. Output with no usable line boundary inside the budget, such as minified JSON, falls back to a character-aligned cut rather than giving up most of the budget.
- **Stalled and cut streams**: a provider that accepts a request and then goes quiet ends the turn as a timeout, never as a clean stop, and the text it already sent is kept. A stream that ends with no `[DONE]` and no finish reason was cut off, and says so: the turn ends as `truncated`, because a dropped connection otherwise looked exactly like a finished answer and handed the caller half a sentence as the result. `--stream-idle-timeout` sets the silence a stream may hold; the HTTP client enforces it between bytes, and the session enforces twice that between events, for a provider that sends keep-alive bytes but no content. A stall reported as an error used to throw away the answer the user could still see on screen.
- **Reset during work**: `reset` waits for the running generation to stop before it clears the conversation, but that wait is bounded at 30 seconds, because only `bash` checks for cancellation once it has started. A generation that outlives the wait is fenced off: the session counts conversations, and a write that names the old one is dropped rather than appended to the new conversation and its session file.
- **Skills**: place `.md` files in `~/.config/rupi/skills/` — injected into the system prompt on startup
- **Context files**: `CLAUDE.md` and `AGENTS.md` from cwd and ancestor directories are loaded automatically
- **Prompt caching**: rupi treats the request prefix as something to protect, because re-prefilling a full context window on every turn is the single largest avoidable cost in a long run.
  - The system prompt is built once per session and carries the date, not the clock. It sits in the first message, so a timestamp that ticks would make every turn a cache miss for the whole conversation. Run `date` when a task needs the exact time.
  - Tool schemas are serialized once, when the provider is created.
  - The conversation is append-only. Nothing already sent is ever edited, so an automatic prefix cache (vLLM, SGLang, OpenAI, DeepSeek) keeps matching. A compaction is the only thing that breaks the prefix, and it replaces the head by design.
  - Two `cache_control` breakpoints are sent for servers that need explicit ones: a static breakpoint on the system message, and a moving breakpoint on the last message. The moving one is the important half — marking only the system message would leave the entire conversation, which is nearly all of the tokens, re-prefilled every turn. Because the history is append-only, this turn's breakpoint is the next turn's cache hit. Servers with automatic caching ignore the field.
  - The compaction call reuses the same prefix; see **Cache-aligned summarization** below.
- **Compaction**: checked before each model request, including successive tool turns. When the context passes `window - 16384` tokens, **auto-compact** calls the LLM to summarize the old messages. A **snip** pass truncates long tool-role messages older than the last 6 turns, which costs nothing and lets more messages fit in the retained tail. The snip is applied to the tail that compaction stores, never to the live conversation: a run that snipped and then declined to compact would rewrite message bodies the provider had already cached and gain nothing for it. Set the window with `--context-window` (default 128000). A compaction is recorded in the session file, and resuming honours it: everything the summary replaced is left out, so a compacted session does not reopen over budget and immediately compact again.
- **Cache-aligned summarization**: the summarizer call replays the conversation's own system prompt, its tool schemas, and the region verbatim, then appends the compaction instruction as the final user message. That makes the call a genuine prefix of the last routed request, so the provider serves it from its KV cache instead of re-prefilling the whole span. The region replayed is the text from **before** the snip pass, because a snipped body no longer matches what the provider cached. The kept tail still gets the snipped copy. If an endpoint rejects the tool schemas on a non-streaming call, rupi retries without them and the summary still lands.
- **Checkpoint validation**: a summarizer response is checked before it is stored. Headings are matched on what they say, not how they are typeset, so `**Current Work**` and `## 1. Current Work` both count, and a fenced code block inside a section does not end it. A response that is empty, too short, missing a required section, carrying a section with no substance, or simply the instruction echoed back is refused. Whatever comes back replaces the whole region, so a refusal, a filtered stub, or a stream cut off at the token limit would delete the context silently and leave the agent continuing from nothing. rupi then retries once, and if that also fails it builds a checkpoint mechanically from the messages — the files touched, the commands run, the last exchanges — which invents nothing and always works.
- **Checkpoint framing**: the summary is stored as a message that states it is established background, wrapped in `<compacted-summary>` tags. The instruction tells the summarizer to merge a prior checkpoint rather than re-condense it, so repeated compactions consolidate instead of decaying.
- **Tool pairing**: the compaction cut is snapped to a boundary where neither half splits an assistant tool call from its results. A split pair is a hard 400 from every OpenAI-compatible endpoint, on the summarizer request or on the next turn. Loading a transcript repairs the same shape from the other direction. The list is rebuilt rather than patched: an assistant keeps only the calls that were answered, each answer follows it immediately, and a result that answers nothing is carried after the batch as user text rather than left between a call and its answer. A crash, or a reset that gave up waiting for a wedged tool, can land in the middle of one batch of calls.
- **No hard limits**: the agent runs indefinitely until the task is done. When context approaches the window limit, snip + auto-compact keeps the agent going. Optionally set `--timeout <secs>` to cap execution time.
- **Steer / follow-up**: type while the agent generates — normal Enter queues as follow-up (processed after the current turn). Use `/steer <message>` to interrupt immediately. In RPC mode, set `"streamingBehavior": "steer"` or `"followUp"` on the prompt command.
- **Memory**: add `--memory` to persist key facts across sessions. The agent reads/writes `~/.config/rupi/MEMORY.md` — reads on startup, overwrites with bullet points during execution.
- **Resetting sessions**: RPC `new_session` cancels active generation and waits for its tools to finish before clearing history and opening a new transcript.
- **Compaction is one self-describing record**: it states how many messages its summary replaced, and carries the checkpoint body and the task anchor with it. Replay used to clear everything above the record, which was only true while compaction was the only writer — a message appended by another task during the summarization round trip landed above it and was deleted on the next resume. Writing the checkpoint and anchor as separate lines had a subtler cost: the rebuild puts the anchor between the checkpoint and the retained tail, but a separate line could only land after the tail, so replay reconstructed a different order and the next compaction's index, measured against one order and applied to the other, deleted messages no summary had touched. One record fixes the order and makes the whole compaction a single atomic append. A record written before these fields keeps the old meaning, so an existing transcript still resumes.
- **Session persistence**: conversations saved as JSONL in `~/.config/rupi_sessions/` with UUID filenames. Spill artifacts live beside them under `spill/<session-id>/`, and are not pruned automatically. `--sessions-dir <path>` puts them somewhere else — one directory per tenant, or a path an embedding process controls. The transcript is recreated if something deletes it mid-run.
- **Session resumption**: use `--session <id>` to resume a previous conversation from where you left off. The agent remembers all prior messages (up to the last compaction). An unknown id *starts* that session rather than falling back to a random one, so a caller that owns the id gets a predictable transcript path from the first turn. Works in interactive, raw, and RPC modes.
- **Error reporting**: `message_end` includes `stop_reason` (`"stop"`, `"error"`, `"tool_calls"`, `"timeout"`) and error text in `content` when applicable.
- **Generation ID**: `X-Generation-Id` from response headers emitted as an early event, and written to the session file so per-generation cost can still be reconciled with the provider after the stream is gone
- **Cost**: usage and cost data from the API included in the `message_end` event

## Calling via code (Python / Node.js / Java)

This is the intended way to use rupi. Each client below starts the binary in RPC mode, writes one command per line, and reads events until `agent_end`.

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

## Development checks

The lockfile is committed, and CI uses locked dependency resolution. Run:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
make dist
```

The default suite includes deterministic HTTP streaming, retry, cancellation,
compaction, subprocess, and search-cache regressions. Live-provider/model tests
are explicitly ignored by default. Run them separately in an environment with
test credentials and permission to download the search model:

```sh
cargo test --locked --test e2e_test -- --ignored --test-threads=1
```

`make dist` builds the release binary before packaging it and takes the version
from `Cargo.toml`. Published releases require both the build matrix and the test,
formatting, and Clippy checks to succeed.

Live tests accept `RUPI_BASE_URL`, `RUPI_API_KEY`, and `RUPI_MODEL` directly from
the environment, before consulting config files. They keep session transcripts
in a temporary directory. The API reachability test makes an actual streaming
request and verifies Unicode output and token usage.

For a live test of the executable itself, including file creation, read-back,
process restart/session resumption, and reset:

```sh
cargo build --locked
python tests/live_rpc_test.py target/debug/rupi
```

This check uses the same environment variables, creates an isolated temporary
workspace, and fails on API errors, missing usage, incorrect file contents, or
failed session recall. It does not print credentials.
