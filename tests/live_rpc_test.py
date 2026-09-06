#!/usr/bin/env python3
"""Opt-in live CLI validation using RUPI_BASE_URL, RUPI_API_KEY, RUPI_MODEL.

Run: python tests/live_rpc_test.py [path/to/rupi]
Uses a temporary workspace and sessions directory. Never prints credentials.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid


class RpcClient:
    def __init__(self, binary, workspace, session=None):
        args = [str(binary), "--rpc", "--sessions-dir", str(workspace / "sessions")]
        if session:
            args += ["--session", session]
        self.proc = subprocess.Popen(
            args, cwd=workspace, env=os.environ.copy(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.events = queue.Queue()
        self.errors = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.error_reader = threading.Thread(target=self._read_errors, daemon=True)
        self.error_reader.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.events.put(json.loads(line))
            except ValueError:
                self.events.put({"type": "invalid_json", "line": line})
        self.events.put({"type": "eof"})

    def _read_errors(self):
        for line in self.proc.stderr:
            self.errors.append(line)

    def send(self, command):
        self.proc.stdin.write(json.dumps(command) + "\n")
        self.proc.stdin.flush()

    def until(self, predicate, timeout=120):
        deadline = time.monotonic() + timeout
        events = []
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            assert event["type"] not in ("eof", "invalid_json"), event
            if event["type"] == "response":
                assert event["success"], event
            events.append(event)
            if predicate(event):
                return events
        raise AssertionError("Timed out waiting for RPC events")

    def turn(self, message):
        self.send({"type": "prompt", "message": message})
        events = self.until(lambda e: e["type"] == "agent_end")
        ends = [e["message"] for e in events if e["type"] == "message_end"]
        assert ends, "No assistant response"
        assert all(m.get("stop_reason") not in ("error", "timeout", "aborted") for m in ends), ends
        assert any(m.get("usage", {}).get("input", 0) > 0 for m in ends), "Missing live token usage"
        return events

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join(timeout=2)
        self.error_reader.join(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()


def assistant_text(events):
    return "\n".join(
        content.get("text", "")
        for event in events if event["type"] == "message_end"
        for content in event["message"]["content"]
    )


def run():
    for name in ("RUPI_BASE_URL", "RUPI_API_KEY", "RUPI_MODEL"):
        assert os.environ.get(name), f"Set {name} to run live tests"
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/rupi").resolve()
    token = "live-proof-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="rupi-live-cli-") as directory:
        workspace = Path(directory)
        client = RpcClient(binary, workspace)
        try:
            events = client.turn(
                f"Use the write tool to create proof.txt in the current directory containing exactly {token} followed by a newline. "
                "Then use the read tool to verify its contents. Reply with the file's contents. Use no other tools."
            )
            assert (workspace / "proof.txt").read_text().strip() == token
            executed = [e["tool_name"] for e in events if e["type"] == "tool_execution_end"]
            assert "write" in executed and "read" in executed, executed
            assert token in assistant_text(events)
            print("PASS: live CLI write + read; file contents verified", flush=True)
            transcripts = list((workspace / "sessions").glob("*.jsonl"))
            assert len(transcripts) == 1, transcripts
            session_id = transcripts[0].stem
        finally:
            client.close()

        client = RpcClient(binary, workspace, session_id)
        try:
            events = client.turn("Without using tools, what exact token did you just write into proof.txt? Reply only with that token.")
            assert token in assistant_text(events)
            assert not any(e["type"] == "tool_execution_start" for e in events)
            print("PASS: restarted CLI resumed session and recalled token without tools", flush=True)

            client.send({"type": "new_session", "id": "reset"})
            client.until(lambda e: e.get("id") == "reset")
            client.send({"type": "get_messages", "id": "messages"})
            response = client.until(lambda e: e.get("id") == "messages")[-1]
            assert response["data"]["messages"] == []
            events = client.turn("Reply with exactly: fresh-session-ok")
            assert "fresh-session-ok" in assistant_text(events)
            print("PASS: session reset cleared history and a fresh live request succeeded", flush=True)
        finally:
            client.close()
    print("Live CLI checks passed for requested model " + os.environ["RUPI_MODEL"], flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        # Some API errors can echo request information; redact before displaying.
        key = os.environ.get("RUPI_API_KEY", "")
        message = str(error)
        if key:
            message = message.replace(key, "[REDACTED]")
        print("FAIL: " + message, file=sys.stderr)
        sys.exit(1)
