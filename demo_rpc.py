#!/usr/bin/env python3
import subprocess
import json
import sys

# Start the rupi process in RPC mode
proc = subprocess.Popen(
    ["./target/release/rupi", "--rpc"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

def send(cmd):
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()

def read_until(target_type, timeout=10):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            obj = json.loads(line)
            print(f"Received: {obj}", file=sys.stderr)
            if obj.get("type") == target_type:
                return obj
        except json.JSONDecodeError:
            continue
    return None

# Send a prompt
send({"type": "prompt", "id": "1", "message": "What is 2+2?"})
print("Sent prompt: What is 2+2?", file=sys.stderr)

# Read response
response = read_until("response")
if response:
    print(f"Response: {response}", file=sys.stderr)

# Read message updates
while True:
    event = read_until("agent_end", timeout=30)
    if event:
        print(f"Agent ended: {event}", file=sys.stderr)
        break
    else:
        print("No agent_end event received within 30 seconds.", file=sys.stderr)
        break

# Terminate the process
proc.terminate()
proc.wait()
