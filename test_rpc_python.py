import subprocess, json, time

# Start rupi in RPC mode
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        print(line.strip())
        obj = json.loads(line.strip())
        if obj.get("type") == target_type:
            return obj
        if obj.get("type") == "agent_end":
            print("Agent ended.")
            proc.terminate()
            return obj
    return None

# Send a prompt command
print(">>> Sending prompt command")
send({"type": "prompt", "id": "1", "message": "Say hello"})

# Read until response
print(">>> Waiting for response")
response = read_until("response")
if response:
    print("✓ Response received:", response)
else:
    print("✗ No response received")

# Read until agent_end or message_update
print(">>> Streaming events")
event_count = 0
while event_count < 10:  # Limit to prevent infinite loop
    event = read_until("agent_end", timeout=5)
    if event:
        print("✓ Agent end event received")
        break
    event = read_until("message_update", timeout=1)
    if event and "delta" in event.get("assistant_message_event", {}):
        print("Delta:", event["assistant_message_event"]["delta"])
    event_count += 1

# Cleanup
proc.terminate()
proc.wait()
