#!/bin/bash

# Test script for rupi RPC mode

# Start rupi in RPC mode
./target/release/rupi --rpc &
RUPI_PID=$!

# Give it a moment to start up
sleep 2

# Send a prompt command via stdin
(echo '{"type":"prompt","id":"1","message":"Say hello"}' ; sleep 5) | nc -w 1 localhost 0 > /tmp/rupi_output.txt 2>&1 &

# Wait a bit for the response
sleep 3

# Kill rupi
echo '{"type":"abort","id":"2"}' | nc -w 1 localhost 0
kill -TERM $RUPI_PID 2>/dev/null
wait $RUPI_PID 2>/dev/null

# Output the results
echo "=== Raw Output ==="
cat /tmp/rupi_output.txt

echo ""
echo "=== Checking for Expected Events ==="
if grep -q '"type":"response"' /tmp/rupi_output.txt; then
    echo "✓ Response event found"
else
    echo "✗ Response event not found"
fi

if grep -q '"type":"message_update"' /tmp/rupi_output.txt; then
    echo "✓ Message update event found"
else
    echo "✗ Message update event not found"
fi

if grep -q '"type":"agent_end"' /tmp/rupi_output.txt; then
    echo "✓ Agent end event found"
else
    echo "✗ Agent end event not found"
fi

# Clean up
rm -f /tmp/rupi_output.txt
