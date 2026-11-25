//! Tools executed by another process, usually on another machine.
//!
//! The local half of "the agent runs here, the tools run there". The host is any
//! command that speaks the tool protocol on stdin and stdout, which
//! `rupi --tool-daemon` does:
//!
//! ```text
//! --tool-command "ssh build-box rupi --tool-daemon"
//! --tool-command "docker exec -i work rupi --tool-daemon"
//! ```
//!
//! rupi stays out of the transport. There is no SSH logic here, no key handling
//! and no deployment: the command is the user's, so any transport that gives a
//! pipe works, and the credentials stay where the user put them.
//!
//! Calls are serialized. A reader thread owns the host's stdout and feeds a
//! channel, so a wedged host times out instead of hanging the agent, and a host
//! that dies is replaced on the next call.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Mutex, OnceLock};

/// Tools that mean nothing away from the conversation that owns them.
const LOCAL_ONLY: [&str; 2] = ["todo_write", "goal"];

/// Seconds added to the bash ceiling before a silent host is given up on.
const HOST_GRACE_SECS: u64 = 60;

/// Longest answer a host may send, in bytes.
///
/// `BufRead::lines` has no cap. A host that wrote an endless line with no newline
/// took the agent to 20 GB of memory in 41 seconds, and the deadline never fired
/// because nothing was waiting on time — it was waiting for a newline.
const MAX_ANSWER_BYTES: usize = 8 * 1024 * 1024;

/// How often the wait for an answer looks up to check for a cancellation.
const CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// What the reader thread sends when an answer passes the cap.
const ANSWER_TOO_LARGE: &str = "\u{0}rupi: answer too large";

struct ToolHost {
    command: String,
    running: Option<Running>,
}

struct Running {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

fn host() -> &'static Mutex<Option<ToolHost>> {
    static HOST: OnceLock<Mutex<Option<ToolHost>>> = OnceLock::new();
    HOST.get_or_init(|| Mutex::new(None))
}

/// Route every tool through `command`, or back to this process when `None`.
pub fn set_tool_command(command: Option<String>) {
    let mut guard = host().lock().unwrap_or_else(|e| e.into_inner());
    *guard = command.map(|command| ToolHost {
        command,
        running: None,
    });
}

/// Whether tools run somewhere else.
pub fn is_remote() -> bool {
    host().lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// Whether this tool must run in this process whatever the configuration says.
pub fn runs_locally(name: &str) -> bool {
    LOCAL_ONLY.contains(&name)
}

/// Send a tool call to the host and return its result.
///
/// `None` means no host is configured, and the caller runs the tool itself.
/// Every failure of the host returns `Some` with an explanation, because the
/// model has to learn that the tool did not run — reporting nothing would read
/// as a tool that returned nothing.
pub fn forward(
    call: &crate::tools::ToolCall,
    control: &crate::tools::BashControl,
) -> Option<String> {
    if runs_locally(&call.name) {
        return None;
    }
    // Checked before the queue and again after it. Calls wait their turn behind
    // one another, so without the second check an abort still let every queued
    // `write` and `edit` run, one after another, on the host.
    if !is_remote() {
        return None;
    }
    if control.stop_requested() {
        return Some(crate::tools::cancelled_notice());
    }
    let mut guard = host().lock().unwrap_or_else(|e| e.into_inner());
    let host = guard.as_mut()?;
    if control.stop_requested() {
        return Some(crate::tools::cancelled_notice());
    }
    // The id the answer must echo is unique to this exchange, not to the call. A
    // provider can repeat a call id, and a host that answered twice then handed
    // one call's output to the next. Draining stale lines alone is a race: the
    // leftover can arrive just after the drain.
    let exchange_id = format!("{}#{}", call.id, next_sequence());
    let request = serde_json::json!({
        "id": exchange_id,
        "name": call.name,
        "arguments": call.arguments,
    });
    Some(
        match exchange(host, &exchange_id, &request.to_string(), control) {
            Ok(result) => result,
            Err(e) => {
                // The next call starts a fresh host. A broken pipe is usually the
                // connection, not the command.
                stop_host(host);
                if e == "cancelled" {
                    crate::tools::cancelled_notice()
                } else {
                    format!(
                        "Error: the tool host ({}) did not answer: {}. The tool did not run.",
                        host.command, e
                    )
                }
            }
        },
    )
}

/// A number that never repeats in this process.
fn next_sequence() -> u64 {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// End the host and everything it started.
///
/// Killing the shell alone left its children running: a timeout on `ssh box rupi
/// --tool-daemon` orphaned the remote command, and a `docker exec` host left the
/// container process alive. The host leads its own process group for exactly this.
fn stop_host(host: &mut ToolHost) {
    let Some(mut running) = host.running.take() else {
        return;
    };
    #[cfg(unix)]
    {
        let group = running.child.id() as i32;
        if group > 1 {
            unsafe {
                libc::kill(-group, libc::SIGKILL);
            }
        }
    }
    let _ = running.child.kill();
    let _ = running.child.wait();
}

/// One request and its answer, starting the host if it is not running.
fn exchange(
    host: &mut ToolHost,
    id: &str,
    request: &str,
    control: &crate::tools::BashControl,
) -> Result<String, String> {
    if host.running.is_none() {
        host.running = Some(start(&host.command)?);
    }
    let running = host.running.as_mut().expect("just started");
    // Anything still in the channel answers a call that is over. A host that
    // answered twice, with a provider that reused an id, otherwise handed one
    // call's output to the next.
    loop {
        match running.lines.try_recv() {
            Ok(_) => continue,
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => return Err("the host closed its output".into()),
        }
    }
    writeln!(running.stdin, "{}", request).map_err(|e| e.to_string())?;
    running.stdin.flush().map_err(|e| e.to_string())?;

    let limit = std::time::Duration::from_secs(crate::tools::bash_timeout_max() + HOST_GRACE_SECS);
    let deadline = std::time::Instant::now() + limit;
    // Polled rather than waited on in one go, so an abort reaches a tool running
    // on another machine. One blocking wait made every remote call uncancellable
    // for up to the whole limit.
    let line = loop {
        match running.lines.recv_timeout(CANCEL_POLL) {
            Ok(line) => break line,
            Err(RecvTimeoutError::Disconnected) => return Err("the host closed its output".into()),
            Err(RecvTimeoutError::Timeout) => {}
        }
        if control.stop_requested() {
            return Err("cancelled".into());
        }
        if let Ok(Some(status)) = running.child.try_wait() {
            return Err(format!("the host exited ({})", status));
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!("no answer within {}s", limit.as_secs()));
        }
    };
    if line == ANSWER_TOO_LARGE {
        return Err(format!("the answer passed {} bytes", MAX_ANSWER_BYTES));
    }
    let answer: serde_json::Value =
        serde_json::from_str(&line).map_err(|e| format!("unreadable answer: {}", e))?;
    // Calls are serialized, so the answer belongs to this request. A mismatch
    // means the host is out of step, and continuing would attribute one tool's
    // output to another call.
    if answer["id"].as_str() != Some(id) {
        return Err(format!(
            "answered call {:?} while {} was in flight",
            answer["id"], id
        ));
    }
    if let Some(error) = answer["error"].as_str() {
        return Ok(format!("Error from the tool host: {}", error));
    }
    answer["result"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "the answer carried no result".to_string())
}

/// Read one line, stopping if it passes `cap` bytes.
fn read_capped<R: BufRead>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    cap: usize,
    too_large: &mut bool,
) -> std::io::Result<usize> {
    let mut total = 0usize;
    loop {
        let available = match reader.fill_buf() {
            Ok([]) => return Ok(total),
            Ok(bytes) => bytes,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        match available.iter().position(|byte| *byte == b'\n') {
            Some(at) => {
                buffer.extend_from_slice(&available[..at]);
                reader.consume(at + 1);
                return Ok(total + at + 1);
            }
            None => {
                let taken = available.len();
                buffer.extend_from_slice(available);
                reader.consume(taken);
                total += taken;
                if buffer.len() > cap {
                    *too_large = true;
                    return Ok(total);
                }
            }
        }
    }
}

/// Start the host and the thread that reads its answers.
fn start(command: &str) -> Result<Running, String> {
    let mut command_builder = Command::new("sh");
    command_builder
        .arg("-c")
        .arg(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // Inherited, so a message from ssh or from the remote rupi reaches the
        // user instead of disappearing into a pipe nobody reads.
        .stderr(Stdio::inherit());
    // Its own process group, so ending the host ends what it started.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command_builder.process_group(0);
    }
    let mut child = command_builder
        .spawn()
        .map_err(|e| format!("cannot start the tool host: {}", e))?;
    let stdin = child.stdin.take().ok_or("the host has no stdin")?;
    let stdout = child.stdout.take().ok_or("the host has no stdout")?;
    // Bounded, so a host that spews lines blocks its own reader thread instead of
    // growing this process without limit.
    let (tx, lines): (SyncSender<String>, Receiver<String>) = std::sync::mpsc::sync_channel(8);
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buffer: Vec<u8> = Vec::new();
        loop {
            buffer.clear();
            // Read with a cap. `lines()` has none: a host that wrote an endless
            // line with no newline took this process to 20 GB in 41 seconds, and
            // no deadline could fire, because nothing was waiting on time.
            let mut too_large = false;
            let read = read_capped(&mut reader, &mut buffer, MAX_ANSWER_BYTES, &mut too_large);
            if too_large {
                let _ = tx.send(ANSWER_TOO_LARGE.to_string());
                break;
            }
            match read {
                Ok(0) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buffer).trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok(Running {
        child,
        stdin,
        lines,
    })
}
