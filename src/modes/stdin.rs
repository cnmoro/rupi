use std::io::{Read, Write};

/// A special string sent through the stdin channel when double-Esc is detected.
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// A special string sent when Ctrl+D (EOF) is pressed.
pub const EOF_SIG: &str = "\x04";

/// Read a line from the terminal with full readline editing.
/// Uses rustyline for proper handling of arrows, home, end, etc.
/// Returns `None` on EOF/Ctrl+D.
pub fn read_line_edited(prompt: &str) -> Option<String> {
    use rustyline::{Cmd, DefaultEditor, KeyCode, KeyEvent, Modifiers};

    let config = rustyline::config::Builder::new()
        .build();
    let mut rl = match DefaultEditor::with_config(config) {
        Ok(rl) => rl,
        Err(_) => return None,
    };
    // Bind Alt+Enter to insert a newline
    rl.bind_sequence(
        KeyEvent(KeyCode::Enter, Modifiers::ALT),
        Cmd::Newline,
    );

    match rl.readline(prompt) {
        Ok(line) => {
            if line.is_empty() {
                None
            } else {
                Some(line)
            }
        }
        Err(rustyline::error::ReadlineError::Eof) => None,
        Err(rustyline::error::ReadlineError::Interrupted) => None,
        Err(_) => None,
    }
}

/// Spawn a temporary stdin reader for the duration of streaming.
/// Reads bytes one at a time, sends complete lines through the channel,
/// and detects double-Esc for loop cancellation.
pub fn spawn_streaming_reader(
    tx: tokio::sync::mpsc::UnboundedSender<String>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        // Ensure terminal is in sane mode (cooked + echo)
        #[cfg(unix)]
        ensure_cooked_mode();
        let mut stdin = std::io::stdin();
        let mut buf: Vec<u8> = Vec::new();
        let mut last_esc = std::time::Instant::now();
        let mut esc_count = 0;

        loop {
            if stop.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let mut byte = [0u8; 1];
            match stdin.read(&mut byte) {
                Ok(0) => {
                    let _ = tx.send(EOF_SIG.to_string());
                    break;
                }
                Err(_) => break,
                Ok(_) => {
                    let b = byte[0];
                    match b {
                        0x1b => {
                            let now = std::time::Instant::now();
                            if now.duration_since(last_esc) < std::time::Duration::from_millis(500)
                                && esc_count >= 1
                            {
                                let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                                esc_count = 0;
                                buf.clear();
                                continue;
                            }
                            esc_count += 1;
                            last_esc = now;
                            let _ = read_escape_seq(&stdin);
                        }
                        b'\n' | b'\r' => {
                            if !buf.is_empty() {
                                let line = String::from_utf8_lossy(&buf).trim().to_string();
                                if !line.is_empty() {
                                    let _ = tx.send(line);
                                }
                            }
                            buf.clear();
                            esc_count = 0;
                        }
                        0x04 => {
                            let _ = tx.send(EOF_SIG.to_string());
                            buf.clear();
                        }
                        0x7f | 0x08 => {
                            buf.pop();
                            // Erase the character on screen (terminal might be in raw mode)
                            let _ = std::io::stdout().write_all(b"\x08 \x08");
                            let _ = std::io::stdout().flush();
                            esc_count = 0;
                        }
                        _ => {
                            buf.push(b);
                            esc_count = 0;
                        }
                    }
                }
            }
        }
    });
}

#[cfg(unix)]
fn ensure_cooked_mode() {
    use std::os::unix::io::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    if let Ok(mut termios) = unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(fd, &mut t) == 0 { Ok(t) } else { Err(()) }
    } {
        // Only restore if terminal is in non-canonical mode
        if termios.c_lflag & libc::ICANON == 0 {
            termios.c_lflag |= libc::ICANON | libc::ECHO | libc::ISIG;
            termios.c_iflag |= libc::BRKINT | libc::ICRNL | libc::IXON;
            termios.c_oflag |= libc::OPOST | libc::ONLCR;
            let _ = unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &termios) };
        }
    }
}

#[cfg(not(unix))]
fn ensure_cooked_mode() {}

fn read_escape_seq(stdin: &std::io::Stdin) -> std::io::Result<()> {
    let mut byte = [0u8; 1];
    let mut handle = stdin.lock();
    handle.read(&mut byte)?;
    match byte[0] {
        b'[' => {
            loop {
                handle.read(&mut byte)?;
                if byte[0] >= 0x40 && byte[0] <= 0x7e {
                    break;
                }
            }
        }
        b'O' => {
            handle.read(&mut byte)?;
        }
        _ => {}
    }
    Ok(())
}
