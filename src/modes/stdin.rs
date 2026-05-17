use std::io::{BufRead, Read, Write};
use std::time::Duration;

/// A special string sent through the stdin channel when double-Esc is detected.
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// A special string sent when Ctrl+D (EOF) is pressed.
pub const EOF_SIG: &str = "\x04";

/// Spawn a stdin reader that sends lines through an mpsc channel.
/// Detects double-Esc (`\x1b\x1b`) within 500ms and sends `CANCEL_LOOP_SIG`.
/// Uses cbreak mode with manual echo handling for a proper REPL experience.
pub fn spawn_stdin_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    #[cfg(unix)]
    {
        if let Ok(()) = enter_cbreak() {
            std::thread::spawn(move || cbreak_reader(tx));
            return;
        }
    }
    // Fallback: line-based reader
    std::thread::spawn(move || line_reader(tx));
}

// ── Terminal mode management ───────────────────────────────────────────────

#[cfg(unix)]
fn enter_cbreak() -> Result<(), ()> {
    use std::os::unix::io::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    let mut termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut termios) } != 0 {
        return Err(());
    }
    // Save original for restore
    std::thread_local! {
        static ORIG_TERMIOS: std::cell::RefCell<Option<libc::termios>> = const { std::cell::RefCell::new(None) };
    }
    ORIG_TERMIOS.with(|t| *t.borrow_mut() = Some(termios));

    let mut raw = termios;
    raw.c_iflag |= libc::ICRNL;
    raw.c_oflag |= libc::OPOST | libc::ONLCR;
    raw.c_cflag |= libc::CS8;
    raw.c_lflag &= !(libc::ICANON | libc::ECHO);
    raw.c_cc[libc::VMIN] = 1;
    raw.c_cc[libc::VTIME] = 0;
    if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &raw) } != 0 {
        return Err(());
    }
    Ok(())
}

#[cfg(unix)]
fn restore_termios() {
    use std::os::unix::io::AsRawFd;
    std::thread_local! {
        static ORIG_TERMIOS: std::cell::RefCell<Option<libc::termios>> = const { std::cell::RefCell::new(None) };
    }
    ORIG_TERMIOS.with(|t| {
        if let Some(orig) = t.borrow().as_ref() {
            let fd = std::io::stdin().as_raw_fd();
            let _ = unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, orig) };
        }
    });
}

// ── Echo-handling helpers ──────────────────────────────────────────────────

/// Print a single visible character to stdout (with proper newline conversion).
fn echo_char(c: u8) {
    let _ = std::io::stdout().write_all(&[c]);
    let _ = std::io::stdout().flush();
}

/// Erase one character from the terminal display.
fn echo_backspace() {
    let _ = std::io::stdout().write_all(b"\x08 \x08");
    let _ = std::io::stdout().flush();
}

/// Print a visual representation of a non-printable character (like \x1b → ^[).
fn echo_control(c: u8) {
    // Caret notation: ^@ = 0x00, ^A = 0x01, ..., ^[ = 0x1b, ..., ^? = 0x7f
    let code = if c == 0x7f { b'?' } else { c ^ 0x40 };
    let _ = std::io::stdout().write_all(&[b'^', code]);
    let _ = std::io::stdout().flush();
}

// ── cbreak reader ──────────────────────────────────────────────────────────

#[cfg(unix)]
fn cbreak_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    let mut stdin = std::io::stdin();
    let mut buf: Vec<u8> = Vec::new();
    let mut last_esc = std::time::Instant::now();
    let mut esc_count = 0;

    loop {
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
                    // ── Escape key ────────────────────────────────────
                    0x1b => {
                        let now = std::time::Instant::now();
                        if now.duration_since(last_esc) < Duration::from_millis(500)
                            && esc_count >= 1
                        {
                            // Double-Esc → cancel loop
                            let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                            esc_count = 0;
                            buf.clear();
                        } else {
                            esc_count += 1;
                            // Don't add Esc to buffer, don't echo it
                        }
                        last_esc = now;
                    }

                    // ── Enter ─────────────────────────────────────────
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

                    // ── Ctrl+D (EOF) ──────────────────────────────────
                    0x04 => {
                        let _ = tx.send(EOF_SIG.to_string());
                        buf.clear();
                        esc_count = 0;
                    }

                    // ── Backspace / DEL ────────────────────────────────
                    0x7f | 0x08 => {
                        if !buf.is_empty() {
                            buf.pop();
                            echo_backspace();
                        }
                        esc_count = 0;
                    }

                    // ── Printable ASCII ───────────────────────────────
                    0x20..=0x7e => {
                        buf.push(b);
                        echo_char(b);
                        esc_count = 0;
                    }

                    // ── Tab ────────────────────────────────────────────
                    0x09 => {
                        buf.push(b' ');
                        echo_char(b' ');
                        esc_count = 0;
                    }

                    // ── Ctrl+C, Ctrl+Z etc (handled by ISIG) ──────────
                    0x03 | 0x1a => {
                        buf.clear();
                        esc_count = 0;
                    }

                    // ── Other control chars / escape sequences ────────
                    _ => {
                        esc_count = 0;
                    }
                }
            }
        }
    }
    restore_termios();
}

// ── Fallback line reader ───────────────────────────────────────────────────

fn line_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        buf.clear();
        match stdin.lock().read_line(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let input = buf.trim().to_string();
                if input.is_empty() {
                    continue;
                }
                if input.contains("\x1b\x1b") {
                    let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                } else {
                    let _ = tx.send(input);
                }
            }
        }
    }
}

#[cfg(not(unix))]
fn enter_cbreak() -> Result<(), ()> {
    Err(())
}
