use std::io::{BufRead, Read};
use std::time::Duration;

/// A special string sent through the stdin channel when double-Esc is detected.
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// A special string sent when Ctrl+D (EOF) is pressed.
pub const EOF_SIG: &str = "\x04";

/// Spawn a stdin reader that sends lines through an mpsc channel.
/// Detects double-Esc (`\x1b\x1b`) within 500ms and sends `CANCEL_LOOP_SIG`.
/// Uses cbreak terminal mode (ICANON off) for immediate per-key reads while
/// keeping ECHO on and output processing enabled for normal REPL behavior.
pub fn spawn_stdin_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    #[cfg(unix)]
    {
        if let Ok(()) = enable_cbreak() {
            std::thread::spawn(move || cbreak_reader(tx));
            return;
        }
    }
    // Fallback: line-based reader
    std::thread::spawn(move || line_reader(tx));
}

#[cfg(unix)]
fn enable_cbreak() -> Result<(), ()> {
    use std::os::unix::io::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    let mut termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut termios) } != 0 {
        return Err(());
    }
    let mut raw = termios;
    raw.c_iflag |= libc::ICRNL;
    raw.c_oflag |= libc::OPOST | libc::ONLCR;
    raw.c_cflag |= libc::CS8;
    raw.c_lflag &= !libc::ICANON;
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
    let fd = std::io::stdin().as_raw_fd();
    if let Ok(mut termios) = unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(fd, &mut t) == 0 { Ok(t) } else { Err(()) }
    } {
        termios.c_lflag |= libc::ECHO | libc::ICANON | libc::ISIG;
        termios.c_iflag |= libc::BRKINT | libc::ICRNL | libc::IXON;
        termios.c_oflag |= libc::OPOST | libc::ONLCR;
        let _ = unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &termios) };
    }
}

#[cfg(unix)]
fn cbreak_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    let mut stdin = std::io::stdin();
    let mut buf = Vec::new();
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
                if b == 0x1b {
                    let now = std::time::Instant::now();
                    if now.duration_since(last_esc) < Duration::from_millis(500) {
                        // Double-Esc!
                        let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                        esc_count = 0;
                        buf.clear();
                    } else {
                        esc_count = 1;
                        buf.push(b);
                    }
                    last_esc = now;
                } else if b == b'\n' || b == b'\r' {
                    if !buf.is_empty() {
                        let line = String::from_utf8_lossy(&buf).trim().to_string();
                        if !line.is_empty() {
                            // Also check for double-Esc in the middle of a line
                            if line.contains("\x1b\x1b") {
                                let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                            } else {
                                let _ = tx.send(line);
                            }
                        }
                    }
                    buf.clear();
                    esc_count = 0;
                } else if b == 0x04 {
                    // Ctrl+D
                    let _ = tx.send(EOF_SIG.to_string());
                    buf.clear();
                    esc_count = 0;
                } else {
                    buf.push(b);
                    esc_count = 0;
                }
            }
        }
    }
    restore_termios();
}

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
fn enable_cbreak() -> Result<(), ()> {
    Err(())
}
