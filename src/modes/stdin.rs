use std::io::Read;
use std::time::Duration;

/// A special string sent through the stdin channel when double-Esc is detected.
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// Spawn a stdin reader that detects double-Esc for loop cancellation.
/// On Unix, uses raw terminal mode; on other platforms falls back to line reading.
pub fn spawn_stdin_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    if let Ok(mut terminal) = enable_raw_mode() {
        std::thread::spawn(move || {
            let mut stdin = std::io::stdin();
            let mut buf = Vec::new();
            let mut last_esc = std::time::Instant::now();
            let mut in_esc_window = false;
            loop {
                let mut byte = [0u8; 1];
                match stdin.read(&mut byte) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let b = byte[0];
                        if b == 0x1b {
                            let now = std::time::Instant::now();
                            if in_esc_window && now.duration_since(last_esc) < Duration::from_millis(500) {
                                let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                                in_esc_window = false;
                                buf.clear();
                            } else {
                                in_esc_window = true;
                                last_esc = now;
                                buf.push(b);
                            }
                        } else if b == b'\n' || b == b'\r' {
                            if !buf.is_empty() {
                                let line = String::from_utf8_lossy(&buf).trim().to_string();
                                if !line.is_empty() {
                                    let _ = tx.send(line);
                                }
                            }
                            buf.clear();
                            in_esc_window = false;
                        } else if b == 0x04 {
                            break;
                        } else {
                            buf.push(b);
                            in_esc_window = false;
                        }
                    }
                }
            }
            let _ = disable_raw_mode(terminal);
        });
    } else {
        // Fallback: simple line reader
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let input = line.trim().to_string();
                        if !input.is_empty() {
                            let _ = tx.send(input);
                        }
                    }
                }
            }
        });
    }
}

#[cfg(unix)]
fn enable_raw_mode() -> Result<std::os::unix::io::RawFd, ()> {
    use std::os::unix::io::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    let mut termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut termios) } != 0 {
        return Err(());
    }
    let mut raw = termios;
    raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
    raw.c_oflag |= libc::OPOST | libc::ONLCR;
    raw.c_cflag |= libc::CS8;
    raw.c_lflag &= !(libc::ICANON | libc::IEXTEN | libc::ISIG);
    raw.c_cc[libc::VMIN] = 1;
    raw.c_cc[libc::VTIME] = 0;
    if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &raw) } != 0 {
        return Err(());
    }
    Ok(fd)
}

#[cfg(unix)]
fn disable_raw_mode(_fd: std::os::unix::io::RawFd) {
    use std::os::unix::io::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    let mut termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut termios) } == 0 {
        termios.c_lflag |= libc::ECHO | libc::ICANON | libc::ISIG;
        termios.c_iflag |= libc::BRKINT | libc::ICRNL | libc::IXON;
        termios.c_oflag |= libc::OPOST | libc::ONLCR;
        let _ = unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &termios) };
    }
}

#[cfg(not(unix))]
fn enable_raw_mode() -> Result<(), ()> {
    Err(())
}

#[cfg(not(unix))]
fn disable_raw_mode(_: ()) {}
