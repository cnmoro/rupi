use std::io::{BufRead, Read};

/// A special string sent through the stdin channel when double-Esc is detected.
/// In cooked terminal mode, the user must press Esc twice followed by Enter
/// (which delivers the bytes \x1b\x1b\n).
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// Spawn a stdin reader that sends lines through an mpsc channel.
/// Detects double-Esc (`\x1b\x1b`) anywhere in the input line and sends
/// `CANCEL_LOOP_SIG` instead. Uses simple blocking line I/O — no raw terminal
/// mode, so the REPL behaves normally (ECHO on, Enter sends \n, no ^M issues).
pub fn spawn_stdin_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let input = buf.trim().to_string();
                    if input.is_empty() {
                        continue;
                    }
                    // Detect double-Esc anywhere in the input
                    if input.contains("\x1b\x1b") {
                        let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                    } else {
                        let _ = tx.send(input);
                    }
                }
            }
        }
    });
}
