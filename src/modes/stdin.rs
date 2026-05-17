use std::io::BufRead;

/// A special string sent through the stdin channel when double-Esc is detected.
/// In cooked terminal mode, the user must press Esc twice followed by Enter
/// (which delivers the bytes \x1b\x1b\n).
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// A special string sent when Ctrl+D (EOF) is pressed.
pub const EOF_SIG: &str = "\x04";

/// Spawn a stdin reader that sends lines through an mpsc channel.
/// Detects double-Esc (`\x1b\x1b`) anywhere in the input line and sends
/// `CANCEL_LOOP_SIG` instead. Ctrl+D sends `EOF_SIG` so the main loop
/// can distinguish EOF from normal input (e.g., to cancel a loop instead
/// of exiting). Uses simple blocking line I/O — no raw terminal mode.
pub fn spawn_stdin_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf) {
                Ok(0) => {
                    // EOF (Ctrl+D) — send special signal instead of breaking,
                    // so the main loop can react (e.g. cancel loop, exit REPL).
                    let _ = tx.send(EOF_SIG.to_string());
                    // After EOF, stdin is closed — no more input possible
                    break;
                }
                Err(_) => break,
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
