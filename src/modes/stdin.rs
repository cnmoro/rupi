use console::Term;

/// A special string sent through the stdin channel when double-Esc is detected.
pub const CANCEL_LOOP_SIG: &str = "\x1b\x1b";

/// A special string sent when Ctrl+D (EOF) is pressed.
pub const EOF_SIG: &str = "\x04";

/// Spawn a stdin reader that sends lines through an mpsc channel.
/// Uses `console::Term::read_line()` for full line editing (arrows, home,
/// end, backspace, delete, etc.) out of the box. Detects double-Esc
/// (`\x1b\x1b`) in the input and sends `CANCEL_LOOP_SIG` instead.
pub fn spawn_stdin_reader(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    std::thread::spawn(move || {
        let term = Term::stdout();
        loop {
            // read_line handles all terminal input, editing, and echoing.
            // It blocks until Enter is pressed.
            match term.read_line() {
                Ok(line) => {
                    // read_line strips trailing newline; check for empty
                    if line.is_empty() {
                        continue;
                    }
                    // Detect double-Esc anywhere in the input (user pressed
                    // Esc+Esc+Enter)
                    if line.contains("\x1b\x1b") {
                        let _ = tx.send(CANCEL_LOOP_SIG.to_string());
                    } else {
                        let _ = tx.send(line);
                    }
                }
                Err(_) => break,
            }
        }
    });
}
