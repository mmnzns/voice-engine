//! Timed words from Whisper's pieces. Ported from Ensayo's `speech/words.rs` (MIT, Miguel Monzones).

use crate::api::Word;

/// Text engines emit for non-speech ("[BLANK_AUDIO]", "(music)", "*laughs*") or nothing at all. (Ensayo's rule:
/// punctuation-only pieces are not noise here, `clean` glues them to the word before.)
fn is_noise(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    let wrapped = |o: char, c: char| t.starts_with(o) && t.ends_with(c);
    wrapped('[', ']') || wrapped('(', ')') || wrapped('*', '*')
}

fn is_punctuation(text: &str) -> bool {
    let t = text.trim();
    !t.is_empty() && t.chars().all(|c| !c.is_alphanumeric())
}

/// Drops noise, glues punctuation-only pieces to the word before them, and keeps times in order.
pub fn clean(raw: Vec<Word>) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::with_capacity(raw.len());
    for mut w in raw {
        w.text = w.text.trim().to_string();
        if is_noise(&w.text) {
            continue;
        }
        if w.end_ms < w.start_ms {
            w.end_ms = w.start_ms;
        }
        if is_punctuation(&w.text) {
            if let Some(prev) = out.last_mut() {
                prev.text.push_str(&w.text);
                prev.end_ms = prev.end_ms.max(w.end_ms);
            }
            continue;
        }
        if let Some(prev) = out.last()
            && w.start_ms < prev.end_ms
        {
            w.start_ms = prev.end_ms;
            w.end_ms = w.end_ms.max(w.start_ms);
        }
        out.push(w);
    }
    out
}
