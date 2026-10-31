//! Timed words from Whisper's pieces. Ported from Ensayo's `speech/words.rs` (MIT, Miguel Monzones).

use crate::api::{Segment, Word};

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
    clean_flagged(raw.into_iter().map(|w| (w, true)).collect())
        .into_iter()
        .map(|(w, _)| w)
        .collect()
}

/// [`clean`] for pieces that carry a flag (whether the piece started a new word: [`Segment::starts_word`]). The flag
/// stays with its piece: a piece dropped takes its flag with it, and one glued to the word before leaves that
/// word's flag as it was.
pub fn clean_flagged(raw: Vec<(Word, bool)>) -> Vec<(Word, bool)> {
    let mut out: Vec<(Word, bool)> = Vec::with_capacity(raw.len());
    for (mut w, flag) in raw {
        w.text = w.text.trim().to_string();
        if is_noise(&w.text) {
            continue;
        }
        if w.end_ms < w.start_ms {
            w.end_ms = w.start_ms;
        }
        if is_punctuation(&w.text) {
            if let Some((prev, _)) = out.last_mut() {
                prev.text.push_str(&w.text);
                prev.end_ms = prev.end_ms.max(w.end_ms);
            }
            continue;
        }
        if let Some((prev, _)) = out.last()
            && w.start_ms < prev.end_ms
        {
            w.start_ms = prev.end_ms;
            w.end_ms = w.end_ms.max(w.start_ms);
        }
        out.push((w, flag));
    }
    out
}

/// The text of `segments` as one string: each segment's trimmed text, with a single space before it when it
/// [starts a word](Segment::starts_word) and something has been written already. Segments with no text are skipped.
/// Whisper puts no space between segments in Chinese, Japanese or Korean, and this does not add one there; for
/// Parakeet every segment starts a word, so it is the segments joined with single spaces.
pub fn join_segments(segments: &[Segment]) -> String {
    let mut out = String::new();
    for s in segments {
        let text = s.text.trim();
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() && s.starts_word {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}
