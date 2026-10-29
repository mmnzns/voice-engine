//! Thinking: models like Qwen 3.5 can write their reasoning between `<think>` and `</think>` before answering.
//! llama.cpp's built-in template engine can't pass the template's `enable_thinking` switch, so the helper ends the
//! prompt itself: an empty thinking block to answer straight away (what Qwen's own template writes for
//! `enable_thinking=false`), or an opening tag to think first.

pub(crate) const OPEN: &str = "<think>";
pub(crate) const CLOSE: &str = "</think>";

/// The prompt ending that skips thinking.
pub(crate) const SKIP: &str = "<think>\n\n</think>\n\n";
/// The prompt ending that starts thinking.
pub(crate) const START: &str = "<think>\n";
/// Written by the helper when thinking runs past its budget.
pub(crate) const FORCE_END: &str = "\n</think>\n\n";

/// Does a chat template have a thinking step?
pub(crate) fn template_thinks(template: &str) -> bool {
    template.contains(OPEN)
}

/// What the helper appends to the prompt: nothing for a template without a thinking step, otherwise the ending
/// that starts thinking (`thinking`) or skips it.
pub(crate) fn prompt_ending(thinks: bool, thinking: bool) -> &'static str {
    match (thinks, thinking) {
        (false, _) => "",
        (true, true) => START,
        (true, false) => SKIP,
    }
}

/// Where the thinking ends in text written so far (the byte just after `</think>`), if it has.
pub(crate) fn end_of_thinking(text: &[u8]) -> Option<usize> {
    text.windows(CLOSE.len())
        .position(|w| w == CLOSE.as_bytes())
        .map(|i| i + CLOSE.len())
}

/// Removes `<think>...</think>` blocks from an answer. An unclosed block (cut off while thinking) drops everything
/// from the tag on. A closing tag with no opening one drops everything before it. From Ensayo's `ai/json.rs` (MIT).
pub(crate) fn strip_think(text: &str) -> String {
    let mut out = text.to_string();
    if let Some(close) = out.rfind(CLOSE)
        && out.find(OPEN).is_none_or(|open| open > close)
    {
        out = out[close + CLOSE.len()..].to_string();
    }
    while let Some(open) = out.find(OPEN) {
        match out[open..].find(CLOSE) {
            Some(rel) => out.replace_range(open..open + rel + CLOSE.len(), ""),
            None => out.truncate(open),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_with_a_think_tag_think() {
        assert!(template_thinks(
            "{% if enable_thinking %}<think>{% endif %}"
        ));
        assert!(!template_thinks("<|im_start|>assistant"));
    }

    #[test]
    fn the_prompt_ending_follows_the_template_and_the_request() {
        assert_eq!(prompt_ending(true, true), "<think>\n");
        assert_eq!(prompt_ending(true, false), "<think>\n\n</think>\n\n");
        // A template without a thinking step never gets an ending, whatever was asked.
        assert_eq!(prompt_ending(false, true), "");
        assert_eq!(prompt_ending(false, false), "");
    }

    #[test]
    fn the_end_of_thinking_is_found_after_the_closing_tag() {
        assert_eq!(end_of_thinking(b"plan it</think>\n\n{}"), Some(15));
        assert_eq!(end_of_thinking(b"still thinking"), None);
    }

    #[test]
    fn the_forced_end_closes_the_thinking() {
        let text = format!("half a plan{}", FORCE_END);
        assert_eq!(
            end_of_thinking(text.as_bytes()),
            Some("half a plan\n</think>".len())
        );
    }

    #[test]
    fn think_blocks_are_stripped() {
        for (input, want) in [
            ("<think>a</think>answer", "answer"),
            ("before<think>a</think> after", "before after"),
            ("answer<think>cut off", "answer"),
            ("reasoning without an opening tag</think>answer", "answer"),
            ("plain answer", "plain answer"),
            ("<think>a</think>x<think>b</think>y", "xy"),
        ] {
            assert_eq!(strip_think(input), want, "{input:?}");
        }
    }
}
