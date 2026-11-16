//! The prompt for Gemma 4, whose chat template llama.cpp's built-in template engine cannot run (it answers "error -1":
//! the template is Jinja with features the built-in matcher does not have, and its turn marks are new). The helper
//! writes the prompt itself, in the format of the template in the model's own GGUF file (Google's "Gemma 4 Canonical
//! Chat Template", 2026-07-09) for what this engine sends: an optional system text and one user text.
//!
//! The format: the start-of-text mark, the system turn (only when there is a system text), the user turn, and the
//! opening of the model's turn. Each text is trimmed, as the template does. Thinking stays off: the template turns it
//! on only with `enable_thinking`, which the helper never sets, so no thinking mark is written and a reply has no
//! thinking (the catalog's `thinks` is false for Gemma 4, and so is the helper's `inspect`). There are no tools.
//!
//! ```text
//! <bos><|turn>system
//! {system}<turn|>
//! <|turn>user
//! {user}<turn|>
//! <|turn>model
//! ```

const BOS: &str = "<bos>";
const TURN: &str = "<|turn>";
const END: &str = "<turn|>";

/// Is this Gemma 4's chat template: it has the new turn marks.
pub(crate) fn is_gemma4(template: &str) -> bool {
    template.contains(TURN) && template.contains(END)
}

/// The prompt for `system` and `user`, in the model's own format (see the module docs). Parsed with special tokens on,
/// like the prompts the other chat templates write.
pub(crate) fn prompt(system: &str, user: &str) -> String {
    let mut out = String::from(BOS);
    if !system.trim().is_empty() {
        out.push_str(&format!("{TURN}system\n{}{END}\n", system.trim()));
    }
    out.push_str(&format!("{TURN}user\n{}{END}\n{TURN}model\n", user.trim()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_system_turn_a_user_turn_and_the_opening_of_the_models_turn() {
        assert_eq!(
            prompt("Be brief.", "Hello there"),
            "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHello there<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn no_system_text_writes_no_system_turn() {
        for system in ["", "  \n "] {
            assert_eq!(
                prompt(system, "Hello there"),
                "<bos><|turn>user\nHello there<turn|>\n<|turn>model\n"
            );
        }
    }

    #[test]
    fn the_texts_are_trimmed_as_the_template_does_and_nothing_else_changes() {
        assert_eq!(
            prompt("  s \n", "\n two\nlines \"quotes\"  "),
            "<bos><|turn>system\ns<turn|>\n<|turn>user\ntwo\nlines \"quotes\"<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn only_gemma_4s_template_is_recognised() {
        let gemma = "{{- bos_token -}}{{- '<|turn>' + role + '\\n' }}{{- '<turn|>\\n' -}}";
        assert!(is_gemma4(gemma));
        // The older Gemma format, ChatML and Qwen's template are not it.
        assert!(!is_gemma4("<start_of_turn>user\n{{ x }}<end_of_turn>"));
        assert!(!is_gemma4("<|im_start|>user\n{{ x }}<|im_end|>"));
        assert!(!is_gemma4("{% if enable_thinking %}<think>{% endif %}"));
        assert!(!is_gemma4(""));
    }
}
