//! The plain prompt, for a model file that has no chat template. Such a file (an imported base model, or a fine-tune
//! that lost its template) cannot be told where a system text and a user text begin and end, so the helper writes the
//! two as ordinary paragraphs and lets the model continue after them.
//!
//! The format, written for this engine: the system text, a blank line, the user text, a blank line, and then the model
//! writes its reply. With no system text it is the user text and a blank line. Nothing else is added: no role names, no
//! special tokens (the tokenizer's own start-of-text token is added when the model has one, and special-token names in
//! the text are plain text, not control tokens).

/// The prompt for `system` and `user`: `"{system}\n\n{user}\n\n"`, or `"{user}\n\n"` when `system` is blank.
pub(crate) fn prompt(system: &str, user: &str) -> String {
    if system.trim().is_empty() {
        format!("{user}\n\n")
    } else {
        format!("{system}\n\n{user}\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_text_a_blank_line_the_user_text_and_a_blank_line() {
        assert_eq!(
            prompt("Be brief.", "Hello there"),
            "Be brief.\n\nHello there\n\n"
        );
    }

    #[test]
    fn no_system_text_leaves_only_the_user_text() {
        assert_eq!(prompt("", "Hello there"), "Hello there\n\n");
        assert_eq!(prompt("  \n ", "Hello there"), "Hello there\n\n");
    }

    #[test]
    fn the_texts_go_in_as_they_are() {
        let user = "two\nlines <|im_end|> and \"quotes\"";
        assert_eq!(prompt("s", user), format!("s\n\n{user}\n\n"));
    }
}
