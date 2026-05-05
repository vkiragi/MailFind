//! Simple text chunker. Splits on paragraph then sentence boundaries while
//! keeping each chunk under `MAX_CHARS`. Good enough for email-sized text and
//! avoids pulling in tokenizers / model-specific splitters.

const MAX_CHARS: usize = 1000;
const MIN_CHARS: usize = 60;

pub fn split(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![];
    }
    if trimmed.chars().count() <= MAX_CHARS {
        return vec![trimmed.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for paragraph in trimmed.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        for sentence in split_sentences(paragraph) {
            let candidate_len = current.chars().count() + sentence.chars().count() + 1;
            if candidate_len > MAX_CHARS && !current.is_empty() {
                push_chunk(&mut chunks, std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&sentence);
            // Long sentence: hard split.
            while current.chars().count() > MAX_CHARS {
                let split_at = char_boundary(&current, MAX_CHARS);
                let leftover = current.split_off(split_at);
                push_chunk(&mut chunks, std::mem::take(&mut current));
                current = leftover;
            }
        }
        if current.chars().count() >= MIN_CHARS {
            push_chunk(&mut chunks, std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        push_chunk(&mut chunks, current);
    }
    chunks
}

fn push_chunk(out: &mut Vec<String>, c: String) {
    let trimmed = c.trim().to_string();
    if !trimmed.is_empty() {
        out.push(trimmed);
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?' | '\n') && current.trim().chars().count() > 8 {
            out.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn char_boundary(s: &str, n: usize) -> usize {
    let mut chars = s.char_indices();
    match chars.nth(n) {
        Some((i, _)) => i,
        None => s.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_short_text_unchanged() {
        let chunks = split("hello world");
        assert_eq!(chunks, vec!["hello world".to_string()]);
    }

    #[test]
    fn splits_on_paragraph_breaks() {
        let text = format!(
            "{}\n\n{}",
            "x".repeat(800),
            "y".repeat(800),
        );
        let chunks = split(&text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn handles_empty_input() {
        let chunks = split("");
        assert!(chunks.is_empty());
    }
}
