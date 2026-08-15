//! RFC822 message parsing using `mail-parser`. Produces a uniform
//! [`ParsedMessage`] that the rest of the system stores and indexes.

use chrono::{DateTime, TimeZone, Utc};
use mail_parser::{Address, MessageParser};
use uuid::Uuid;

use crate::db::queries::NewMessage;
use crate::error::{AppError, AppResult};

/// Normalized representation of a parsed email used by ingestion and IMAP sync.
#[derive(Debug, Clone)]
pub struct ParsedMessage {
    pub rfc822_message_id: Option<String>,
    pub thread_id: Option<String>,
    pub subject: Option<String>,
    pub sender_display: Option<String>,
    pub sender_email: Option<String>,
    pub sender_domain: Option<String>,
    pub recipients: Option<String>,
    pub sent_at: Option<DateTime<Utc>>,
    pub body_plain: Option<String>,
    pub body_html: Option<String>,
    pub has_attachments: bool,
    pub raw_size: usize,
}

impl ParsedMessage {
    /// Generates a short snippet to preview in search results. Mirrors the
    /// prior backend behavior in
    /// `packages/backend/rag/context.py::_prepare_email_context`.
    pub fn snippet(&self) -> String {
        let body = self
            .body_plain
            .as_deref()
            .unwrap_or_else(|| self.body_html.as_deref().unwrap_or(""));
        compact_text(&clean_body(body), 200)
    }

    pub fn into_new_message(
        self,
        account_id: &str,
        mailbox_id: Option<String>,
        imap_uid: Option<i64>,
    ) -> NewMessage {
        let snippet = self.snippet();
        NewMessage {
            id: Uuid::new_v4().to_string(),
            account_id: account_id.to_string(),
            mailbox_id,
            imap_uid,
            rfc822_message_id: self.rfc822_message_id,
            thread_id: self.thread_id,
            subject: self.subject,
            sender: self.sender_display,
            sender_email: self.sender_email,
            sender_domain: self.sender_domain,
            recipients: self.recipients,
            sent_at: self.sent_at,
            received_at: self.sent_at,
            snippet: Some(snippet),
            body_plain: self.body_plain,
            body_html: self.body_html,
            has_attachments: self.has_attachments,
            raw_size: Some(self.raw_size as i64),
        }
    }
}

pub fn parse_rfc822(bytes: &[u8]) -> AppResult<ParsedMessage> {
    let raw_size = bytes.len();
    let parser = MessageParser::default();
    let msg = parser
        .parse(bytes)
        .ok_or_else(|| AppError::MailParse("failed to parse rfc822 bytes".into()))?;

    let subject = msg.subject().map(|s| s.to_string());
    let (sender_display, sender_email) = first_address(msg.from());
    let sender_domain = sender_email
        .as_ref()
        .and_then(|e| e.split('@').nth(1).map(|d| d.to_lowercase()));
    let recipients = format_addresses(msg.to())
        .into_iter()
        .chain(format_addresses(msg.cc()))
        .collect::<Vec<_>>()
        .join(", ");
    let recipients = if recipients.is_empty() {
        None
    } else {
        Some(recipients)
    };

    let sent_at = msg.date().and_then(|d| {
        Utc.timestamp_opt(d.to_timestamp(), 0)
            .single()
    });

    let body_plain = msg.body_text(0).map(|c| c.to_string());
    let body_html = msg.body_html(0).map(|c| c.to_string());
    let has_attachments = msg.attachments().any(|p| !p.is_message());

    let rfc822_message_id = msg.message_id().map(|s| s.to_string());
    let thread_id = msg
        .in_reply_to()
        .as_text_list()
        .and_then(|list| list.first().map(|s| s.to_string()))
        .or_else(|| {
            msg.references()
                .as_text_list()
                .and_then(|list| list.first().map(|s| s.to_string()))
        })
        .or_else(|| rfc822_message_id.clone());

    Ok(ParsedMessage {
        rfc822_message_id,
        thread_id,
        subject,
        sender_display,
        sender_email,
        sender_domain,
        recipients,
        sent_at,
        body_plain,
        body_html,
        has_attachments,
        raw_size,
    })
}

fn first_address(addr: Option<&Address>) -> (Option<String>, Option<String>) {
    let Some(addr) = addr else {
        return (None, None);
    };
    let list = match addr {
        Address::List(items) => items.as_slice(),
        Address::Group(groups) => {
            for g in groups {
                if let Some(first) = g.addresses.first() {
                    return (
                        first.name.as_ref().map(|c| c.to_string()),
                        first.address.as_ref().map(|c| c.to_string()),
                    );
                }
            }
            return (None, None);
        }
    };
    list.first()
        .map(|a| {
            (
                a.name.as_ref().map(|c| c.to_string()),
                a.address.as_ref().map(|c| c.to_string()),
            )
        })
        .unwrap_or((None, None))
}

fn format_addresses(addr: Option<&Address>) -> Vec<String> {
    let Some(addr) = addr else { return vec![] };
    match addr {
        Address::List(items) => items
            .iter()
            .filter_map(|a| {
                let name = a.name.as_ref().map(|c| c.to_string());
                let email = a.address.as_ref().map(|c| c.to_string());
                match (name, email) {
                    (Some(n), Some(e)) => Some(format!("{n} <{e}>")),
                    (None, Some(e)) => Some(e),
                    (Some(n), None) => Some(n),
                    (None, None) => None,
                }
            })
            .collect(),
        Address::Group(groups) => groups
            .iter()
            .flat_map(|g| {
                g.addresses
                    .iter()
                    .filter_map(|a| a.address.as_ref().map(|c| c.to_string()))
            })
            .collect(),
    }
}

/// Builds the text used to chunk a message for indexing: a small header
/// blurb (subject/from/to) followed by the cleaned, truncated body. Shared by
/// live ingest (`mail::import`, `mail::sync`) and the `rechunk`/backfill
/// example binaries so they can never drift from each other — the three call
/// sites used to carry byte-identical logic independently.
pub fn build_chunk_input(subject: &str, sender: &str, recipients: &str, body: &str, max_chars: usize) -> String {
    let header_blurb = format!("Subject: {subject}\nFrom: {sender}\nTo: {recipients}\n");
    format!("{header_blurb}\n{}", compact_text(&clean_body(body), max_chars))
}

/// Decodes HTML entities, strips invisible zero-width characters, and drops
/// CSS blocks. Applied to email bodies before they become a stored snippet or
/// indexed chunk text, so marketing-email preheader padding (`&zwnj;` /
/// `&nbsp;` filler used to control preview length) and raw CSS don't pollute
/// what the user sees or what search matches against.
pub fn clean_body(s: &str) -> String {
    strip_css(&strip_invisible(&decode_entities(s)))
}

/// Decodes common named and numeric HTML/XML entities to their literal
/// characters. Deliberately narrow — this is NOT a full HTML parser and never
/// touches `<`/`>` outside of actual `&lt;`/`&gt;` entities, so bare
/// angle-bracketed URLs in plain-text mail (`<https://example.com>`) survive
/// untouched. (`mail_parser::decoders::html::html_to_text` was considered and
/// rejected for this reason — it strips anything tag-shaped.)
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    // Longest named entity below plus a small margin.
    const MAX_ENTITY_LEN: usize = 10;
    while i < n {
        if chars[i] != '&' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut semi = None;
        while j < n && j - i <= MAX_ENTITY_LEN {
            match chars[j] {
                ';' => {
                    semi = Some(j);
                    break;
                }
                '&' => break,
                c if c.is_whitespace() => break,
                _ => j += 1,
            }
        }
        if let Some(semi) = semi {
            let body: String = chars[i + 1..semi].iter().collect();
            if let Some(decoded) = decode_entity_body(&body) {
                out.push(decoded);
                i = semi + 1;
                continue;
            }
        }
        // Not a recognized entity (e.g. "AT&T", a bare `&`) — keep it literal.
        out.push('&');
        i += 1;
    }
    out
}

fn decode_entity_body(body: &str) -> Option<char> {
    if let Some(rest) = body.strip_prefix('#') {
        let (radix, digits) = match rest.strip_prefix('x').or_else(|| rest.strip_prefix('X')) {
            Some(hex) => (16, hex),
            None => (10, rest),
        };
        if digits.is_empty() {
            return None;
        }
        return u32::from_str_radix(digits, radix).ok().and_then(char::from_u32);
    }
    Some(match body {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{00A0}',
        "zwnj" => '\u{200C}',
        "zwj" => '\u{200D}',
        "zwsp" => '\u{200B}',
        "mdash" => '\u{2014}',
        "ndash" => '\u{2013}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201C}',
        "rdquo" => '\u{201D}',
        "hellip" => '\u{2026}',
        "copy" => '\u{00A9}',
        "reg" => '\u{00AE}',
        "trade" => '\u{2122}',
        _ => return None,
    })
}

/// Removes invisible/formatting characters: ZWSP/ZWNJ/ZWJ/LRM/RLM
/// (U+200B-200F), BOM (U+FEFF), combining grapheme joiner (U+034F), and soft
/// hyphen (U+00AD). Runs after entity decoding so both literal UTF-8
/// characters and ones produced by decoding `&zwnj;`-style entities are
/// caught. Marketing email uses these as filler to control preview-text
/// length — measured on a 5,000-snippet sample of the live corpus, U+034F
/// alone appeared 8,514 times, more than any other offender including ZWNJ.
/// Left in place they eat the snippet's char budget and (for U+034F
/// especially) often render as visible tofu/dotted-circle glyphs.
pub fn strip_invisible(s: &str) -> String {
    if !s.chars().any(is_invisible) {
        return s.to_string();
    }
    s.chars().filter(|c| !is_invisible(*c)).collect()
}

fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{200B}'..='\u{200F}' | '\u{FEFF}' | '\u{034F}' | '\u{00AD}'
    )
}

/// Strip inline CSS rules (`@media`/`.selector { … }`) that bleed into
/// `body_plain` of marketing/transactional email. These flood the FTS index
/// with junk tokens like `!important`, hex colors, and CSS keywords that
/// match unrelated user queries (e.g. "important emails" matched eBay device
/// alerts solely on `!important;` declarations in the embedded styles).
///
/// Walks the string with brace-depth tracking. For each balanced `{…}`
/// block, drops it if the contents look CSS-shaped (contains both `:` and
/// `;`). Also rewinds to drop the preceding selector tokens (e.g.
/// `.mh_N1_accountGeneric .title a`). Conservative: never strips a `{…}`
/// block that doesn't look like CSS (e.g. code snippets, JSON in body).
pub fn strip_css(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut keep = vec![true; n];
    let mut i = 0;
    while i < n {
        if chars[i] != '{' {
            i += 1;
            continue;
        }
        // Find matching `}` with depth tracking.
        let mut depth = 1;
        let mut j = i + 1;
        while j < n {
            match chars[j] {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        if depth != 0 {
            i += 1;
            continue;
        }
        // chars[i..=j] is a balanced `{...}` block.
        let inner: String = chars[i + 1..j].iter().collect();
        let css_like = inner.contains(':') && inner.contains(';');
        if !css_like {
            i = j + 1;
            continue;
        }
        for k in i..=j {
            keep[k] = false;
        }
        // Walk back over selector tokens. Selectors are runs of identifier-ish
        // chars plus `.`, `#`, `:`, `*`, `,`, `>`, `+`, `~`, `(`, `)`, spaces,
        // and the `@media (...)` prefix. Stop at newline or a sentence-ending
        // punctuation that is followed by whitespace (i.e. the end of real
        // prose).
        let mut k = i;
        while k > 0 {
            let prev = chars[k - 1];
            if prev == '\n' {
                break;
            }
            // Sentence end: `.`, `!`, or `?` followed by whitespace (real prose
            // boundary) — but only when the char itself isn't part of a CSS
            // token like `.foo` or `!important`. We've already passed any
            // such CSS by now since they're inside braces; here we're scanning
            // backwards through selector text only.
            if matches!(prev, '.' | '!' | '?')
                && chars
                    .get(k)
                    .map(|c| c.is_whitespace())
                    .unwrap_or(true)
                && k >= 2
                && !chars[k - 2].is_whitespace()
                && chars[k - 2].is_alphabetic()
            {
                break;
            }
            keep[k - 1] = false;
            k -= 1;
        }
        i = j + 1;
    }
    chars
        .iter()
        .zip(keep.iter())
        .filter_map(|(c, &k)| if k { Some(*c) } else { None })
        .collect()
}

/// Collapse whitespace and trim to `max_chars` characters.
pub fn compact_text(s: &str, max_chars: usize) -> String {
    let mut buf = String::with_capacity(s.len().min(max_chars + 64));
    let mut last_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_space {
                buf.push(' ');
                last_space = true;
            }
        } else {
            buf.push(c);
            last_space = false;
        }
        if buf.chars().count() >= max_chars {
            break;
        }
    }
    buf.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = b"From: Alice <alice@example.com>\r\n\
To: Bob <bob@example.com>\r\n\
Subject: Hello\r\n\
Date: Mon, 1 Jan 2024 12:00:00 +0000\r\n\
Message-ID: <abc123@example.com>\r\n\
\r\n\
This is a test message body.\r\n";

    #[test]
    fn parses_basic_rfc822() {
        let parsed = parse_rfc822(SAMPLE).unwrap();
        assert_eq!(parsed.subject.as_deref(), Some("Hello"));
        assert_eq!(parsed.sender_email.as_deref(), Some("alice@example.com"));
        assert_eq!(parsed.sender_domain.as_deref(), Some("example.com"));
        assert!(parsed
            .body_plain
            .as_deref()
            .unwrap_or("")
            .contains("test message"));
        assert_eq!(
            parsed.rfc822_message_id.as_deref(),
            Some("abc123@example.com")
        );
    }

    #[test]
    fn snippet_collapses_whitespace() {
        let parsed = parse_rfc822(SAMPLE).unwrap();
        let s = parsed.snippet();
        assert!(s.contains("test message"));
        assert!(!s.contains("\r\n"));
    }

    #[test]
    fn strip_css_removes_inline_rules() {
        // Real eBay device-alert sample (truncated).
        let input = "Here's what you need to know.\n\
@media (prefers-color-scheme:dark) { .mgh{background:#212121!important;} } \
Let's make sure this was you \
.mh_N1_accountGeneric .title a:focus, .mh_N1_accountGeneric .subcopy a:focus, \
.mh_N1_accountGeneric .legalese a:focus{outline:3px solid #000000!important;padding:0px!important;} \
.mh_N1_accountGeneric .title a{color:#111820!important;} \
Time of sign-in Mar 11, 2026.";
        let out = strip_css(input);
        // The signal we care about: the literal `!important` flood is gone.
        assert!(!out.contains("!important"), "CSS leaked through: {out}");
        // And real prose survives.
        assert!(out.contains("what you need to know"));
        assert!(out.contains("Time of sign-in"));
    }

    #[test]
    fn strip_css_preserves_non_css_braces() {
        // Code snippets / JSON in prose shouldn't be touched.
        let input = "Here is some json: { \"name\": \"alice\" } and that's all.";
        let out = strip_css(input);
        assert!(out.contains("\"name\""), "stripped non-CSS braces: {out}");
    }

    #[test]
    fn decode_entities_handles_named_and_numeric() {
        assert_eq!(decode_entities("Rock &amp; Roll"), "Rock & Roll");
        assert_eq!(decode_entities("a &lt; b &gt; c"), "a < b > c");
        assert_eq!(decode_entities("&quot;quoted&quot;"), "\"quoted\"");
        assert_eq!(decode_entities("caf&#233;"), "café");
        assert_eq!(decode_entities("caf&#xE9;"), "café");
        // Real preheader-padding example from the corpus.
        assert_eq!(decode_entities("Chanel&zwnj; &zwnj;&nbsp;"), "Chanel\u{200C} \u{200C}\u{00A0}");
    }

    #[test]
    fn decode_entities_leaves_bare_ampersands_alone() {
        // No terminating `;` nearby -> not an entity, keep literal.
        assert_eq!(decode_entities("AT&T"), "AT&T");
        assert_eq!(decode_entities("Rock & Roll forever"), "Rock & Roll forever");
    }

    #[test]
    fn strip_invisible_removes_zero_width_chars() {
        let input = "Chanel\u{200C} \u{200C}\u{00A0}\u{FEFF}extraits";
        let out = strip_invisible(input);
        assert!(!out.contains('\u{200C}'));
        assert!(!out.contains('\u{FEFF}'));
        assert!(out.contains("Chanel"));
        assert!(out.contains("extraits"));
    }

    #[test]
    fn strip_invisible_removes_combining_grapheme_joiner_and_soft_hyphen() {
        // U+034F (combining grapheme joiner) was the single most common
        // padding character found in the live corpus (8,514 hits in a
        // 5,000-snippet sample) -- more common than ZWNJ. Left in place it
        // often renders as a visible tofu/dotted-circle glyph, not nothing.
        let input = "rule\u{034F} \u{034F} \u{034F}soft\u{00AD}hyphen\u{200E}\u{200F}end";
        let out = strip_invisible(input);
        assert!(!out.contains('\u{034F}'), "CGJ not stripped: {out:?}");
        assert!(!out.contains('\u{00AD}'), "soft hyphen not stripped: {out:?}");
        assert!(!out.contains('\u{200E}'), "LRM not stripped: {out:?}");
        assert!(!out.contains('\u{200F}'), "RLM not stripped: {out:?}");
        assert!(out.contains("rule"));
        assert!(out.contains("end"));
    }

    #[test]
    fn clean_body_decodes_and_strips_padding_entities() {
        // Real shape of the bug: CHANEL-style preheader filler that showed up
        // as literal "&zwnj;" text in stored snippets.
        let input = "Where being exceptional is the first rule &zwnj; &zwnj; &zwnj; &zwnj;";
        let out = clean_body(input);
        assert!(!out.contains("&zwnj;"), "raw entity leaked through: {out}");
        assert!(!out.contains('\u{200C}'), "decoded zero-width char not stripped: {out}");
        assert!(out.contains("exceptional"));
    }

    #[test]
    fn clean_body_preserves_angle_bracket_urls() {
        // Guard against ever swapping in an HTML-tag-stripping decoder here —
        // plain-text mail commonly wraps links in angle brackets.
        let input = "View this email in your browser <https://example.com/a?x=1&amp;y=2>.";
        let out = clean_body(input);
        assert!(
            out.contains("<https://example.com/a?x=1&y=2>"),
            "angle-bracketed URL was mangled: {out}"
        );
    }

    #[test]
    fn snippet_decodes_entities_and_strips_css() {
        // CSS block on its own line (mirrors real marketing-email shape, and
        // strip_css's backward selector-scan relies on a `\n` boundary to know
        // where prose ends — see strip_css_removes_inline_rules).
        let raw = format!(
            "From: Alice <alice@example.com>\r\n\
To: Bob <bob@example.com>\r\n\
Subject: Hello\r\n\
Date: Mon, 1 Jan 2024 12:00:00 +0000\r\n\
Message-ID: <abc123@example.com>\r\n\
\r\n\
Where being exceptional is the first rule &zwnj; &zwnj;\r\n\
.selector {{ color: red; important: yes; }}\r\n\
Time of sign-in today.\r\n"
        );
        let parsed = parse_rfc822(raw.as_bytes()).unwrap();
        let s = parsed.snippet();
        assert!(!s.contains("&zwnj;"), "entity leaked into snippet: {s}");
        assert!(!s.contains("color: red"), "CSS leaked into snippet: {s}");
        assert!(s.contains("exceptional"));
        assert!(s.contains("Time of sign-in"));
    }
}
