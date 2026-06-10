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
        compact_text(body, 200)
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
}
