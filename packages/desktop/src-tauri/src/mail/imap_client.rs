//! Minimal blocking IMAP client. Wraps the `imap` crate with the small set of
//! operations the rest of the app needs: connect, list mailboxes, fetch
//! recent UIDs, fetch a single message body. The blocking client is wrapped
//! with `tokio::task::spawn_blocking` at call sites so it doesn't stall the
//! async runtime.

use imap::types::Fetch;
use imap::Session;
use native_tls::TlsConnector;

use crate::error::{AppError, AppResult};

pub struct ImapConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
}

pub struct MailboxSummary {
    pub name: String,
    pub uid_validity: Option<u32>,
    pub uid_next: Option<u32>,
    pub messages: u32,
}

pub struct FetchedMessage {
    pub uid: u32,
    pub raw: Vec<u8>,
}

type ImapSession = Session<native_tls::TlsStream<std::net::TcpStream>>;

pub fn connect(cfg: &ImapConfig) -> AppResult<ImapSession> {
    let tls = TlsConnector::builder()
        .build()
        .map_err(|e| AppError::Imap(format!("tls: {e}")))?;
    let client = imap::connect((cfg.host.as_str(), cfg.port), &cfg.host, &tls)
        .map_err(|e| AppError::Imap(format!("connect: {e}")))?;
    let session = client
        .login(&cfg.user, &cfg.password)
        .map_err(|(e, _)| AppError::Imap(format!("login: {e}")))?;
    Ok(session)
}

pub fn list_mailboxes(session: &mut ImapSession) -> AppResult<Vec<MailboxSummary>> {
    let names = session
        .list(Some(""), Some("*"))
        .map_err(|e| AppError::Imap(format!("list: {e}")))?;
    let mut out = Vec::with_capacity(names.len());
    for entry in names.iter() {
        let name = entry.name().to_string();
        match session.examine(&name) {
            Ok(box_state) => out.push(MailboxSummary {
                name,
                uid_validity: box_state.uid_validity,
                uid_next: box_state.uid_next,
                messages: box_state.exists,
            }),
            Err(e) => {
                tracing::warn!(?e, name, "examine failed; skipping");
            }
        }
    }
    Ok(out)
}

/// Fetch the most recent `limit` messages in `mailbox`, returning their UID and
/// raw RFC822 bytes. Used for the simple "pull recent mail" sync path. A real
/// product would track per-mailbox UIDNEXT for incremental sync; we keep that
/// state in the `mailboxes` table and pass `since_uid` to limit the range.
pub fn fetch_recent(
    session: &mut ImapSession,
    mailbox: &str,
    since_uid: Option<u32>,
    limit: u32,
) -> AppResult<Vec<FetchedMessage>> {
    let state = session
        .select(mailbox)
        .map_err(|e| AppError::Imap(format!("select: {e}")))?;

    let uid_next = state.uid_next.unwrap_or(1);
    if uid_next <= 1 {
        return Ok(vec![]);
    }
    let lower = match since_uid {
        Some(u) => u.saturating_add(1),
        None => uid_next.saturating_sub(limit),
    };
    let upper = uid_next.saturating_sub(1);
    if lower > upper {
        return Ok(vec![]);
    }

    let range = format!("{}:{}", lower, upper);
    let fetches = session
        .uid_fetch(&range, "(UID RFC822)")
        .map_err(|e| AppError::Imap(format!("uid_fetch: {e}")))?;

    let mut out = Vec::new();
    for f in fetches.iter() {
        out.push(parse_fetch(f)?);
    }
    Ok(out)
}

fn parse_fetch(f: &Fetch) -> AppResult<FetchedMessage> {
    let uid = f
        .uid
        .ok_or_else(|| AppError::Imap("fetch missing UID".into()))?;
    let body = f
        .body()
        .or_else(|| f.text())
        .ok_or_else(|| AppError::Imap("fetch missing body".into()))?;
    Ok(FetchedMessage {
        uid,
        raw: body.to_vec(),
    })
}

pub fn logout(mut session: ImapSession) {
    if let Err(e) = session.logout() {
        tracing::warn!(?e, "imap logout error");
    }
}
