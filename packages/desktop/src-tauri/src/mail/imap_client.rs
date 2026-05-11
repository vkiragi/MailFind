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

/// Search for UIDs in `mailbox` with INTERNALDATE on or after `since_date`,
/// optionally limited to UIDs greater than `since_uid` (incremental).
/// `since_date` must be in IMAP date format: `DD-Mon-YYYY`.
pub fn search_since_date(
    session: &mut ImapSession,
    mailbox: &str,
    since_date: &str,
    since_uid: Option<u32>,
) -> AppResult<Vec<u32>> {
    session
        .select(mailbox)
        .map_err(|e| AppError::Imap(format!("select: {e}")))?;

    let mut criterion = format!("SINCE {}", since_date);
    if let Some(uid) = since_uid {
        criterion.push_str(&format!(" UID {}:*", uid.saturating_add(1)));
    }
    let uids = session
        .uid_search(&criterion)
        .map_err(|e| AppError::Imap(format!("uid_search: {e}")))?;
    let mut sorted: Vec<u32> = uids.into_iter().collect();
    sorted.sort_unstable();
    Ok(sorted)
}

/// Fetch the given UIDs in batches over a single, long-lived IMAP session.
/// iCloud throttles per-account *new* connections aggressively
/// (`[UNAVAILABLE]` responses, then cooldown), so we never reconnect — we
/// keep the existing session and pace the FETCHes with a small delay.
/// On a transient `[UNAVAILABLE]` response, we sleep 30s and retry the same
/// batch once before giving up with a friendly error.
pub fn fetch_uids_batched(
    session: &mut ImapSession,
    uids: &[u32],
    batch_size: usize,
    mut on_batch: impl FnMut(usize, usize),
) -> AppResult<Vec<FetchedMessage>> {
    use std::thread::sleep;
    use std::time::Duration;

    const INTER_BATCH_DELAY: Duration = Duration::from_millis(250);
    const RATE_LIMIT_BACKOFF: Duration = Duration::from_secs(30);

    let mut out = Vec::with_capacity(uids.len());
    let total = uids.len();

    for (i, chunk) in uids.chunks(batch_size.max(1)).enumerate() {
        if i > 0 {
            sleep(INTER_BATCH_DELAY);
        }
        let range = chunk
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let fetches = match session.uid_fetch(&range, "(UID BODY.PEEK[])") {
            Ok(f) => f,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("UNAVAILABLE") {
                    tracing::warn!(
                        ?e,
                        backoff_secs = RATE_LIMIT_BACKOFF.as_secs(),
                        "iCloud rate-limited; backing off"
                    );
                    sleep(RATE_LIMIT_BACKOFF);
                    session.uid_fetch(&range, "(UID BODY.PEEK[])").map_err(|e| {
                        AppError::RateLimited(format!(
                            "iCloud is throttling this account. Try again in 10 minutes \
                             with a smaller window. Details: {e}"
                        ))
                    })?
                } else {
                    return Err(AppError::Imap(format!("uid_fetch: {e}")));
                }
            }
        };

        for f in fetches.iter() {
            out.push(parse_fetch(f)?);
        }
        on_batch(out.len(), total);
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
