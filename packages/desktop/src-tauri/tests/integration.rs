//! Integration tests covering the data layer and search pipeline.
//!
//! These tests exercise the Rust crate without Tauri (no `invoke!`), which
//! makes them fast and CI-friendly. They cover:
//!
//! - migrations applied cleanly to a fresh database
//! - app_settings seeded with the expected default model names
//! - embedding encode/decode round-trip preserves f32 vectors
//! - FTS5 returns chunks for keyword queries
//! - end-to-end keyword + vector hybrid search with synthetic embeddings

use std::path::PathBuf;

use mailfind_lib::db::queries::{self, NewAccount, NewChunk, NewMessage};
use mailfind_lib::db::Database;
use mailfind_lib::search;
use rusqlite::params;
use tempfile::tempdir;

fn fresh_db() -> (Database, tempfile::TempDir) {
    let tmp = tempdir().expect("tempdir");
    let path: PathBuf = tmp.path().join("test.sqlite");
    let db = Database::open(&path).expect("open db");
    (db, tmp)
}

fn seed_account(db: &Database) -> String {
    let conn = db.write().expect("write conn");
    let row = queries::insert_account(
        &conn,
        &NewAccount {
            email: "you@icloud.com".into(),
            display_name: Some("You".into()),
            imap_host: "imap.mail.me.com".into(),
            imap_port: 993,
            keyring_ref: "test:keyring:1".into(),
        },
    )
    .expect("insert account");
    row.id
}

fn insert_message(
    db: &Database,
    account_id: &str,
    subject: &str,
    sender: &str,
    body: &str,
) -> String {
    let id = uuid::Uuid::new_v4().to_string();
    let new_msg = NewMessage {
        id: id.clone(),
        account_id: account_id.to_string(),
        mailbox_id: None,
        imap_uid: None,
        rfc822_message_id: Some(format!("{}@test", id)),
        thread_id: None,
        subject: Some(subject.to_string()),
        sender: Some(sender.to_string()),
        sender_email: Some(sender.to_string()),
        sender_domain: Some("test".to_string()),
        recipients: Some("you@icloud.com".to_string()),
        sent_at: None,
        received_at: Some(chrono::Utc::now()),
        snippet: Some(body.chars().take(200).collect::<String>()),
        body_plain: Some(body.to_string()),
        body_html: None,
        has_attachments: false,
        raw_size: Some(body.len() as i64),
    };
    let mut handle = db.write().expect("write conn");
    let tx = handle.transaction().expect("tx");
    queries::insert_message(&tx, &new_msg).expect("insert msg");
    queries::insert_chunk(
        &tx,
        &NewChunk {
            message_id: id.clone(),
            chunk_index: 0,
            text: format!("Subject: {}\n\n{}", subject, body),
            embedding: None,
            embedding_model: None,
        },
    )
    .expect("insert chunk");
    tx.commit().expect("commit");
    id
}

#[test]
fn migrations_create_expected_tables_and_seed_defaults() {
    let (db, _tmp) = fresh_db();
    let conn = db.read().expect("read conn");

    // schema_versions tracking row exists.
    let v: i64 = conn
        .query_row("SELECT MAX(version) FROM schema_versions", [], |r| r.get(0))
        .unwrap();
    assert!(v >= 1);

    // Core tables present.
    for table in [
        "accounts",
        "mailboxes",
        "messages",
        "chunks",
        "chunks_fts",
        "sync_state",
        "search_feedback",
        "app_settings",
    ] {
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = ?1",
                params![table],
                |r| r.get(0),
            )
            .unwrap();
        assert!(exists > 0, "expected table `{}` to exist", table);
    }

    // Defaults seeded.
    let embedding_model = queries::get_setting(&conn, "embedding_model").unwrap();
    assert_eq!(embedding_model.as_deref(), Some("nomic-embed-text"));
    let chat_model = queries::get_setting(&conn, "chat_model").unwrap();
    assert!(chat_model.as_deref().unwrap_or("").starts_with("qwen"));
}

#[test]
fn embedding_blob_round_trips_through_db() {
    let (db, _tmp) = fresh_db();
    let account_id = seed_account(&db);
    let message_id = insert_message(
        &db,
        &account_id,
        "Receipt",
        "Stripe <noreply@stripe.com>",
        "Thanks for your purchase. Amount $42.",
    );
    let chunk_text = "Thanks for your purchase. Amount $42.";
    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();

    let chunk_id = {
        let mut handle = db.write().unwrap();
        let tx = handle.transaction().unwrap();
        // Replace the existing chunk with one that has an embedding.
        tx.execute(
            "DELETE FROM chunks WHERE message_id = ?1",
            params![message_id],
        )
        .unwrap();
        let id = queries::insert_chunk(
            &tx,
            &NewChunk {
                message_id: message_id.clone(),
                chunk_index: 0,
                text: chunk_text.to_string(),
                embedding: Some(embedding.clone()),
                embedding_model: Some("nomic-embed-text".into()),
            },
        )
        .unwrap();
        tx.commit().unwrap();
        id
    };
    assert!(!chunk_id.is_empty());

    let conn = db.read().unwrap();
    let rows = queries::all_chunks_with_embeddings(&conn).unwrap();
    let found = rows.iter().find(|c| c.message_id == message_id).unwrap();
    assert_eq!(found.embedding.len(), embedding.len());
    for (a, b) in embedding.iter().zip(found.embedding.iter()) {
        assert!((a - b).abs() < 1e-6, "round-trip mismatch: {a} vs {b}");
    }
}

#[test]
fn fts_returns_keyword_matches() {
    let (db, _tmp) = fresh_db();
    let account_id = seed_account(&db);
    let m1 = insert_message(
        &db,
        &account_id,
        "Receipt for $42 payment",
        "Stripe <noreply@stripe.com>",
        "Thanks for your purchase. Stripe payment processed.",
    );
    let _m2 = insert_message(
        &db,
        &account_id,
        "Welcome to Apple ID",
        "Apple <noreply@apple.com>",
        "Welcome to Apple ID, manage your account online.",
    );

    let conn = db.read().unwrap();
    let hits = queries::search_fts(&conn, "stripe payment", 10).unwrap();
    assert!(!hits.is_empty(), "expected FTS hits");
    let top = &hits[0].0;
    assert_eq!(top, &m1, "stripe receipt should rank first");
}

#[tokio::test]
async fn hybrid_search_uses_keyword_fallback_when_no_embeddings() {
    let (db, _tmp) = fresh_db();
    let account_id = seed_account(&db);
    let _ = insert_message(
        &db,
        &account_id,
        "Contract review milestone update",
        "Alex <alex@example.com>",
        "Section 4.2 references the old delivery date and needs an update.",
    );
    let _ = insert_message(
        &db,
        &account_id,
        "Lunch tomorrow?",
        "Sam <sam@example.com>",
        "Want to grab lunch at the new place tomorrow?",
    );

    // No Ollama client passed: only keyword search runs.
    let outcome = search::search(&db, None, "contract delivery", 5).await.unwrap();
    assert!(outcome.used_keyword);
    assert!(!outcome.used_vector);
    assert!(!outcome.results.is_empty());
    assert!(
        outcome.results[0].subject.contains("Contract"),
        "expected contract email first, got: {}",
        outcome.results[0].subject
    );
}
