//! Schema migrations for the local SQLite store. These run on startup and are
//! idempotent. New schema changes should be added as additional `Migration`
//! entries with strictly increasing version numbers.

use rusqlite::{params, Connection};

use crate::error::AppResult;

struct Migration {
    version: i64,
    description: &'static str,
    sql: &'static str,
}

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        description: "initial schema",
        sql: include_str!("schema/001_init.sql"),
    },
];

pub fn run(conn: &mut Connection) -> AppResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );",
    )?;

    for m in MIGRATIONS {
        let already: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_versions WHERE version = ?1)",
                params![m.version],
                |row| row.get(0),
            )
            .unwrap_or(false);
        if already {
            continue;
        }

        let tx = conn.transaction()?;
        tx.execute_batch(m.sql)?;
        tx.execute(
            "INSERT INTO schema_versions (version, description) VALUES (?1, ?2)",
            params![m.version, m.description],
        )?;
        tx.commit()?;
    }
    Ok(())
}
