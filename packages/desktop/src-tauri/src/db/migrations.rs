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
    Migration {
        version: 2,
        description: "perf indexes for status counters",
        sql: include_str!("schema/002_perf_indexes.sql"),
    },
    Migration {
        version: 3,
        description: "default chat model to qwen2.5:7b-instruct",
        sql: include_str!("schema/003_default_chat_model.sql"),
    },
    Migration {
        version: 4,
        description: "default chat model to qwen3:8b",
        sql: include_str!("schema/004_default_chat_model_qwen3.sql"),
    },
    Migration {
        version: 5,
        description: "mark chat model source auto for RAM-based auto-pick",
        sql: include_str!("schema/005_chat_model_source.sql"),
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
