pub mod migrations;
pub mod queries;

use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::OpenFlags;

use crate::error::{AppError, AppResult};

pub type SqlitePool = Pool<SqliteConnectionManager>;
pub type SqliteConn = PooledConnection<SqliteConnectionManager>;

/// Wrapper around the connection pool plus a coarse write lock so we never have
/// two writers attempting to mutate SQLite at once. SQLite handles this with
/// busy-timeouts, but a Mutex makes it explicit and gives clearer errors.
pub struct Database {
    pool: SqlitePool,
    write_lock: Mutex<()>,
    pub path: PathBuf,
}

impl Database {
    pub fn open(path: &Path) -> AppResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let manager = SqliteConnectionManager::file(path)
            .with_flags(OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE)
            .with_init(|conn| {
                conn.pragma_update(None, "journal_mode", "WAL")?;
                conn.pragma_update(None, "synchronous", "NORMAL")?;
                conn.pragma_update(None, "foreign_keys", "ON")?;
                conn.busy_timeout(std::time::Duration::from_secs(5))?;
                Ok(())
            });
        let pool = Pool::builder()
            .max_size(8)
            .build(manager)
            .map_err(AppError::from)?;

        let mut conn = pool.get()?;
        migrations::run(&mut conn)?;
        Ok(Self {
            pool,
            write_lock: Mutex::new(()),
            path: path.to_path_buf(),
        })
    }

    /// Get a read connection from the pool. Multiple read connections may run in parallel.
    pub fn read(&self) -> AppResult<SqliteConn> {
        Ok(self.pool.get()?)
    }

    /// Acquire the global write lock and return a connection. Ensures only one
    /// writer runs at a time (and blocks until the previous writer drops the guard).
    pub fn write(&self) -> AppResult<WriteHandle<'_>> {
        let guard = self.write_lock.lock();
        let conn = self.pool.get()?;
        Ok(WriteHandle {
            conn,
            _guard: guard,
        })
    }
}

pub struct WriteHandle<'a> {
    pub conn: SqliteConn,
    _guard: parking_lot::MutexGuard<'a, ()>,
}

impl<'a> std::ops::Deref for WriteHandle<'a> {
    type Target = SqliteConn;
    fn deref(&self) -> &Self::Target {
        &self.conn
    }
}

impl<'a> std::ops::DerefMut for WriteHandle<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.conn
    }
}
