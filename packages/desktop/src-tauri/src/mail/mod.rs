//! Mail subsystem: parsing, IMAP sync, fixture ingestion. The domain types in
//! `parser` are shared between fixtures (`fixtures.rs`) and live IMAP fetches
//! (`imap_client.rs`), so search/retrieval doesn't care where a message came
//! from.

pub mod fixtures;
pub mod imap_client;
pub mod parser;
pub mod sync;

pub use fixtures::{ingest_path, IngestReport};
pub use parser::{parse_rfc822, ParsedMessage};
