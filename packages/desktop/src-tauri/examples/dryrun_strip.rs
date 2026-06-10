//! Dry-run the CSS stripper against the live DB without modifying anything.
//! Prints before/after length and a small sample.

use mailfind_lib::db::Database;
use mailfind_lib::mail::parser::strip_css;
use mailfind_lib::state;

fn main() {
    let data_dir = state::default_data_dir().expect("data dir");
    let db = Database::open(&data_dir.join("mailfind.sqlite")).expect("open");
    let conn = db.read().expect("read");

    let mut stmt = conn
        .prepare(
            "SELECT id, subject, sender_email, COALESCE(body_plain,'') \
             FROM messages \
             WHERE LOWER(COALESCE(body_plain,'')) LIKE '%!important%' \
             LIMIT 5",
        )
        .expect("prepare");
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1).unwrap_or_default(),
                r.get::<_, String>(2).unwrap_or_default(),
                r.get::<_, String>(3)?,
            ))
        })
        .unwrap();
    for row in rows.flatten() {
        let (_id, subj, sender, body) = row;
        let cleaned = strip_css(&body);
        println!("--- {sender} | {subj} ---");
        println!("before: {} chars / !important count: {}", body.len(), body.matches("!important").count());
        println!(" after: {} chars / !important count: {}", cleaned.len(), cleaned.matches("!important").count());
        let sample: String = cleaned.chars().take(180).collect();
        println!("sample: {sample}\n");
    }
}
