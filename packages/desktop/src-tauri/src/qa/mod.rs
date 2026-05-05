//! Q&A over retrieved email chunks. Mirrors the spirit of the prior backend's
//! `_prepare_email_context` + chat-with-emails flow in
//! `packages/backend/main.py`, but with citations that link back to local
//! message IDs instead of Gmail URLs.

pub mod answer;

pub use answer::{ask, AnswerCitation, AnswerOutcome};
