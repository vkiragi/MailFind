//! Secure credential storage backed by the OS keychain (macOS Keychain on
//! Apple platforms via the `keyring` crate). The local SQLite database only
//! stores a stable reference to the keychain entry, never the password
//! itself.

use crate::error::AppResult;

const SERVICE: &str = "com.mailfind.desktop";

pub fn store_password(keyring_ref: &str, password: &str) -> AppResult<()> {
    let entry = keyring::Entry::new(SERVICE, keyring_ref)?;
    entry.set_password(password)?;
    Ok(())
}

pub fn fetch_password(keyring_ref: &str) -> AppResult<String> {
    let entry = keyring::Entry::new(SERVICE, keyring_ref)?;
    Ok(entry.get_password()?)
}

pub fn delete_password(keyring_ref: &str) -> AppResult<()> {
    let entry = keyring::Entry::new(SERVICE, keyring_ref)?;
    if let Err(e) = entry.delete_password() {
        // Ignore not-found so removing an account is idempotent.
        if !matches!(e, keyring::Error::NoEntry) {
            return Err(e.into());
        }
    }
    Ok(())
}
