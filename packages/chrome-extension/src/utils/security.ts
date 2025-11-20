/**
 * Security utilities for password-protected encryption
 *
 * This module provides:
 * - Password-based key derivation (PBKDF2)
 * - Encrypted key storage
 * - Session management with auto-lock
 * - Unlock/lock functionality
 */

const STORAGE_KEYS = {
  ENCRYPTED_KEY: 'mailfind_encrypted_key',
  KEY_SALT: 'mailfind_key_salt',
  LOCK_TIMEOUT: 'mailfind_lock_timeout',
  PASSWORD_HINT: 'mailfind_password_hint',
  LAST_ACTIVITY: 'mailfind_last_activity',
  SESSION_KEY: 'mailfind_session_key' // Session storage key (persists across panel opens/closes)
}

const PBKDF2_ITERATIONS = 100000 // Strong iteration count
const AES_KEY_LENGTH = 256
const SALT_LENGTH = 16

// Auto-lock timer
let autoLockTimer: number | null = null

/**
 * Store decrypted key in session storage (persists across panel opens/closes)
 */
async function storeSessionKey(key: CryptoKey): Promise<void> {
  try {
    const rawKey = await crypto.subtle.exportKey('raw', key)
    const keyArray = new Uint8Array(rawKey)
    const keyBase64 = btoa(String.fromCharCode(...keyArray))
    
    await chrome.storage.session.set({
      [STORAGE_KEYS.SESSION_KEY]: keyBase64
    })
    console.log('🔐 [Security] Session key stored in chrome.storage.session')
  } catch (error) {
    console.error('❌ [Security] Failed to store session key:', error)
    throw error
  }
}

/**
 * Retrieve decrypted key from session storage
 */
async function getSessionKey(): Promise<CryptoKey | null> {
  try {
    const result = await chrome.storage.session.get(STORAGE_KEYS.SESSION_KEY)
    if (!result[STORAGE_KEYS.SESSION_KEY]) {
      console.log('🔐 [Security] No session key found in storage')
      return null
    }

    const keyBase64 = result[STORAGE_KEYS.SESSION_KEY]
    const keyArray = Uint8Array.from(atob(keyBase64), c => c.charCodeAt(0))
    
    const key = await crypto.subtle.importKey(
      'raw',
      keyArray,
      { name: 'AES-GCM', length: AES_KEY_LENGTH },
      true,
      ['encrypt', 'decrypt']
    )
    
    console.log('🔐 [Security] Session key retrieved from chrome.storage.session')
    return key
  } catch (error) {
    console.error('❌ [Security] Failed to retrieve session key:', error)
    return null
  }
}

/**
 * Clear session key from session storage
 */
async function clearSessionKey(): Promise<void> {
  try {
    await chrome.storage.session.remove(STORAGE_KEYS.SESSION_KEY)
    console.log('🔐 [Security] Session key cleared from chrome.storage.session')
  } catch (error) {
    console.error('❌ [Security] Failed to clear session key:', error)
  }
}

/**
 * Derive a key from password using PBKDF2
 */
async function deriveKeyFromPassword(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const encoder = new TextEncoder()
  const passwordBuffer = encoder.encode(password)

  // Import password as key material
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    passwordBuffer,
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  )

  // Derive AES-GCM key from password
  const derivedKey = await crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: PBKDF2_ITERATIONS,
      hash: 'SHA-256'
    },
    keyMaterial,
    { name: 'AES-GCM', length: AES_KEY_LENGTH },
    true,
    ['encrypt', 'decrypt']
  )

  return derivedKey
}

/**
 * Generate a new encryption key and encrypt it with password
 */
export async function setupPasswordProtection(password: string, hint?: string): Promise<void> {
  try {
    console.log('🔐 [Security] Starting password protection setup...')

    // Generate random salt
    const salt = crypto.getRandomValues(new Uint8Array(SALT_LENGTH))
    console.log('🔐 [Security] Generated salt')

    // Derive key from password
    const passwordKey = await deriveKeyFromPassword(password, salt)
    console.log('🔐 [Security] Derived key from password')

    // Generate the actual encryption key for emails
    const emailEncryptionKey = await crypto.subtle.generateKey(
      { name: 'AES-GCM', length: AES_KEY_LENGTH },
      true,
      ['encrypt', 'decrypt']
    )
    console.log('🔐 [Security] Generated email encryption key')

    // Export the email encryption key
    const rawEmailKey = await crypto.subtle.exportKey('raw', emailEncryptionKey)
    console.log('🔐 [Security] Exported email key')

    // Encrypt the email key with the password-derived key
    const iv = crypto.getRandomValues(new Uint8Array(12))
    const encryptedKeyBuffer = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: iv },
      passwordKey,
      rawEmailKey
    )
    console.log('🔐 [Security] Encrypted email key')

    // Store encrypted key, salt, and IV
    const encryptedKeyData = {
      encryptedKey: btoa(String.fromCharCode(...new Uint8Array(encryptedKeyBuffer))),
      iv: btoa(String.fromCharCode(...iv)),
      salt: btoa(String.fromCharCode(...salt))
    }

    console.log('🔐 [Security] Storing encrypted data in chrome.storage.local...')
    await chrome.storage.local.set({
      [STORAGE_KEYS.ENCRYPTED_KEY]: JSON.stringify(encryptedKeyData),
      [STORAGE_KEYS.KEY_SALT]: btoa(String.fromCharCode(...salt)),
      [STORAGE_KEYS.PASSWORD_HINT]: hint || '',
      [STORAGE_KEYS.LOCK_TIMEOUT]: 15 // Default 15 minutes
    })
    console.log('🔐 [Security] Stored encrypted data successfully')

    // Store in session storage (persists across panel opens/closes)
    await storeSessionKey(emailEncryptionKey)

    console.log('🔐 [Security] Password protection setup complete')
  } catch (error) {
    console.error('❌ [Security] Setup failed at step:', error)
    throw new Error(`Password protection setup failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/**
 * Check if password protection is enabled
 */
export async function isPasswordProtected(): Promise<boolean> {
  const result = await chrome.storage.local.get(STORAGE_KEYS.ENCRYPTED_KEY)
  return !!result[STORAGE_KEYS.ENCRYPTED_KEY]
}

/**
 * Check if currently unlocked (key in session storage)
 */
export async function isUnlocked(): Promise<boolean> {
  const sessionKey = await getSessionKey()
  return sessionKey !== null
}

/**
 * Unlock with password
 */
export async function unlock(password: string): Promise<boolean> {
  try {
    const result = await chrome.storage.local.get([
      STORAGE_KEYS.ENCRYPTED_KEY,
      STORAGE_KEYS.LOCK_TIMEOUT
    ])

    if (!result[STORAGE_KEYS.ENCRYPTED_KEY]) {
      throw new Error('No encrypted key found')
    }

    const encryptedKeyData = JSON.parse(result[STORAGE_KEYS.ENCRYPTED_KEY])
    const salt = Uint8Array.from(atob(encryptedKeyData.salt), c => c.charCodeAt(0))
    const iv = Uint8Array.from(atob(encryptedKeyData.iv), c => c.charCodeAt(0))
    const encryptedKey = Uint8Array.from(atob(encryptedKeyData.encryptedKey), c => c.charCodeAt(0))

    // Derive key from password
    const passwordKey = await deriveKeyFromPassword(password, salt)

    // Try to decrypt the email encryption key
    const decryptedKeyBuffer = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: iv },
      passwordKey,
      encryptedKey
    )

    // Import the decrypted key
    const emailKey = await crypto.subtle.importKey(
      'raw',
      decryptedKeyBuffer,
      { name: 'AES-GCM', length: AES_KEY_LENGTH },
      true,
      ['encrypt', 'decrypt']
    )

    // Store in session storage (persists across panel opens/closes)
    await storeSessionKey(emailKey)

    // Update last activity
    await updateLastActivity()

    // Start auto-lock timer
    const timeout = result[STORAGE_KEYS.LOCK_TIMEOUT] || 15
    startAutoLock(timeout)

    console.log('🔓 [Security] Unlocked successfully')
    return true
  } catch (error) {
    console.error('❌ [Security] Unlock failed:', error)
    return false
  }
}

/**
 * Lock (clear session key)
 */
export async function lock(): Promise<void> {
  await clearSessionKey()
  stopAutoLock()
  console.log('🔒 [Security] Locked')
}

/**
 * Get encryption key (only if unlocked)
 */
export async function getEncryptionKey(): Promise<CryptoKey | null> {
  const sessionKey = await getSessionKey()
  
  if (!sessionKey) {
    console.warn('⚠️ [Security] Attempted to get key while locked')
    return null
  }

  // Update last activity
  await updateLastActivity()

  return sessionKey
}

/**
 * Get encryption key as base64 (for API calls)
 */
export async function getEncryptionKeyAsBase64(): Promise<string | null> {
  const key = await getEncryptionKey()
  if (!key) return null

  const rawKey = await crypto.subtle.exportKey('raw', key)
  const keyArray = new Uint8Array(rawKey)
  return btoa(String.fromCharCode(...keyArray))
}

/**
 * Get password hint
 */
export async function getPasswordHint(): Promise<string | null> {
  const result = await chrome.storage.local.get(STORAGE_KEYS.PASSWORD_HINT)
  return result[STORAGE_KEYS.PASSWORD_HINT] || null
}

/**
 * Update last activity timestamp
 */
async function updateLastActivity(): Promise<void> {
  await chrome.storage.local.set({
    [STORAGE_KEYS.LAST_ACTIVITY]: Date.now()
  })
}

/**
 * Start auto-lock timer
 */
function startAutoLock(timeoutMinutes: number): void {
  stopAutoLock()

  if (timeoutMinutes === 0) {
    console.log('🔓 [Security] Auto-lock disabled')
    return // Auto-lock disabled
  }

  const timeoutMs = timeoutMinutes * 60 * 1000

  autoLockTimer = window.setTimeout(async () => {
    console.log('⏰ [Security] Auto-lock triggered')
    await lock()
  }, timeoutMs)

  console.log(`⏱️ [Security] Auto-lock set for ${timeoutMinutes} minutes`)
}

/**
 * Stop auto-lock timer
 */
function stopAutoLock(): void {
  if (autoLockTimer !== null) {
    clearTimeout(autoLockTimer)
    autoLockTimer = null
  }
}

/**
 * Reset auto-lock timer (on activity)
 */
export async function resetAutoLock(): Promise<void> {
  const sessionKey = await getSessionKey()
  if (!sessionKey) return // Already locked

  await updateLastActivity()

  const result = await chrome.storage.local.get(STORAGE_KEYS.LOCK_TIMEOUT)
  const timeout = result[STORAGE_KEYS.LOCK_TIMEOUT] || 15
  startAutoLock(timeout)
}

/**
 * Get current lock timeout
 */
export async function getLockTimeout(): Promise<number> {
  const result = await chrome.storage.local.get(STORAGE_KEYS.LOCK_TIMEOUT)
  return result[STORAGE_KEYS.LOCK_TIMEOUT] || 15
}

/**
 * Set lock timeout
 */
export async function setLockTimeout(minutes: number): Promise<void> {
  await chrome.storage.local.set({
    [STORAGE_KEYS.LOCK_TIMEOUT]: minutes
  })

  const sessionKey = await getSessionKey()
  if (sessionKey) {
    startAutoLock(minutes)
  }
}

/**
 * Change password
 */
export async function changePassword(oldPassword: string, newPassword: string, hint?: string): Promise<boolean> {
  try {
    // First, unlock with old password to verify it
    const unlocked = await unlock(oldPassword)
    const sessionKey = await getSessionKey()
    if (!unlocked || !sessionKey) {
      return false
    }

    // Export current email key
    const rawEmailKey = await crypto.subtle.exportKey('raw', sessionKey)

    // Generate new salt for new password
    const newSalt = crypto.getRandomValues(new Uint8Array(SALT_LENGTH))
    const newPasswordKey = await deriveKeyFromPassword(newPassword, newSalt)

    // Encrypt email key with new password
    const iv = crypto.getRandomValues(new Uint8Array(12))
    const encryptedKeyBuffer = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: iv },
      newPasswordKey,
      rawEmailKey
    )

    // Store new encrypted key
    const encryptedKeyData = {
      encryptedKey: btoa(String.fromCharCode(...new Uint8Array(encryptedKeyBuffer))),
      iv: btoa(String.fromCharCode(...iv)),
      salt: btoa(String.fromCharCode(...newSalt))
    }

    await chrome.storage.local.set({
      [STORAGE_KEYS.ENCRYPTED_KEY]: JSON.stringify(encryptedKeyData),
      [STORAGE_KEYS.KEY_SALT]: btoa(String.fromCharCode(...newSalt)),
      [STORAGE_KEYS.PASSWORD_HINT]: hint || ''
    })

    console.log('🔐 [Security] Password changed successfully')
    return true
  } catch (error) {
    console.error('❌ [Security] Password change failed:', error)
    return false
  }
}

/**
 * Check if user has existing encryption (old system)
 */
export async function hasLegacyEncryption(): Promise<boolean> {
  const result = await chrome.storage.local.get('mailfind_encryption_key')
  return !!result['mailfind_encryption_key']
}

/**
 * Migrate from legacy encryption to password-protected
 */
export async function migrateLegacyKey(password: string, hint?: string): Promise<boolean> {
  try {
    console.log('🔐 [Security] Starting legacy key migration...')

    const result = await chrome.storage.local.get('mailfind_encryption_key')
    if (!result['mailfind_encryption_key']) {
      console.error('❌ [Security] No legacy key found')
      return false
    }

    console.log('🔐 [Security] Legacy key found, type:', typeof result['mailfind_encryption_key'])
    console.log('🔐 [Security] Legacy key value:', result['mailfind_encryption_key'])

    // Get legacy key - handle both string and object cases
    let legacyKeyData
    if (typeof result['mailfind_encryption_key'] === 'string') {
      console.log('🔐 [Security] Parsing legacy key as JSON string')
      legacyKeyData = JSON.parse(result['mailfind_encryption_key'])
    } else if (typeof result['mailfind_encryption_key'] === 'object') {
      console.log('🔐 [Security] Legacy key is already an object')
      legacyKeyData = result['mailfind_encryption_key']
    } else {
      throw new Error(`Unexpected legacy key type: ${typeof result['mailfind_encryption_key']}`)
    }

    console.log('🔐 [Security] Legacy key data:', legacyKeyData)
    console.log('🔐 [Security] Importing legacy key...')

    const legacyKey = await crypto.subtle.importKey(
      'jwk',
      legacyKeyData,
      { name: 'AES-GCM', length: AES_KEY_LENGTH },
      true,
      ['encrypt', 'decrypt']
    )
    console.log('🔐 [Security] Legacy key imported successfully')

    // Export it
    console.log('🔐 [Security] Exporting legacy key to raw format...')
    const rawEmailKey = await crypto.subtle.exportKey('raw', legacyKey)
    console.log('🔐 [Security] Legacy key exported')

    // Encrypt with password
    console.log('🔐 [Security] Encrypting with password...')
    const salt = crypto.getRandomValues(new Uint8Array(SALT_LENGTH))
    const passwordKey = await deriveKeyFromPassword(password, salt)
    const iv = crypto.getRandomValues(new Uint8Array(12))
    const encryptedKeyBuffer = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: iv },
      passwordKey,
      rawEmailKey
    )
    console.log('🔐 [Security] Key encrypted with password')

    // Store encrypted key
    const encryptedKeyData = {
      encryptedKey: btoa(String.fromCharCode(...new Uint8Array(encryptedKeyBuffer))),
      iv: btoa(String.fromCharCode(...iv)),
      salt: btoa(String.fromCharCode(...salt))
    }

    console.log('🔐 [Security] Saving encrypted key to storage...')
    await chrome.storage.local.set({
      [STORAGE_KEYS.ENCRYPTED_KEY]: JSON.stringify(encryptedKeyData),
      [STORAGE_KEYS.KEY_SALT]: btoa(String.fromCharCode(...salt)),
      [STORAGE_KEYS.PASSWORD_HINT]: hint || '',
      [STORAGE_KEYS.LOCK_TIMEOUT]: 15
    })
    console.log('🔐 [Security] Encrypted key saved')

    // Remove legacy key
    console.log('🔐 [Security] Removing legacy key...')
    await chrome.storage.local.remove('mailfind_encryption_key')
    console.log('🔐 [Security] Legacy key removed')

    // Store in session storage (persists across panel opens/closes)
    await storeSessionKey(legacyKey)

    console.log('🔐 [Security] Migrated legacy key to password protection')
    return true
  } catch (error) {
    console.error('❌ [Security] Migration failed:', error)
    return false
  }
}
