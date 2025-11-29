/**
 * Zero-Knowledge Encryption Utilities
 *
 * This module provides client-side encryption for MailFind's privacy-first architecture.
 * All email content is encrypted with AES-256-GCM before being sent to the server.
 * The encryption key NEVER leaves the user's browser.
 */

const STORAGE_KEY = 'mailfind_encryption_key'
const ALGORITHM = 'AES-GCM'
const KEY_LENGTH = 256

export interface EncryptedData {
  encrypted: string  // Base64 encoded encrypted data
  iv: string         // Base64 encoded initialization vector
}

export interface EmailPayload {
  subject: string
  sender: string
  content: string
  embedding?: number[]
}

/**
 * Generate a new AES-256 encryption key
 * This should only be called once per user on first install
 */
export async function generateEncryptionKey(): Promise<CryptoKey> {
  const key = await crypto.subtle.generateKey(
    {
      name: ALGORITHM,
      length: KEY_LENGTH,
    },
    true,  // extractable (so we can export for backup)
    ['encrypt', 'decrypt']
  )
  return key
}

/**
 * Export encryption key to JWK format for storage
 */
export async function exportKey(key: CryptoKey): Promise<JsonWebKey> {
  return await crypto.subtle.exportKey('jwk', key)
}

/**
 * Import encryption key from JWK format
 */
export async function importKey(jwk: JsonWebKey): Promise<CryptoKey> {
  return await crypto.subtle.importKey(
    'jwk',
    jwk,
    {
      name: ALGORITHM,
      length: KEY_LENGTH,
    },
    true,
    ['encrypt', 'decrypt']
  )
}

/**
 * Store encryption key in Chrome extension local storage
 */
export async function storeEncryptionKey(key: CryptoKey): Promise<void> {
  const jwk = await exportKey(key)

  // Use Chrome extension storage API if available, otherwise localStorage
  if (typeof chrome !== 'undefined' && chrome.storage) {
    await chrome.storage.local.set({ [STORAGE_KEY]: jwk })
  } else {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(jwk))
  }
}

/**
 * Retrieve encryption key from storage
 * If no key exists, generate a new one
 */
export async function getEncryptionKey(): Promise<CryptoKey> {
  let jwk: JsonWebKey | null = null

  // Try to get from Chrome extension storage first
  if (typeof chrome !== 'undefined' && chrome.storage) {
    const result = await chrome.storage.local.get(STORAGE_KEY)
    jwk = result[STORAGE_KEY] || null
  } else {
    // Fallback to localStorage
    const stored = localStorage.getItem(STORAGE_KEY)
    jwk = stored ? JSON.parse(stored) : null
  }

  // If no key exists, generate a new one
  if (!jwk) {
    console.log('[Encryption] No key found, generating new encryption key')
    const key = await generateEncryptionKey()
    await storeEncryptionKey(key)
    return key
  }

  return await importKey(jwk)
}

/**
 * Encrypt data using AES-GCM
 */
export async function encrypt(data: string, key: CryptoKey): Promise<EncryptedData> {
  // Generate a random initialization vector (IV)
  const iv = crypto.getRandomValues(new Uint8Array(12))

  // Encode the data as UTF-8
  const encodedData = new TextEncoder().encode(data)

  // Encrypt the data
  const encryptedBuffer = await crypto.subtle.encrypt(
    {
      name: ALGORITHM,
      iv: iv,
    },
    key,
    encodedData
  )

  // Convert to base64 for storage
  const encryptedArray = new Uint8Array(encryptedBuffer)
  const encrypted = btoa(String.fromCharCode(...encryptedArray))
  const ivBase64 = btoa(String.fromCharCode(...iv))

  return {
    encrypted,
    iv: ivBase64,
  }
}

/**
 * Decrypt data using AES-GCM
 */
export async function decrypt(encryptedData: EncryptedData, key: CryptoKey): Promise<string> {
  // Decode from base64
  const encrypted = Uint8Array.from(atob(encryptedData.encrypted), c => c.charCodeAt(0))
  const iv = Uint8Array.from(atob(encryptedData.iv), c => c.charCodeAt(0))

  // Decrypt the data
  const decryptedBuffer = await crypto.subtle.decrypt(
    {
      name: ALGORITHM,
      iv: iv,
    },
    key,
    encrypted
  )

  // Decode as UTF-8
  return new TextDecoder().decode(decryptedBuffer)
}

/**
 * Encrypt an email payload (subject, sender, content)
 */
export async function encryptEmail(email: EmailPayload): Promise<EncryptedData> {
  const key = await getEncryptionKey()
  const json = JSON.stringify(email)
  return await encrypt(json, key)
}

/**
 * Decrypt an email payload
 */
export async function decryptEmail(encryptedData: EncryptedData): Promise<EmailPayload> {
  const key = await getEncryptionKey()
  const json = await decrypt(encryptedData, key)
  return JSON.parse(json)
}

/**
 * Encrypt embedding vector
 */
export async function encryptEmbedding(embedding: number[]): Promise<EncryptedData> {
  const key = await getEncryptionKey()
  const json = JSON.stringify(embedding)
  return await encrypt(json, key)
}

/**
 * Decrypt embedding vector
 */
export async function decryptEmbedding(encryptedData: EncryptedData): Promise<number[]> {
  const key = await getEncryptionKey()
  const json = await decrypt(encryptedData, key)
  return JSON.parse(json)
}

/**
 * Hash thread ID using SHA-256 (one-way hash for deduplication)
 */
export async function hashThreadId(threadId: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(threadId)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hashArray = new Uint8Array(hashBuffer)
  return Array.from(hashArray)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('')
}

/**
 * Extract domain from email address
 * e.g., "john.doe@github.com" -> "github.com"
 */
export function extractDomain(email: string): string {
  const match = email.match(/@(.+)$/)
  return match ? match[1].toLowerCase() : ''
}

/**
 * Export encryption key as downloadable backup
 */
export async function downloadKeyBackup(): Promise<void> {
  const key = await getEncryptionKey()
  const jwk = await exportKey(key)

  const blob = new Blob([JSON.stringify(jwk, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)

  const a = document.createElement('a')
  a.href = url
  a.download = `mailfind-encryption-key-${Date.now()}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Import encryption key from backup file
 */
export async function importKeyFromBackup(file: File): Promise<void> {
  const text = await file.text()
  const jwk = JSON.parse(text)
  const key = await importKey(jwk)
  await storeEncryptionKey(key)
}

/**
 * Check if encryption is enabled (key exists)
 */
export async function isEncryptionEnabled(): Promise<boolean> {
  if (typeof chrome !== 'undefined' && chrome.storage) {
    const result = await chrome.storage.local.get(STORAGE_KEY)
    return !!result[STORAGE_KEY]
  } else {
    return !!localStorage.getItem(STORAGE_KEY)
  }
}

/**
 * Get encryption key as base64 for sending to backend
 * Backend uses this transiently for encrypt/decrypt operations (zero-knowledge)
 */
export async function getEncryptionKeyAsBase64(): Promise<string> {
  const key = await getEncryptionKey()
  const rawKey = await crypto.subtle.exportKey('raw', key)
  const keyArray = new Uint8Array(rawKey)
  return btoa(String.fromCharCode(...keyArray))
}
