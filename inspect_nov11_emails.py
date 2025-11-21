#!/usr/bin/env python3
"""
Inspect the Nov 11 emails to see what data is actually stored.
"""

import os
from pathlib import Path
from supabase import create_client

# Load environment variables from .env file
def load_env_file():
    """Load .env file from backend directory"""
    env_path = Path(__file__).parent / "packages" / "backend" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env_file()

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_PUBLIC_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Supabase credentials not found")
    exit(1)

# Normalize URL
if not SUPABASE_URL.startswith("http"):
    SUPABASE_URL = "https://" + SUPABASE_URL
if SUPABASE_URL.endswith("/"):
    SUPABASE_URL = SUPABASE_URL[:-1]

print(f"🔍 Connecting to Supabase: {SUPABASE_URL}\n")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

print("="*80)
print("INSPECTING NOV 11 EMAILS - FULL FIELD ANALYSIS")
print("="*80)

try:
    # Get all fields for Nov 11 emails
    result = sb.table("emails").select(
        "id, subject, sender, content, from_name, from_email, "
        "encrypted_content, iv, thread_id, created_at"
    ).gte("created_at", "2025-11-11T00:00:00").lte(
        "created_at", "2025-11-11T23:59:59"
    ).order("created_at", desc=True).limit(10).execute()
    
    emails = result.data if result.data else []
    
    if not emails:
        print("❌ No emails found from Nov 11")
        exit(1)
    
    print(f"✅ Found {len(emails)} emails from Nov 11\n")
    
    for i, email in enumerate(emails, 1):
        print(f"\n{'='*80}")
        print(f"EMAIL {i}")
        print(f"{'='*80}")
        
        print(f"ID: {email.get('id', 'N/A')}")
        print(f"Thread ID: {email.get('thread_id', 'N/A')}")
        print(f"Created At: {email.get('created_at', 'N/A')}")
        print()
        
        # Check plaintext fields
        subject = email.get('subject')
        sender = email.get('sender')
        from_name = email.get('from_name')
        from_email = email.get('from_email')
        content = email.get('content')
        
        print("PLAINTEXT FIELDS:")
        print(f"  subject: {subject if subject else '❌ EMPTY'}")
        print(f"  sender: {sender if sender else '❌ EMPTY'}")
        print(f"  from_name: {from_name if from_name else '❌ EMPTY'}")
        print(f"  from_email: {from_email if from_email else '❌ EMPTY'}")
        print(f"  content: {'✅ EXISTS' if content else '❌ EMPTY'} ({len(content) if content else 0} chars)")
        print()
        
        # Check encrypted fields
        encrypted_content = email.get('encrypted_content')
        iv = email.get('iv')
        
        print("ENCRYPTED FIELDS:")
        print(f"  encrypted_content: {'✅ EXISTS' if encrypted_content else '❌ EMPTY'} ({len(encrypted_content) if encrypted_content else 0} chars)")
        print(f"  iv: {'✅ EXISTS' if iv else '❌ EMPTY'}")
        print()
        
        # Diagnosis
        has_plaintext = bool(subject or sender or content)
        has_encrypted = bool(encrypted_content and iv)
        
        print("DIAGNOSIS:")
        if has_plaintext:
            print("  ✅ Has plaintext data - searchable!")
        elif has_encrypted:
            print("  ⚠️  ONLY encrypted data - NOT searchable!")
            print("     Problem: Search uses plaintext fields, not encrypted")
            print("     Solution: Need to store decrypted data in plaintext fields too")
        else:
            print("  ❌ NO DATA AT ALL - sync failed!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    plaintext_count = sum(1 for e in emails if e.get('subject') or e.get('sender'))
    encrypted_only_count = sum(1 for e in emails if e.get('encrypted_content') and not (e.get('subject') or e.get('sender')))
    empty_count = sum(1 for e in emails if not e.get('encrypted_content') and not (e.get('subject') or e.get('sender')))
    
    print(f"✅ Emails with plaintext (searchable): {plaintext_count}")
    print(f"⚠️  Emails with ONLY encrypted data: {encrypted_only_count}")
    print(f"❌ Completely empty emails: {empty_count}")
    print()
    
    if encrypted_only_count > 0:
        print("🔧 FIX NEEDED:")
        print("Your emails are encrypted but the plaintext searchable fields are empty.")
        print("The sync process needs to:")
        print("  1. Decrypt the email content")
        print("  2. Store decrypted subject/sender/content for search")
        print("  3. Keep encrypted_content for security")
        print()
        print("Check your /sync-inbox endpoint - it should populate both!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

