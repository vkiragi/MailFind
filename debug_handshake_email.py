#!/usr/bin/env python3
"""
Debug script to find the Handshake email in your database.
This will help us understand why the unified search isn't finding it.
"""

import os
from pathlib import Path
from supabase import create_client

# Load environment variables from .env file
def load_env_file():
    """Load .env file from backend directory"""
    env_path = Path(__file__).parent / "packages" / "backend" / ".env"
    if env_path.exists():
        print(f"Loading .env from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    else:
        print(f"Warning: .env file not found at {env_path}")

load_env_file()

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_PUBLIC_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Supabase credentials not found in environment")
    print("Set SUPABASE_PUBLIC_URL and SERVICE_ROLE environment variables")
    exit(1)

# Normalize URL
if not SUPABASE_URL.startswith("http"):
    SUPABASE_URL = "https://" + SUPABASE_URL
if SUPABASE_URL.endswith("/"):
    SUPABASE_URL = SUPABASE_URL[:-1]

print(f"🔍 Connecting to Supabase: {SUPABASE_URL}")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

print("\n" + "="*70)
print("SEARCH 1: Emails with 'handshake' in sender")
print("="*70)

try:
    result = sb.table("emails").select("subject, sender, created_at, content").ilike("sender", "%handshake%").limit(10).execute()
    emails = result.data if result.data else []
    
    if emails:
        print(f"✅ Found {len(emails)} emails from Handshake:\n")
        for i, email in enumerate(emails, 1):
            print(f"{i}. Subject: {email.get('subject', 'N/A')}")
            print(f"   Sender: {email.get('sender', 'N/A')}")
            print(f"   Date: {email.get('created_at', 'N/A')}")
            content = email.get('content', '')
            has_content = len(content) > 0 if content else False
            print(f"   Has Content: {has_content} ({len(content) if content else 0} chars)")
            
            # Check for payment-related keywords
            if content:
                content_lower = content.lower()
                keywords = ['payment', 'delay', 'delayed', 'processing', 'federal', 'holiday']
                found_keywords = [kw for kw in keywords if kw in content_lower]
                if found_keywords:
                    print(f"   Payment Keywords: {', '.join(found_keywords)}")
            print()
    else:
        print("❌ No emails found with 'handshake' in sender")
except Exception as e:
    print(f"❌ Error: {e}")

print("="*70)
print("SEARCH 2: Emails with 'handshake' in subject")
print("="*70)

try:
    result = sb.table("emails").select("subject, sender, created_at, content").ilike("subject", "%handshake%").limit(10).execute()
    emails = result.data if result.data else []
    
    if emails:
        print(f"✅ Found {len(emails)} emails:\n")
        for i, email in enumerate(emails, 1):
            print(f"{i}. Subject: {email.get('subject', 'N/A')}")
            print(f"   Sender: {email.get('sender', 'N/A')}")
            print(f"   Date: {email.get('created_at', 'N/A')}")
            print()
    else:
        print("❌ No emails found with 'handshake' in subject")
except Exception as e:
    print(f"❌ Error: {e}")

print("="*70)
print("SEARCH 3: Emails with 'payment' AND 'delay' in content")
print("="*70)

try:
    result = sb.table("emails").select("subject, sender, created_at, content").ilike("content", "%payment%").ilike("content", "%delay%").limit(10).execute()
    emails = result.data if result.data else []
    
    if emails:
        print(f"✅ Found {len(emails)} emails:\n")
        for i, email in enumerate(emails, 1):
            print(f"{i}. Subject: {email.get('subject', 'N/A')}")
            print(f"   Sender: {email.get('sender', 'N/A')}")
            print(f"   Date: {email.get('created_at', 'N/A')}")
            
            # Show snippet
            content = email.get('content', '')
            if content:
                # Find context around "payment" or "delay"
                words = content.lower().split()
                for idx, word in enumerate(words):
                    if 'payment' in word or 'delay' in word:
                        start = max(0, idx - 5)
                        end = min(len(words), idx + 6)
                        snippet = ' '.join(words[start:end])
                        print(f"   Snippet: ...{snippet}...")
                        break
            print()
    else:
        print("❌ No emails found with both 'payment' and 'delay' in content")
except Exception as e:
    print(f"❌ Error: {e}")

print("="*70)
print("SEARCH 4: Recent emails from November 11, 2025")
print("="*70)

try:
    result = sb.table("emails").select("subject, sender, created_at").gte("created_at", "2025-11-11T00:00:00").lte("created_at", "2025-11-11T23:59:59").order("created_at", desc=True).limit(20).execute()
    emails = result.data if result.data else []
    
    if emails:
        print(f"✅ Found {len(emails)} emails from Nov 11:\n")
        for i, email in enumerate(emails, 1):
            print(f"{i}. Subject: {email.get('subject', 'N/A')}")
            print(f"   Sender: {email.get('sender', 'N/A')}")
            print(f"   Date: {email.get('created_at', 'N/A')}")
            print()
    else:
        print("❌ No emails found from November 11, 2025")
        print("   The email might not be synced yet!")
except Exception as e:
    print(f"❌ Error: {e}")

print("="*70)
print("DIAGNOSIS")
print("="*70)

print("""
If no results above:
1. ❌ The Handshake email is NOT in your database yet
   → You need to sync your emails first
   → Run: POST /sync-inbox with appropriate date range

If results found BUT no content:
2. ⚠️  The email was synced WITHOUT content
   → The keyword search can't find it
   → You need to re-sync with content enabled

If results found WITH content:
3. ✅ The email is there!
   → The search algorithm might need threshold adjustments
   → Try lowering match_threshold in unified-search endpoint
""")

print("\n💡 Next Steps:")
print("1. If email not found: Sync your inbox to get Nov 11 emails")
print("2. If email found but no content: Re-sync with content extraction")
print("3. If email found with content: Adjust search thresholds")

