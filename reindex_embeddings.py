#!/usr/bin/env python3
"""
Re-index existing emails with corrected embeddings that include sender names.
This fixes the "Handshake email not found" issue.
"""

import os
from pathlib import Path
from supabase import create_client
from sentence_transformers import SentenceTransformer
import time

# Load environment variables from .env file
def load_env_file():
    """Load .env file from backend directory"""
    env_path = Path(__file__).parent / "packages" / "backend" / ".env"
    if env_path.exists():
        print(f"📁 Loading .env from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    else:
        print(f"⚠️  Warning: .env file not found at {env_path}")

load_env_file()

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_PUBLIC_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Supabase credentials not found")
    print("Set SUPABASE_PUBLIC_URL and SERVICE_ROLE environment variables")
    exit(1)

# Normalize URL
if not SUPABASE_URL.startswith("http"):
    SUPABASE_URL = "https://" + SUPABASE_URL
if SUPABASE_URL.endswith("/"):
    SUPABASE_URL = SUPABASE_URL[:-1]

print(f"🔗 Connecting to Supabase: {SUPABASE_URL}")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🤖 Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded")

def create_enhanced_embedding_text(subject: str, content: str, sender: str) -> str:
    """Create enhanced text for embedding - FIXED VERSION with full sender name"""
    # Extract domain from sender for context
    sender_domain = ""
    if sender and "@" in sender:
        try:
            sender_domain = sender.split("@")[-1].split(">")[0].strip()
        except:
            pass
    
    # Add context based on sender domain
    context_keywords = []
    if "yahoo" in sender_domain.lower():
        context_keywords.extend(["yahoo", "yahoo sports", "fantasy sports"])
    if "espn" in sender_domain.lower():
        context_keywords.extend(["espn", "sports news", "fantasy"])
    if "nfl" in sender_domain.lower():
        context_keywords.extend(["nfl", "football", "fantasy football"])
    
    # Extract keywords from subject and content
    subject_lower = subject.lower() if subject else ""
    content_lower = content.lower() if content else ""
    
    if "fantasy" in subject_lower or "fantasy" in content_lower:
        context_keywords.extend(["fantasy sports", "fantasy football", "fantasy news"])
    if "update" in subject_lower or "news" in subject_lower:
        context_keywords.extend(["news", "update", "announcement", "breaking"])
    
    # Build enhanced text - FIXED: Include full sender name
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if sender:
        parts.append(f"From: {sender}")  # ← FIXED: Full sender with name
    elif sender_domain:
        parts.append(f"From: {sender_domain}")
    if context_keywords:
        parts.append(f"Keywords: {' '.join(set(context_keywords))}")
    if content:
        content_truncated = content[:6000] + "..." if len(content) > 6000 else content
        parts.append(f"Content: {content_truncated}")
    
    return "\n".join(parts)


def reindex_emails(limit=None, specific_sender=None):
    """
    Re-index emails with corrected embeddings.
    
    Args:
        limit: Max number of emails to reindex (None = all)
        specific_sender: Only reindex emails from this sender (e.g., "handshake")
    """
    print("\n" + "="*70)
    print("RE-INDEXING EMAILS WITH FIXED EMBEDDINGS")
    print("="*70)
    
    # Build query
    query = sb.table("emails").select("id, subject, content, sender, from_name, from_email")
    
    if specific_sender:
        query = query.or_(f"sender.ilike.%{specific_sender}%,from_name.ilike.%{specific_sender}%,from_email.ilike.%{specific_sender}%")
        print(f"🔍 Filtering for sender: {specific_sender}")
    
    if limit:
        query = query.limit(limit)
        print(f"📊 Limiting to {limit} emails")
    else:
        print(f"📊 Re-indexing ALL emails (this may take a while...)")
    
    print("\n⏳ Fetching emails from database...")
    result = query.execute()
    emails = result.data if result.data else []
    
    if not emails:
        print("❌ No emails found to re-index")
        return
    
    print(f"✅ Found {len(emails)} emails to re-index\n")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, email in enumerate(emails, 1):
        email_id = email.get('id')
        subject = email.get('subject') or ''
        content = email.get('content') or ''
        sender = email.get('sender') or email.get('from_name') or email.get('from_email') or ''
        
        # Show progress
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/{len(emails)} - {subject[:50]}...")
        
        try:
            # Create enhanced text with FIXED sender inclusion
            enhanced_text = create_enhanced_embedding_text(subject, content, sender)
            
            # Generate new embedding
            embedding = model.encode(enhanced_text, normalize_embeddings=True)
            
            # Update in database
            sb.table("emails").update({
                "embedding": embedding.tolist()
            }).eq("id", email_id).execute()
            
            updated_count += 1
            
            # Rate limiting to avoid overwhelming the database
            if i % 50 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error updating email {email_id}: {e}")
            error_count += 1
            continue
    
    print("\n" + "="*70)
    print("RE-INDEXING COMPLETE")
    print("="*70)
    print(f"✅ Successfully updated: {updated_count} emails")
    print(f"⚠️  Skipped: {skipped_count} emails")
    print(f"❌ Errors: {error_count} emails")
    
    if updated_count > 0:
        print("\n🎉 Your emails now include sender names in embeddings!")
        print("   Try your search again: 'handshake delaying payments'")


if __name__ == "__main__":
    import sys
    
    print("\n🔧 Email Embedding Re-indexer")
    print("This fixes the missing sender name issue in embeddings\n")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            print("⚠️  WARNING: This will re-index ALL emails in your database!")
            response = input("Continue? (yes/no): ")
            if response.lower() == 'yes':
                reindex_emails()
            else:
                print("❌ Cancelled")
        elif sys.argv[1] == "--sender":
            if len(sys.argv) > 2:
                sender = sys.argv[2]
                reindex_emails(specific_sender=sender)
            else:
                print("Usage: python reindex_embeddings.py --sender <sender_name>")
        elif sys.argv[1] == "--limit":
            if len(sys.argv) > 2:
                limit = int(sys.argv[2])
                reindex_emails(limit=limit)
            else:
                print("Usage: python reindex_embeddings.py --limit <number>")
        else:
            print("Unknown option. Usage:")
            print("  python reindex_embeddings.py --sender handshake   # Re-index Handshake emails only")
            print("  python reindex_embeddings.py --limit 100          # Re-index first 100 emails")
            print("  python reindex_embeddings.py --all                # Re-index ALL emails")
    else:
        # Default: Re-index just Handshake emails
        print("🎯 Quick fix: Re-indexing only Handshake emails")
        print("   (Use --all to re-index everything)\n")
        reindex_emails(specific_sender="handshake")

