import os
import time
from fastapi import FastAPI, HTTPException, Form, Header
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from pathlib import Path
import secrets
import json
import os
from typing import Optional
import base64
from datetime import datetime, timezone
from urllib.parse import urlparse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

# Supabase + Encryption
from supabase import create_client, Client  # type: ignore
from cryptography.fernet import Fernet  # type: ignore
import requests
from sentence_transformers import SentenceTransformer  # type: ignore
import hashlib

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Adaptive Search
from adaptive_search import AdaptiveSearchEngine, perform_adaptive_search

# Smart Search with Query Parser
try:
    from smart_search import smart_search
    SMART_SEARCH_ENABLED = True
except Exception as e:
    print(f"[Warning] Smart search not available: {e}")
    SMART_SEARCH_ENABLED = False

# Global cache for AI scores (in-memory cache with TTL)
AI_SCORE_CACHE = {}
CACHE_TTL = 900  # 15 minutes in seconds  # Will validate at runtime

# Load environment variables from .env file (local and repo root fallbacks)
load_dotenv()
try:
    root_env_path = Path(__file__).resolve().parents[2] / ".env"
    if root_env_path.exists():
        load_dotenv(dotenv_path=str(root_env_path), override=False)
except Exception:
    pass

app = FastAPI()
# Lazy global for embeddings model
_emb_model: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        try:
            _emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {str(e)}")
    return _emb_model

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://mail.google.com",
    ],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for OAuth states (temporary)
oauth_states = {}

# OAuth configuration
# Include userinfo scopes to fetch email/subject identifiers for Supabase records
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'openid',
]
REDIRECT_URI = "http://localhost:8000/callback"

# Supabase client initialization (support multiple env var names)
def _normalize_supabase_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    url = value.strip()
    if not url:
        return None
    low = url.lower()
    if not (low.startswith("http://") or low.startswith("https://")):
        url = "https://" + url
    if url.endswith("/"):
        url = url[:-1]
    return url

SUPABASE_URL = _normalize_supabase_url(
    os.getenv("SUPABASE_URL")
    or os.getenv("SUPABASE_PUBLIC_URL")
    or os.getenv("supabase_public_url")
)
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE")
    or os.getenv("SERVICE_ROLE")
    or os.getenv("service_role")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("supabase_anon_key")
)
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        print(f"[Config] Init Supabase at startup: url={SUPABASE_URL}, key_len={len(SUPABASE_KEY)}")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# Encryption key for storing tokens
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")


def _get_fernet() -> Fernet:
    if not ENCRYPTION_KEY:
        raise HTTPException(status_code=500, detail="ENCRYPTION_KEY not configured")
    try:
        return Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid ENCRYPTION_KEY: {str(e)}")


def _encrypt_dict(data: dict) -> str:
    f = _get_fernet()
    payload = json.dumps(data).encode("utf-8")
    return f.encrypt(payload).decode("utf-8")


def _decrypt_to_dict(token_str: str) -> dict:
    f = _get_fernet()
    raw = f.decrypt(token_str.encode("utf-8"))
    return json.loads(raw.decode("utf-8"))


# ===== Zero-Knowledge Encryption (User-Managed Keys) =====

def _encrypt_with_user_key(data: str, user_key_b64: str) -> tuple[str, str]:
    """
    Encrypt data with user's encryption key (zero-knowledge).
    Returns (encrypted_data_b64, iv_b64)

    The user's key never leaves the browser and is passed as a header.
    The backend uses it transiently for encryption then discards it.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os

    # Decode user's key from base64
    key = base64.b64decode(user_key_b64)

    # Generate random nonce/IV
    nonce = os.urandom(12)  # 96-bit nonce for GCM

    # Encrypt with AES-256-GCM
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data.encode('utf-8'), None)

    # Return as base64
    return (
        base64.b64encode(ciphertext).decode('utf-8'),
        base64.b64encode(nonce).decode('utf-8')
    )


def _decrypt_with_user_key(encrypted_data_b64: str, iv_b64: str, user_key_b64: str) -> str:
    """
    Decrypt data with user's encryption key.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    # Decode from base64
    key = base64.b64decode(user_key_b64)
    ciphertext = base64.b64decode(encrypted_data_b64)
    nonce = base64.b64decode(iv_b64)

    # Decrypt
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    return plaintext.decode('utf-8')


def _hash_thread_id(thread_id: str) -> str:
    """Create SHA-256 hash of thread_id for deduplication without revealing actual ID."""
    return hashlib.sha256(thread_id.encode('utf-8')).hexdigest()


def _extract_domain(email: str) -> str:
    """Extract domain from email address (e.g., 'user@example.com' -> 'example.com')."""
    if '@' in email:
        return email.split('@')[-1].lower()
    return ''


def _get_supabase() -> Client:
    """Return a Supabase client, creating it on demand from env if missing."""
    global supabase
    if supabase is not None:
        return supabase

    url_raw = (
        os.getenv("SUPABASE_URL")
        or os.getenv("SUPABASE_PUBLIC_URL")
        or os.getenv("supabase_public_url")
    )
    key = (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE")
        or os.getenv("SERVICE_ROLE")
        or os.getenv("service_role")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("supabase_anon_key")
    )
    url = _normalize_supabase_url(url_raw)
    print(f"[Config] _get_supabase: raw_url={url_raw!r}, normalized_url={url!r}, key_len={(len(key) if key else 0)}")
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    # Validate host to avoid idna errors later
    try:
        parsed = urlparse(url)
        host = parsed.netloc
        host.encode("idna")  # will raise if invalid
        if not host or (".supabase.co" not in host and ".supabase.net" not in host):
            raise HTTPException(status_code=500, detail=f"Supabase URL invalid: {url}. Expected https://<ref>.supabase.co")
    except UnicodeError as ue:
        raise HTTPException(status_code=500, detail=f"Supabase URL host invalid: {ue}")

    try:
        print(f"[Config] Creating Supabase client: url={url}, key_len={len(key)}")
        supabase = create_client(url, key)
        return supabase
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase init error: {str(e)}")

def get_flow():
    """Create OAuth2 flow"""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="OAuth credentials not configured")
    
    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI]
        }
    }
    
    flow = Flow.from_client_config(client_config, scopes=SCOPES)
    flow.redirect_uri = REDIRECT_URI
    return flow


def _get_credentials_for_user(user_id: Optional[str] = None) -> Credentials:
    """Construct Credentials from encrypted tokens stored in Supabase.

    user_id is the google_user_id. If not provided, use the first user record.
    """
    sb = _get_supabase()

    # Find user by google_user_id or take first
    if user_id:
        res = sb.table("users").select("encrypted_credentials").eq("google_user_id", user_id).limit(1).execute()  # type: ignore[attr-defined]
    else:
        res = sb.table("users").select("encrypted_credentials").limit(1).execute()  # type: ignore[attr-defined]

    rows = getattr(res, "data", []) or []
    if not rows:
        raise HTTPException(status_code=401, detail="No stored user tokens. Please login first.")

    enc = rows[0].get("encrypted_credentials")
    if not enc:
        raise HTTPException(status_code=404, detail="Encrypted credentials missing for user")

    stored = _decrypt_to_dict(enc)

    # Parse expiry if present
    expiry = None
    if stored.get("expiry"):
        from datetime import datetime
        expiry = datetime.fromisoformat(stored["expiry"])

    creds = Credentials(
        token=stored.get("token"),
        refresh_token=stored.get("refresh_token"),
        token_uri=stored.get("token_uri"),
        client_id=stored.get("client_id"),
        client_secret=stored.get("client_secret"),
        scopes=stored.get("scopes", SCOPES),
        expiry=expiry,
    )

    # Refresh if needed and persist new access token
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            updated = dict(stored)
            updated["token"] = creds.token
            updated["expiry"] = creds.expiry.isoformat() if creds.expiry else None
            try:
                # Persist refreshed token back to Supabase (best-effort)
                if user_id:
                    sb.table("users").update({"encrypted_credentials": _encrypt_dict(updated)}).eq("google_user_id", user_id).execute()  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception as e:
        # If refresh token is invalid/expired, user needs to re-authenticate
        if "invalid_grant" in str(e) or "Token has been expired or revoked" in str(e):
            raise HTTPException(
                status_code=401, 
                detail="Authentication expired. Please log out and log back in to reconnect your Google account."
            )
        raise HTTPException(status_code=401, detail=f"Failed to refresh credentials: {str(e)}")

    return creds


def _build_gmail_service(user_id: Optional[str] = None):
    """Build an authenticated Gmail API service using stored credentials."""
    creds = _get_credentials_for_user(user_id)
    try:
        # cache_discovery=False avoids file cache issues in some environments
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gmail service: {str(e)}")


def _base64url_decode(data_str: str) -> str:
    """Decode base64url-encoded Gmail body data to UTF-8 text, ignoring errors."""
    if not data_str:
        return ""
    try:
        padded = data_str + "=" * (-len(data_str) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _collect_plaintext_from_payload(payload: dict) -> list[str]:
    """Traverse Gmail message payload recursively and collect 'text/plain' parts."""
    texts: list[str] = []
    if not payload:
        return texts

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}
    data = body.get("data")

    if mime_type == "text/plain" and data:
        text = _base64url_decode(data)
        if text.strip():
            texts.append(text)

    for part in payload.get("parts", []) or []:
        texts.extend(_collect_plaintext_from_payload(part))

    return texts


def _fetch_thread_plaintext(service, thread_id: str) -> str:
    """Fetch a Gmail thread and return concatenated plain text content from all messages."""
    try:
        thread = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    except HttpError as he:
        raise HTTPException(status_code=he.resp.status if hasattr(he, "resp") else 500,
                            detail=f"Gmail API error: {str(he)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch thread: {str(e)}")

    messages = thread.get("messages", [])
    all_texts: list[str] = []
    for msg in messages:
        payload = msg.get("payload", {})
        texts = _collect_plaintext_from_payload(payload)

        # Fallback: sometimes text/plain is at top-level body without explicit parts
        if not texts:
            body = payload.get("body", {}) or {}
            data = body.get("data")
            if data and (payload.get("mimeType", "").startswith("text/plain") or not payload.get("mimeType")):
                fallback_text = _base64url_decode(data)
                if fallback_text.strip():
                    texts.append(fallback_text)

        all_texts.extend(texts)

    return "\n\n".join([t.strip() for t in all_texts if t and t.strip()])


def _extract_headers(payload: dict) -> dict:
    headers = {h.get('name', '').lower(): h.get('value', '') for h in (payload.get('headers', []) or [])}
    return headers


def _parse_thread_metadata(thread: dict) -> dict:
    # Try to derive a consistent subject and sender from the last message
    messages = thread.get('messages', [])
    if not messages:
        return {"subject": "", "from": ""}
    last = messages[-1]
    payload = last.get('payload', {})
    headers = _extract_headers(payload)
    subject = headers.get('subject', '')
    sender = headers.get('from', '')
    return {"subject": subject, "from": sender}


def _enhance_search_query(query: str) -> str:
    """Enhance search query with synonyms and related terms, preserving entity specificity."""
    query_lower = query.lower()
    enhanced_terms = [query]

    # Detect if this is an entity-specific query (company names, proper nouns)
    entity_indicators = ['anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia']
    has_specific_entity = any(entity in query_lower for entity in entity_indicators)

    # If query contains specific entities, be much more conservative with enhancement
    if has_specific_entity:
        print(f"[Query Enhancement] Entity-specific query detected, minimal enhancement")
        # Only add very specific, non-diluting terms
        if 'anthropic' in query_lower and 'news' in query_lower:
            enhanced_terms.extend(['anthropic', 'claude', 'ai company'])
        return query  # Return original query to preserve entity focus

    # Add related terms for common topics (only when no specific entity detected)
    if 'yahoo fantasy' in query_lower:
        enhanced_terms.extend(['yahoo sports', 'fantasy football', 'fantasy sports', 'yahoo ff'])

    # Be very conservative with news enhancement - only add if no entities present
    if 'breaking news' in query_lower and not has_specific_entity:
        enhanced_terms.extend(['urgent', 'alert'])
    elif 'news about' in query_lower and not has_specific_entity:
        enhanced_terms.extend(['update', 'announcement'])

    # Add specific topic enhancements
    if 'h-1b' in query_lower or 'h1b' in query_lower:
        enhanced_terms.extend(['visa', 'immigration', 'work permit', 'uscis'])

    if 'fantasy' in query_lower and 'football' in query_lower:
        enhanced_terms.extend(['nfl', 'fantasy sports', 'football news'])

    # Add NYT-specific enhancements
    if 'new york times' in query_lower or 'nyt' in query_lower:
        enhanced_terms.extend(['new york times', 'nytimes', 'ny times', 'times newsletter'])

    # Only enhance if we have meaningful additions and no specific entities
    if len(enhanced_terms) > 1 and not has_specific_entity:
        return f"{query} {' '.join(set(enhanced_terms[1:]))}"
    else:
        return query


def _classify_email_categories(subject: str, content: str, sender: str) -> list:
    """
    Use GPT to classify email into relevant categories.
    Returns a list of category strings.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            print("[Classification] OpenAI not available, skipping classification")
            return []
        
        client = OpenAI(api_key=api_key)
        
        # Truncate content to avoid token limits
        content_preview = content[:1000] if content else ""
        
        classification_prompt = f"""
You are an email classifier. Analyze the email and return a JSON array of relevant categories.

Email Details:
Subject: {subject}
Sender: {sender}
Content Preview: {content_preview}

Categories should be specific and useful for search. Choose from these common categories or add others:
- newsletter
- promotion
- invoice
- receipt
- coupon
- job
- social
- news
- update
- announcement
- subscription
- confirmation
- shipping
- billing
- support
- fantasy_sports
- finance
- travel
- shopping
- work
- personal
- spam

Return ONLY a JSON array of categories (2-4 categories max). Examples:
["newsletter", "promotion"]
["invoice", "billing"] 
["news", "announcement"]
["fantasy_sports", "update"]
"""

        response = client.chat.completions.create(
            model="gpt-5-nano",  # Using GPT-5 nano for fast classification
            messages=[{"role": "user", "content": classification_prompt}],
            max_completion_tokens=100
        )
        
        categories_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            import json
            categories = json.loads(categories_text)
            if isinstance(categories, list):
                # Validate categories are strings and limit to 4
                valid_categories = [cat for cat in categories if isinstance(cat, str)][:4]
                print(f"[Classification] Classified email as: {valid_categories}")
                return valid_categories
            else:
                print(f"[Classification] Invalid format: {categories_text}")
                return []
        except json.JSONDecodeError:
            print(f"[Classification] Failed to parse JSON: {categories_text}")
            return []
            
    except Exception as e:
        print(f"[Classification] Error classifying email: {e}")
        return []


def _parse_search_intent(query: str) -> dict:
    """
    Parse search query to identify category filters and other search intents.
    Returns a dict with parsed components.
    """
    print(f"[NLU] Parsing search intent for query: '{query}'")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            print("[NLU] OpenAI not available, using fallback parsing")
            return _fallback_parse_search_intent(query)
        
        print("[NLU] Using OpenAI for NLU parsing")
        client = OpenAI(api_key=api_key)
        
        nlu_prompt = f"""
You are a search query parser. Analyze the user's search query and extract structured information.

User Query: "{query}"

Extract the following information and return ONLY a JSON object:
{{
    "category_filter": null or string,
    "search_terms": string,
    "time_filter": null or string
}}

Category mapping (return the exact category name if detected):
- "show me newsletters" → "newsletter"
- "find invoices" → "invoice" 
- "all receipts" → "receipt"
- "coupons" → "coupon"
- "job emails" → "job"
- "fantasy football" → "fantasy_sports"
- "news updates" → "news"
- "promotions" → "promotion"
- "shipping notifications" → "shipping"
- "billing emails" → "billing"

Time filters:
- "today", "recent" → "recent"
- "this week" → "week"
- "this month" → "month"

Examples:
Query: "show me all coupons from last week"
{{"category_filter": "coupon", "search_terms": "coupons", "time_filter": "week"}}

Query: "find fantasy football updates"
{{"category_filter": "fantasy_sports", "search_terms": "fantasy football updates", "time_filter": null}}

Query: "news about h-1b fees"
{{"category_filter": null, "search_terms": "news about h-1b fees", "time_filter": null}}
"""

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": nlu_prompt}],
            max_completion_tokens=150
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            import json
            result = json.loads(result_text)
            print(f"[NLU] Successfully parsed query intent: {result}")
            return result
        except json.JSONDecodeError:
            print(f"[NLU] Failed to parse JSON response: {result_text}")
            print(f"[NLU] Falling back to keyword-based parsing")
            return _fallback_parse_search_intent(query)
            
    except Exception as e:
        print(f"[NLU] Error parsing search intent: {e}")
        print(f"[NLU] Falling back to keyword-based parsing")
        return _fallback_parse_search_intent(query)


def _fallback_parse_search_intent(query: str) -> dict:
    """
    Fallback parser using simple keyword matching.
    """
    print(f"[NLU] Using fallback keyword-based parsing for: '{query}'")
    query_lower = query.lower()
    
    # Simple category detection
    category_keywords = {
        'newsletter': ['newsletter', 'newsletters'],
        'coupon': ['coupon', 'coupons', 'discount', 'promo code'],
        'invoice': ['invoice', 'invoices', 'bill', 'billing'],
        'receipt': ['receipt', 'receipts', 'purchase'],
        'job': ['job', 'jobs', 'career', 'hiring'],
        'fantasy_sports': ['fantasy football', 'fantasy sports', 'fantasy'],
        'news': ['news', 'breaking', 'announcement'],
        'promotion': ['promotion', 'sale', 'offer', 'deal'],
        'shipping': ['shipping', 'delivery', 'shipped', 'package'],
    }
    
    category_filter = None
    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            category_filter = category
            break
    
    # Simple time detection
    time_filter = None
    if any(word in query_lower for word in ['today', 'recent']):
        time_filter = 'recent'
    elif 'week' in query_lower:
        time_filter = 'week'
    elif 'month' in query_lower:
        time_filter = 'month'
    
    result = {
        'category_filter': category_filter,
        'search_terms': query,
        'time_filter': time_filter
    }
    print(f"[NLU] Fallback parsing result: {result}")
    return result


def _filter_irrelevant_results(query: str, results: list) -> list:
    """
    Filter out results that are clearly irrelevant to the search query.
    This helps reduce false positives from semantic search.
    """
    query_lower = query.lower()
    filtered_results = []
    
    # Extract key terms from the query (excluding common words)
    stop_words = {'all', 'show', 'me', 'find', 'get', 'news', 'about', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    query_terms = [term for term in query_lower.split() if term not in stop_words and len(term) > 2]
    
    print(f"[Filter] Extracted key terms from query '{query}': {query_terms}")
    
    # Define irrelevant patterns for specific queries
    irrelevant_patterns = {
        'h-1b': ['subscription', 'newsletter signup', 'half baked', 'startup ideas', 'times sale'],
        'visa': ['subscription', 'newsletter', 'fantasy football', 'startup'],
        'immigration': ['subscription', 'newsletter', 'fantasy sports'],
        'fees': ['subscription offer', 'newsletter', 'sale price', '$1/wk', 'times sale'],
        'anthropic': ['breaking news', 'morning briefing', 'nyt', 'new york times', 'palestinian', 'trump', 'jimmy kimmel']
    }
    
    # Get relevant irrelevant patterns for this query
    query_irrelevant = []
    for key, patterns in irrelevant_patterns.items():
        if key in query_lower:
            query_irrelevant.extend(patterns)
    
    for result in results:
        subject = result.get('subject', '').lower()
        sender = result.get('sender', '').lower()
        content_preview = result.get('content', '')[:500].lower()
        
        # Check if result contains irrelevant patterns
        is_irrelevant = False
        for pattern in query_irrelevant:
            if pattern in subject or pattern in sender or pattern in content_preview:
                print(f"[Filter] Filtering out '{subject[:50]}...' due to irrelevant pattern: {pattern}")
                is_irrelevant = True
                break
        
        # For specific queries, require at least one key term to be present
        if query_terms and not is_irrelevant:
            # Check if any key query terms appear in the email content
            has_relevant_term = False
            full_text = f"{subject} {sender} {content_preview}"
            
            for term in query_terms:
                if term in full_text:
                    has_relevant_term = True
                    break
            
            if not has_relevant_term:
                print(f"[Filter] Filtering out '{subject[:50]}...' - no key terms found: {query_terms}")
                is_irrelevant = True
        
        # Additional specific filters
        if 'h-1b' in query_lower or 'visa' in query_lower:
            # Filter out clearly unrelated content
            if any(term in subject for term in ['half baked', 'startup ideas', 'birchbox', 'times sale']):
                is_irrelevant = True
            elif any(term in sender for term in ['half baked', 'gethalfbaked']):
                is_irrelevant = True
        
        if not is_irrelevant:
            filtered_results.append(result)
    
    print(f"[Filter] Filtered {len(results)} -> {len(filtered_results)} results")
    return filtered_results


def _boost_recent_emails(search_results: list) -> list:
    """Boost the ranking of recent emails for news-related queries."""
    from datetime import datetime, timezone, timedelta
    
    if not search_results:
        return search_results
    
    now = datetime.now(timezone.utc)
    one_week_ago = now - timedelta(days=7)
    one_month_ago = now - timedelta(days=30)
    
    for result in search_results:
        # Parse created_at timestamp if available
        created_at_str = result.get('created_at')
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                
                # Boost recent emails
                if created_at > one_week_ago:
                    # Very recent - boost by 0.2
                    result['similarity'] = min(1.0, result.get('similarity', 0) + 0.2)
                    result['recency_boost'] = 'week'
                elif created_at > one_month_ago:
                    # Recent - boost by 0.1
                    result['similarity'] = min(1.0, result.get('similarity', 0) + 0.1)
                    result['recency_boost'] = 'month'
                else:
                    result['recency_boost'] = 'none'
            except Exception:
                result['recency_boost'] = 'unknown'
        else:
            result['recency_boost'] = 'no_date'
    
    # Re-sort by similarity after boosting
    search_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    return search_results


def _create_enhanced_embedding_text(subject: str, content: str, sender: str) -> str:
    """Create enhanced text for embedding that includes context and keywords."""
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
    
    # Build enhanced text
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    # FIXED: Include full sender string (Name <email>) so sender names like "Handshake" are indexed
    if sender:
        parts.append(f"From: {sender}")
    elif sender_domain:
        # Fallback to domain only if no sender provided
        parts.append(f"From: {sender_domain}")
    if context_keywords:
        parts.append(f"Keywords: {' '.join(set(context_keywords))}")
    if content:
        # Limit content to first 6000 chars to create much richer search index
        content_truncated = content[:6000] + "..." if len(content) > 6000 else content
        parts.append(f"Content: {content_truncated}")
    
    return "\n".join(parts)

@app.get("/")
def health_check():
    return {"status": "ok"}



@app.get("/login")
def login():
    """Redirect to Google OAuth

    Always uses prompt='consent' to ensure we get a refresh token.
    This is necessary because Google only returns refresh tokens on first auth or with consent.
    """
    try:
        flow = get_flow()

        # Generate state to prevent CSRF
        state = secrets.token_urlsafe(32)
        oauth_states[state] = True

        # Always request consent to get refresh token
        # Note: This will create a new refresh token each time, but we need it
        # because without it, subsequent logins don't provide refresh tokens
        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent',  # Required to get refresh token
            state=state
        )

        return RedirectResponse(url=authorization_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.get("/callback")
def callback(code: str, state: str):
    """Handle OAuth callback, store encrypted tokens in Supabase, and return user id."""
    try:
        print(f"[OAuth] Callback received with state: {state}")
        
        # Verify state to prevent CSRF
        if state not in oauth_states:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        # Remove used state
        del oauth_states[state]
        print(f"[OAuth] State verified, getting Supabase client...")

        sb = _get_supabase()
        print(f"[OAuth] Supabase client obtained, getting OAuth flow...")

        flow = get_flow()
        print(f"[OAuth] Flow obtained, fetching token...")
        flow.fetch_token(code=code)
        print(f"[OAuth] Token fetched successfully")

        # Fetch Google user info
        access_token = flow.credentials.token
        userinfo = {}
        try:
            print(f"[OAuth] Fetching user info from Google...")
            uresp = requests.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            if uresp.ok:
                userinfo = uresp.json() or {}
                print(f"[OAuth] User info fetched successfully: {userinfo.get('email')}")
            else:
                print(f"[OAuth] Failed to fetch user info: {uresp.status_code}")
        except Exception as e:
            print(f"[OAuth] Error fetching user info: {str(e)}")
            userinfo = {}

        google_user_id = userinfo.get("sub") or userinfo.get("id")
        email = userinfo.get("email") or ""
        if not google_user_id:
            # As a fallback, store with a synthetic id based on token fingerprint (not ideal)
            google_user_id = f"anon_{secrets.token_hex(8)}"

        creds_payload = {
            "token": flow.credentials.token,
            "refresh_token": flow.credentials.refresh_token,
            "token_uri": flow.credentials.token_uri,
            "client_id": flow.credentials.client_id,
            "client_secret": flow.credentials.client_secret,
            "scopes": flow.credentials.scopes,
            "expiry": flow.credentials.expiry.isoformat() if flow.credentials.expiry else None,
        }
        
        print(f"[OAuth] Storing credentials: token={bool(creds_payload['token'])}, refresh_token={bool(creds_payload['refresh_token'])}, client_id={bool(creds_payload['client_id'])}, client_secret={bool(creds_payload['client_secret'])}")
        encrypted = _encrypt_dict(creds_payload)

        # Upsert into Supabase users table
        print(f"[OAuth] Upserting user to Supabase: {email}")
        print(f"[OAuth] Supabase URL: {SUPABASE_URL}")
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            existing = sb.table("users").select("id").eq("google_user_id", google_user_id).limit(1).execute()  # type: ignore[attr-defined]
            print(f"[OAuth] Checked existing user, found: {bool(getattr(existing, 'data', []))}")
            if (getattr(existing, "data", []) or []):
                # Update
                print(f"[OAuth] Updating existing user...")
                sb.table("users").update({
                    "email": email,
                    "encrypted_credentials": encrypted,
                    "updated_at": now_iso,
                }).eq("google_user_id", google_user_id).execute()  # type: ignore[attr-defined]
                print(f"[OAuth] User updated successfully")
            else:
                # Insert
                print(f"[OAuth] Inserting new user...")
                sb.table("users").insert({
                    "google_user_id": google_user_id,
                    "email": email,
                    "encrypted_credentials": encrypted,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }).execute()  # type: ignore[attr-defined]
                print(f"[OAuth] User inserted successfully")
        except Exception as e:
            error_msg = str(e)
            print(f"[OAuth] Error during Supabase operation: {error_msg}")
            if "nodename" in error_msg or "servname" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail=f"DNS resolution failed for Supabase URL: {SUPABASE_URL}. Please check your Supabase project settings and verify the URL is correct."
                )
            raise

        return {
            "message": "Authentication successful",
            "user_id": google_user_id,
            "email": email,
            "scopes": flow.credentials.scopes,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Callback error: {str(e)}")

@app.get("/auth/status")
def auth_status():
    """Check authentication status"""
    users_count = 0
    try:
        sb = _get_supabase()
        res = sb.table("users").select("id", count="exact").execute()  # type: ignore[attr-defined]
        users_count = getattr(res, "count", 0) or 0
    except Exception:
        users_count = 0
    return {
        "authenticated_users": users_count,
        "active_states": len(oauth_states)
    }

@app.get("/settings")
def get_settings():
    """Get user settings"""
    try:
        sb = _get_supabase()

        # Get the first user (for now, assuming single user)
        user_res = sb.table("users").select("*").limit(1).execute()
        if not user_res.data:
            print("[Settings] No users found")
            return {
                "autoSyncEnabled": False,
                "syncFrequency": "1hr",
                "userEmail": ""
            }

        user = user_res.data[0]
        user_id = user.get("id")
        user_email = user.get("email", "")

        print(f"[Settings] Found user: {user_email}")

        # Try to get settings from database (table may not exist yet)
        try:
            settings_res = sb.table("user_settings").select("*").eq("user_id", user_id).execute()

            if settings_res.data:
                settings = settings_res.data[0]
                print(f"[Settings] Found settings: auto_sync={settings.get('auto_sync_enabled')}, freq={settings.get('sync_frequency')}")
                return {
                    "autoSyncEnabled": settings.get("auto_sync_enabled", False),
                    "syncFrequency": settings.get("sync_frequency", "1hr"),
                    "userEmail": user_email
                }
        except Exception as settings_error:
            print(f"[Settings] user_settings table may not exist: {settings_error}")

        # Return defaults if no settings found
        print("[Settings] Returning default settings")
        return {
            "autoSyncEnabled": False,
            "syncFrequency": "1hr",
            "userEmail": user_email
        }
    except Exception as e:
        print(f"[Settings] Error getting settings: {e}")
        import traceback
        traceback.print_exc()
        return {
            "autoSyncEnabled": False,
            "syncFrequency": "1hr",
            "userEmail": ""
        }

@app.post("/settings")
def save_settings(request: dict):
    """Save user settings"""
    try:
        sb = _get_supabase()

        # Get the first user (for now, assuming single user)
        user_res = sb.table("users").select("*").limit(1).execute()
        if not user_res.data:
            raise HTTPException(status_code=401, detail="User not authenticated")

        user_id = user_res.data[0].get("id")
        auto_sync_enabled = request.get("autoSyncEnabled", False)
        sync_frequency = request.get("syncFrequency", "1hr")

        print(f"[Settings] Saving settings for user {user_id}: auto_sync={auto_sync_enabled}, frequency={sync_frequency}")

        # Try to save settings (table may not exist yet)
        try:
            # Check if settings exist
            settings_res = sb.table("user_settings").select("*").eq("user_id", user_id).execute()

            if settings_res.data:
                # Update existing settings
                sb.table("user_settings").update({
                    "auto_sync_enabled": auto_sync_enabled,
                    "sync_frequency": sync_frequency
                }).eq("user_id", user_id).execute()
                print(f"[Settings] Updated existing settings")
            else:
                # Insert new settings
                sb.table("user_settings").insert({
                    "user_id": user_id,
                    "auto_sync_enabled": auto_sync_enabled,
                    "sync_frequency": sync_frequency
                }).execute()
                print(f"[Settings] Created new settings")

        except Exception as table_error:
            print(f"[Settings] user_settings table may not exist, cannot save: {table_error}")
            # Return success anyway since we can't create the table dynamically
            # User will need to run the SQL migration
            return {
                "message": "Settings saved (in-memory only - run migration to persist)",
                "autoSyncEnabled": auto_sync_enabled,
                "syncFrequency": sync_frequency
            }

        return {
            "message": "Settings saved successfully",
            "autoSyncEnabled": auto_sync_enabled,
            "syncFrequency": sync_frequency
        }
    except Exception as e:
        print(f"[Settings] Error saving settings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

@app.post("/logout")
def logout():
    """Logout clears pending oauth states and stored credentials."""
    try:
        global oauth_states
        oauth_states.clear()

        # Clear all stored credentials from Supabase
        try:
            sb = _get_supabase()
            # Delete all users (this will clear incomplete credentials)
            sb.table("users").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            print("[Logout] Cleared all stored credentials from Supabase")
        except Exception as e:
            print(f"[Logout] Error clearing credentials: {e}")

        users_count = 0
        try:
            sb = _get_supabase()
            res = sb.table("users").select("id", count="exact").execute()  # type: ignore[attr-defined]
            users_count = getattr(res, "count", 0) or 0
        except Exception:
            users_count = 0
        return {
            "message": "Logout completed - cleared session and stored credentials",
            "authenticated_users": users_count,
            "active_states": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout error: {str(e)}")

@app.get("/analytics")
def get_analytics():
    """Get email analytics and statistics."""
    try:
        sb = _get_supabase()

        # Get the current user (assuming single user for now)
        user_res = sb.table("users").select("google_user_id").limit(1).execute()
        if not user_res.data:
            raise HTTPException(status_code=401, detail="User not authenticated")

        user_id = user_res.data[0].get("google_user_id")

        # Get total emails count
        total_res = sb.table("emails").select("id", count="exact").eq("google_user_id", user_id).execute()
        total_emails = getattr(total_res, "count", 0) or 0

        # Get emails from this week (last 7 days)
        from datetime import timedelta
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        week_res = sb.table("emails").select("id", count="exact").eq("google_user_id", user_id).gte("created_at", week_ago).execute()
        weekly_emails = getattr(week_res, "count", 0) or 0

        # Get important emails (importance_score >= 7)
        important_res = sb.table("emails").select("id", count="exact").eq("google_user_id", user_id).gte("importance_score", 7).execute()
        important_emails = getattr(important_res, "count", 0) or 0

        # Get top senders (top 10)
        all_emails = sb.table("emails").select("sender, sender_domain").eq("google_user_id", user_id).execute()
        emails_data = getattr(all_emails, "data", []) or []

        # Count emails per sender
        sender_counts = {}
        unknown_count = 0  # Track emails with no sender info

        for email in emails_data:
            sender = email.get("sender")
            sender_domain = email.get("sender_domain")

            # Use sender if available, fallback to sender_domain, skip if both are empty
            if sender and sender.strip():
                display_name = sender
            elif sender_domain and sender_domain.strip():
                display_name = sender_domain
            else:
                # Count unknown senders separately but don't include in top senders list
                unknown_count += 1
                continue

            sender_counts[display_name] = sender_counts.get(display_name, 0) + 1

        # Sort by count and get top 10 (excluding unknown senders)
        top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_senders_list = [{"sender": sender, "count": count} for sender, count in top_senders]

        # Get email volume for last 7 days (daily breakdown)
        daily_volume = []
        for i in range(7):
            day_start = (datetime.now(timezone.utc) - timedelta(days=i+1)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_res = sb.table("emails").select("id", count="exact").eq("google_user_id", user_id).gte("created_at", day_start.isoformat()).lt("created_at", day_end.isoformat()).execute()
            day_count = getattr(day_res, "count", 0) or 0

            daily_volume.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": day_count
            })

        # Reverse to get chronological order
        daily_volume.reverse()

        # Get category breakdown (if categories exist)
        categories_res = sb.table("emails").select("categories").eq("google_user_id", user_id).execute()
        categories_data = getattr(categories_res, "data", []) or []

        category_counts = {}
        for email in categories_data:
            categories = email.get("categories", [])
            if isinstance(categories, list):
                for category in categories:
                    category_counts[category] = category_counts.get(category, 0) + 1

        category_breakdown = [{"category": cat, "count": count} for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)]

        # Get automated emails count
        automated_res = sb.table("emails").select("id", count="exact").eq("google_user_id", user_id).eq("is_automated", True).execute()
        automated_emails = getattr(automated_res, "count", 0) or 0

        # Calculate personal emails (non-automated)
        personal_emails = total_emails - automated_emails

        # === INBOX HEALTH SCORE ===
        # Score based on ratio of important/personal emails
        health_score = 0
        if total_emails > 0:
            important_ratio = (important_emails / total_emails) * 100
            personal_ratio = (personal_emails / total_emails) * 100
            # Weighted average: 60% importance, 40% personal
            health_score = int((important_ratio * 0.6) + (personal_ratio * 0.4))

        # === NOISE RATIO ===
        noise_ratio = {
            "personal": personal_emails,
            "automated": automated_emails,
            "personalPercentage": round((personal_emails / total_emails * 100), 1) if total_emails > 0 else 0,
            "automatedPercentage": round((automated_emails / total_emails * 100), 1) if total_emails > 0 else 0
        }

        # === UNSUBSCRIBE CANDIDATES ===
        # Get all emails with sender, is_automated, and has_unsubscribe
        all_emails_detailed = sb.table("emails").select("sender, sender_domain, is_automated, has_unsubscribe").eq("google_user_id", user_id).execute()
        emails_detailed_data = getattr(all_emails_detailed, "data", []) or []

        # Find senders with >5 emails that are automated and have unsubscribe links
        unsubscribe_sender_counts = {}
        for email in emails_detailed_data:
            sender = email.get("sender")
            sender_domain = email.get("sender_domain")
            is_automated = email.get("is_automated", False)
            has_unsubscribe = email.get("has_unsubscribe", False)

            # Use sender if available, fallback to sender_domain, skip if both empty
            if sender and sender.strip():
                display_name = sender
            elif sender_domain and sender_domain.strip():
                display_name = sender_domain
            else:
                continue  # Skip emails with no sender info

            if is_automated or has_unsubscribe:
                if display_name not in unsubscribe_sender_counts:
                    unsubscribe_sender_counts[display_name] = {"count": 0, "has_unsubscribe": False}
                unsubscribe_sender_counts[display_name]["count"] += 1
                if has_unsubscribe:
                    unsubscribe_sender_counts[display_name]["has_unsubscribe"] = True

        # Filter for senders with >5 emails
        unsubscribe_candidates = [
            {"sender": sender, "count": data["count"]}
            for sender, data in unsubscribe_sender_counts.items()
            if data["count"] >= 5 and data["has_unsubscribe"]
        ]
        unsubscribe_candidates.sort(key=lambda x: x["count"], reverse=True)

        # === SMART SENDER INSIGHTS ===
        # VIP Senders: low volume (<10 emails) but high importance (not automated)
        vip_senders = []
        # Time wasters: high volume (>10 emails) and automated
        time_wasters = []

        for display_name, count in sender_counts.items():
            # Check if this sender/domain is automated
            # Match against both sender and sender_domain fields
            is_automated_sender = any(
                (e.get("sender") == display_name or e.get("sender_domain") == display_name)
                and e.get("is_automated")
                for e in emails_detailed_data
            )

            if count < 10 and not is_automated_sender:
                vip_senders.append({"sender": display_name, "count": count})
            elif count > 10 and is_automated_sender:
                time_wasters.append({"sender": display_name, "count": count})

        vip_senders.sort(key=lambda x: x["count"])
        time_wasters.sort(key=lambda x: x["count"], reverse=True)

        # === EMAIL PATTERNS & TRENDS ===
        # Get all emails with timestamps for pattern analysis
        all_emails_with_time = sb.table("emails").select("created_at").eq("google_user_id", user_id).execute()
        emails_with_time = getattr(all_emails_with_time, "data", []) or []

        # Peak email hours (0-23)
        hour_counts = {}
        day_of_week_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Monday=0, Sunday=6

        for email in emails_with_time:
            created_at_str = email.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    hour = created_at.hour
                    day_of_week = created_at.weekday()  # Monday=0, Sunday=6

                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                    day_of_week_counts[day_of_week] = day_of_week_counts.get(day_of_week, 0) + 1
                except:
                    pass

        # Find peak hour
        peak_hour = max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
        peak_hour_data = {
            "hour": peak_hour[0],
            "count": peak_hour[1],
            "label": f"{peak_hour[0]:02d}:00"
        }

        # Hourly distribution for chart
        hourly_distribution = [
            {"hour": h, "count": hour_counts.get(h, 0)}
            for h in range(24)
        ]

        # Find busiest day
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        busiest_day_num = max(day_of_week_counts.items(), key=lambda x: x[1])[0] if day_of_week_counts else 0
        busiest_day_data = {
            "day": day_names[busiest_day_num],
            "count": day_of_week_counts[busiest_day_num],
            "dayNum": busiest_day_num
        }

        # Day of week distribution
        day_of_week_distribution = [
            {"day": day_names[i], "dayNum": i, "count": day_of_week_counts.get(i, 0)}
            for i in range(7)
        ]

        # Week-over-week comparison
        two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        last_week_start = (datetime.now(timezone.utc) - timedelta(days=14))
        last_week_end = (datetime.now(timezone.utc) - timedelta(days=7))
        this_week_start = (datetime.now(timezone.utc) - timedelta(days=7))

        # Last week count
        last_week_res = sb.table("emails").select("id", count="exact").eq("google_user_id", user_id).gte("created_at", last_week_start.isoformat()).lt("created_at", last_week_end.isoformat()).execute()
        last_week_count = getattr(last_week_res, "count", 0) or 0

        # This week count (already have it as weekly_emails)
        this_week_count = weekly_emails

        # Calculate trend
        trend_percentage = 0
        if last_week_count > 0:
            trend_percentage = round(((this_week_count - last_week_count) / last_week_count) * 100, 1)

        week_over_week = {
            "lastWeek": last_week_count,
            "thisWeek": this_week_count,
            "change": this_week_count - last_week_count,
            "percentageChange": trend_percentage,
            "trend": "up" if trend_percentage > 0 else "down" if trend_percentage < 0 else "stable"
        }

        return {
            "totalEmails": total_emails,
            "weeklyEmails": weekly_emails,
            "importantEmails": important_emails,
            "topSenders": top_senders_list,
            "dailyVolume": daily_volume,
            "categoryBreakdown": category_breakdown,
            "emailsWithAttachments": automated_emails,  # Showing automated emails count
            # New analytics
            "inboxHealthScore": health_score,
            "noiseRatio": noise_ratio,
            "unsubscribeCandidates": unsubscribe_candidates[:10],  # Top 10 candidates
            "vipSenders": vip_senders[:5],  # Top 5 VIPs
            "timeWasters": time_wasters[:5],  # Top 5 time wasters
            # Email patterns & trends
            "peakHour": peak_hour_data,
            "hourlyDistribution": hourly_distribution,
            "busiestDay": busiest_day_data,
            "dayOfWeekDistribution": day_of_week_distribution,
            "weekOverWeek": week_over_week
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Analytics] Error fetching analytics: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")

@app.post("/summarize")
def summarize_email_thread(request: dict):
    """Summarize an email thread using OpenAI.

    Expected request body:
    { "messageId": "...", "userId": "..." (optional) }

    The email body text will be fetched from Gmail via the backend using the
    stored credentials. The frontend should not send raw email content.
    """
    try:
        # Extract messageId and optional userId from JSON body
        message_id = request.get("messageId")
        user_id = request.get("userId")
        
        if not message_id:
            raise HTTPException(status_code=400, detail="messageId is required")
        
        print(f"Received summarization request for message ID: {message_id}")

        # Build Gmail service to get message details
        gmail_service = _build_gmail_service(user_id)
        
        # First, try to get the message details to extract the thread ID
        try:
            message = gmail_service.users().messages().get(userId="me", id=message_id).execute()
            thread_id = message.get("threadId")
            
            if not thread_id:
                raise HTTPException(status_code=404, detail=f"No thread ID found for message {message_id}")
                
            print(f"[Summarize] Extracted thread ID {thread_id} from message {message_id}")
            
        except HttpError as he:
            if he.resp.status == 400 and "Invalid id value" in str(he):
                # The message_id might be a Gmail URL fragment, not a real message ID
                # Fall back to the old approach of using it as a thread identifier
                print(f"[Summarize] Invalid message ID {message_id}, trying as URL fragment...")
                
                # The message_id might be a thread ID in decimal format from the frontend
                # Try to convert decimal to hex and use as thread ID
                print(f"[Summarize] Trying to use {message_id} directly as thread ID...")
                
                try:
                    # First, try to use the message_id directly as a thread ID
                    test_thread = gmail_service.users().threads().get(userId="me", id=message_id).execute()
                    thread_id = message_id
                    print(f"[Summarize] Successfully using {message_id} as thread ID")
                    
                except HttpError as direct_error:
                    print(f"[Summarize] Direct use failed: {direct_error}")
                    
                    # Try converting decimal to hex (Gmail API often expects hex format)
                    try:
                        if message_id.isdigit() and len(message_id) > 10:
                            hex_thread_id = hex(int(message_id))[2:]  # Convert to hex, remove '0x' prefix
                            print(f"[Summarize] Trying decimal->hex conversion: {message_id} -> {hex_thread_id}")
                            
                            test_thread = gmail_service.users().threads().get(userId="me", id=hex_thread_id).execute()
                            thread_id = hex_thread_id
                            print(f"[Summarize] Successfully using converted thread ID: {hex_thread_id}")
                        else:
                            raise ValueError("Not a valid decimal thread ID")
                            
                    except (ValueError, HttpError) as hex_error:
                        print(f"[Summarize] Hex conversion also failed: {hex_error}")
                        raise direct_error  # Re-raise the original error to continue to fallback
                    
                except HttpError as thread_error:
                    print(f"[Summarize] {message_id} is not a valid thread ID either: {thread_error}")
                    
                    # Last resort: List recent threads and try to find a different one
                    try:
                        threads_resp = gmail_service.users().threads().list(userId="me", maxResults=50).execute()
                        threads = threads_resp.get('threads', []) or []
                        
                        if threads:
                            # Instead of always using first thread, try to find one that's different from known stale ones
                            stale_thread_ids = ['19930a647fcae0eb', '1990ce3ff04ac1a6']  # Known stale threads
                            
                            for thread in threads:
                                tid = thread.get('id')
                                if tid and tid not in stale_thread_ids:
                                    thread_id = tid
                                    print(f"[Summarize] Using alternative thread as fallback: {thread_id}")
                                    break
                            
                            # If all threads are stale, use the first one
                            if not thread_id and threads:
                                thread_id = threads[0].get('id')
                                print(f"[Summarize] Using first thread as last resort: {thread_id}")
                        else:
                            raise HTTPException(status_code=404, detail="No threads found and invalid message ID")
                    except Exception as list_error:
                        raise HTTPException(status_code=400, detail=f"Could not resolve thread ID: {str(list_error)}")
                        
                except Exception as fallback_error:
                    raise HTTPException(status_code=400, detail=f"Invalid message ID and fallback failed: {str(fallback_error)}")
            else:
                raise HTTPException(status_code=he.resp.status if hasattr(he, "resp") else 500,
                                    detail=f"Gmail API error getting message: {str(he)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get message details: {str(e)}")

        # Ensure OpenAI client is available and API key present
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            raise HTTPException(status_code=500, detail="OpenAI SDK not available or OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        # Fetch clean email text for the thread using the thread ID we got from the message
        email_text = _fetch_thread_plaintext(gmail_service, thread_id)
        if not email_text:
            email_text = (
                "No plain text content found in this thread. Summarize what can be inferred "
                f"from the conversation context of thread {thread_id}."
            )

        system_msg = (
            "You are a helpful assistant that writes concise, factual summaries of email threads. "
            "Output 3-5 sentences followed by up to 3 bullet point action items. If details are missing, "
            "note assumptions explicitly."
        )

        def token_stream():
            try:
                # Stream using OpenAI Responses API
                with client.responses.stream(
                    model="gpt-5-nano",
                    input=f"System: {system_msg}\n\nUser: {email_text}",
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            delta = getattr(event, "delta", "") or ""
                            if delta:
                                yield delta.encode("utf-8")
                        elif event.type == "response.error":
                            err = getattr(event, "error", None)
                            error_message = getattr(err, "message", None) if err else None
                            if error_message:
                                yield f"\n[error] {error_message}".encode("utf-8")
            except Exception as oe:
                yield f"\n[error] OpenAI stream error: {str(oe)}".encode("utf-8")

        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")
        
    except HTTPException as he:
        # Preserve intended HTTP status codes (e.g., 401 for no auth, 404 for not found)
        print(f"[ERROR] HTTPException in /summarize: {he.status_code} - {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Exception in /summarize: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/summarize-content")
def summarize_email_content(request: dict):
    """Summarize email content directly without using Gmail API.
    
    Expected request body:
    { "content": "email content text..." }
    """
    try:
        content = request.get("content", "").strip()
        
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
            
        if len(content) < 50:
            raise HTTPException(status_code=400, detail="content is too short to summarize")
            
        print(f"[Summarize-Content] Received content for summarization: {len(content)} characters")
        print(f"[Summarize-Content] Content preview: {content[:200]}...")
        
        # Ensure OpenAI client is available and API key present
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            raise HTTPException(status_code=500, detail="OpenAI SDK not available or OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        system_msg = (
            "You are a helpful assistant that writes concise, factual summaries of email content. "
            "Output 3-5 sentences followed by up to 3 bullet point action items. If details are missing, "
            "note assumptions explicitly."
        )

        def token_stream():
            try:
                # Stream using OpenAI Responses API
                with client.responses.stream(
                    model="gpt-5-nano",
                    input=f"System: {system_msg}\n\nUser: {content}",
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            delta = getattr(event, "delta", "") or ""
                            if delta:
                                yield delta.encode("utf-8")
                        elif event.type == "response.error":
                            err = getattr(event, "error", None)
                            error_message = getattr(err, "message", None) if err else None
                            if error_message:
                                yield f"\n[error] {error_message}".encode("utf-8")
            except Exception as oe:
                yield f"\n[error] OpenAI stream error: {str(oe)}".encode("utf-8")

        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")
        
    except HTTPException as he:
        print(f"[ERROR] HTTPException in /summarize-content: {he.status_code} - {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Exception in /summarize-content: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            raise HTTPException(status_code=504, detail="Request timed out. OpenAI may be slow.")
        elif "connection" in str(e).lower():
            raise HTTPException(status_code=503, detail="Connection error. Please try again.")
        else:
            raise HTTPException(status_code=500, detail=f"Content summarization error: {str(e)}")

@app.get("/summarize/{thread_id}")
def get_summary_status(thread_id: str):
    """Get summary status for a specific thread"""
    # TODO: Implement summary retrieval from database
    return {
        "thread_id": thread_id,
        "status": "mock_status",
        "message": "Summary endpoint working - ready for Gmail API integration"
    }


@app.post("/search")
async def search_emails(request: dict):
    """Enhanced semantic search through indexed emails with temporal relevance and query expansion.

    Expected request body: { "query": "search terms", "userId": "..." (optional) }
    """
    try:
        print(f"\n=== SEARCH REQUEST START ===")
        print(f"[Search] Request received: {request}")
        
        sb = _get_supabase()
        print(f"[Search] Supabase connection established")
        
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        
        user_id = request.get("userId")
        print(f"[Search] Query: '{query}', User ID: {user_id}")
        
        # Parse search intent to identify category filters
        print(f"[Search] Parsing search intent...")
        search_intent = _parse_search_intent(query)
        category_filter = search_intent.get('category_filter')
        search_terms = search_intent.get('search_terms', query)
        print(f"[Search] Search intent parsed: {search_intent}")
        
        # Load embedding model and generate query embedding
        print(f"[Search] Loading embedding model...")
        model = _get_embedding_model()
        print(f"[Search] Embedding model loaded successfully")
        
        # Enhanced query processing for better search results
        enhanced_query = _enhance_search_query(search_terms)
        print(f"[Search] Original query: '{query}', Enhanced: '{enhanced_query}', Category filter: {category_filter}")
        
        print(f"[Search] Generating query embedding...")
        query_embedding = model.encode(enhanced_query, normalize_embeddings=True)
        
        print(f"[Search] Query: '{query}', embedding shape: {query_embedding.shape}")
        
        # Implement Hybrid Search with AI Re-ranking
        print(f"[Search] Implementing Hybrid Search: Vector + Keyword + AI Re-ranking")
        
        # Part 1: Candidate Generation - Parallel Vector and Keyword Search
        print(f"[Search] Step 1: Candidate Generation")
        
        # 1a. Vector Search - Use existing semantic search
        print(f"[Search] Performing vector search...")
        is_news_query = any(term in search_terms.lower() for term in ['news', 'update', 'recent', 'latest', 'announcement'])

        # Detect entity-specific queries and use stricter thresholds
        entity_indicators = ['anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia']
        has_specific_entity = any(entity in search_terms.lower() for entity in entity_indicators)

        if has_specific_entity:
            match_threshold = 0.35  # Higher threshold for entity queries for better precision
            match_count = 15       # Fewer candidates for faster performance
            print(f"[Search] Entity-specific query detected, using threshold: {match_threshold}")
        else:
            match_threshold = 0.4 if is_news_query else 0.45
            match_count = 12  # Fewer candidates for better performance
        
        vector_results = sb.rpc('match_emails', {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': match_threshold,
            'match_count': match_count
        }).execute()
        
        vector_candidates = getattr(vector_results, "data", []) or []
        print(f"[Search] Vector search found {len(vector_candidates)} candidates")
        
        # 1b. Keyword Search - Find emails with exact query terms
        print(f"[Search] Performing keyword search...")
        keyword_candidates = _perform_keyword_search(sb, query, user_id)

        # Debug: Check what's actually in the database
        print(f"[Debug] Checking database contents...")
        try:
            total_emails = sb.table("emails").select("id", count="exact").execute()
            total_count = getattr(total_emails, "count", 0) or 0
            print(f"[Debug] Total emails in database: {total_count}")

            anthropic_emails = sb.table("emails").select("subject").ilike("subject", "%anthropic%").limit(10).execute()
            anthropic_rows = getattr(anthropic_emails, "data", []) or []
            print(f"[Debug] Emails with 'anthropic' in subject: {len(anthropic_rows)}")
            for row in anthropic_rows[:3]:
                print(f"[Debug] - {row.get('subject', 'No subject')}")

            claude_emails = sb.table("emails").select("subject").ilike("subject", "%claude%").limit(10).execute()
            claude_rows = getattr(claude_emails, "data", []) or []
            print(f"[Debug] Emails with 'claude' in subject: {len(claude_rows)}")
            for row in claude_rows[:3]:
                print(f"[Debug] - {row.get('subject', 'No subject')}")
        except Exception as e:
            print(f"[Debug] Error checking database: {e}")
        
        # 1c. Combine candidates
        all_candidates = _combine_search_candidates(vector_candidates, keyword_candidates)

        if not all_candidates:
            search_results = []
            print(f"[Search] No candidates found from either search method")
        else:
            # Pre-filter candidates to remove clearly irrelevant results
            print(f"[Search] Pre-filtering {len(all_candidates)} candidates")
            filtered_candidates = _prefilter_candidates(all_candidates, query)
            print(f"[Search] After pre-filtering: {len(filtered_candidates)} candidates")

            # Part 2: AI Re-ranking
            print(f"[Search] Step 2: AI Re-ranking of {len(filtered_candidates)} candidates")
            search_results = _ai_rerank_candidates(filtered_candidates, query)
            
            # Limit to top results - reduce for better performance
            max_results = 8 if is_news_query else 6
            search_results = search_results[:max_results]
            
            print(f"[Search] Hybrid search completed: {len(search_results)} final results")
        
        print(f"[Search] Raw results: {len(search_results)} items")
        
        if search_results:
            print(f"[Search] First result sample: {search_results[0].keys() if search_results[0] else 'Empty result'}")
            for i, result in enumerate(search_results[:3]):
                relevance_score = result.get('relevance_score', 'N/A')
                similarity = result.get('similarity', 'N/A')
                source = result.get('source', 'N/A')
                print(f"[Search] Result {i+1}: subject='{result.get('subject', 'N/A')}', relevance_score={relevance_score}, similarity={similarity}, source={source}")
        else:
            print(f"[Search] No results found")
        
        # Set search type for hybrid approach
        search_type = "hybrid_ai_reranked"
        
        # Add created_at field to results if missing (for time filtering and display)
        for result in search_results:
            if 'created_at' not in result and 'thread_id' in result:
                try:
                    # Fetch the created_at from the emails table
                    email_data = sb.table("emails").select("created_at").eq("thread_id", result['thread_id']).limit(1).execute()
                    email_rows = getattr(email_data, "data", []) or []
                    if email_rows:
                        result['created_at'] = email_rows[0].get('created_at')
                except Exception as e:
                    print(f"[Search] Warning: Could not fetch created_at for thread {result.get('thread_id')}: {e}")
        
        # Skip old filtering since AI re-ranking handles relevance
        # Post-process results to boost recent emails for news queries (optional with AI ranking)
        if is_news_query and search_results:
            search_results = _boost_recent_emails(search_results)
        
        print(f"[Search] Final results: {len(search_results)} matching emails")
        
        result_data = {
            "status": "success",
            "query": query,
            "enhanced_query": enhanced_query,
            "results": search_results,
            "count": len(search_results),
            "search_type": search_type
        }
        
        if category_filter:
            result_data["category_filter"] = category_filter
        
        print(f"[Search] Returning response: status={result_data['status']}, count={result_data['count']}, search_type={result_data['search_type']}")
        print(f"=== SEARCH REQUEST END ===\n")
            
        return result_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/smart-search")
async def smart_search_emails(request: dict):
    """Enhanced natural language search that parses queries into structured format.

    This endpoint addresses "it's not fetching what I need" issues by:
    1. Parsing natural language into structured queries (people, topics, time ranges)
    2. Using AI to understand user intent (fetch_emails, summarize, list_receipts)
    3. Applying appropriate search strategies based on the query structure

    Expected request body: { "query": "all updates from john in the last 3 days", "userId": "..." (optional) }

    Returns:
    {
        "results": [...],  # Matching emails
        "parsed_query": {...},  # The structured query that was used
        "search_strategy": "...",  # Description of search approach
        "result_count": 10
    }
    """
    try:
        if not SMART_SEARCH_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="Smart search not available. Install dependencies: pip install openai python-dateutil"
            )

        print(f"\n=== SMART SEARCH REQUEST START ===")
        print(f"[SmartSearch] Request received: {request}")

        sb = _get_supabase()
        query = request.get("query", "").strip()

        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        user_id = request.get("userId")
        print(f"[SmartSearch] Query: '{query}', User ID: {user_id}")

        # Load embedding model
        model = _get_embedding_model()

        # Use smart search with OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        result = smart_search(
            query=query,
            supabase_client=sb,
            embedding_model=model,
            user_id=user_id,
            anthropic_api_key=openai_key  # Parameter name kept for compatibility
        )

        print(f"[SmartSearch] Completed: {result['result_count']} results")
        print(f"[SmartSearch] Strategy: {result['search_strategy']}")
        print(f"=== SMART SEARCH REQUEST END ===\n")

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SmartSearch] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Smart search error: {str(e)}")


@app.post("/unified-search")
async def unified_search(
    request: dict, 
    x_encryption_key: Optional[str] = Header(None)
):
    """
    Unified search endpoint that combines the best of /search, /smart-search, and /chat.
    
    Features:
    - Hybrid search (vector + keyword + AI re-ranking) for better accuracy
    - Natural language parsing for understanding intent
    - Dual response modes: structured results or conversational streaming
    
    Request body:
    {
        "query": "what is that email about handshake delaying payments",
        "mode": "search" | "chat",  # Default: "chat"
        "userId": "..." (optional)
    }
    
    Response modes:
    - "search": Returns structured JSON results (like /search endpoint)
    - "chat": Returns streaming conversational response (like /chat endpoint)
    
    This endpoint solves the semantic search weakness by adding keyword search fallback,
    so queries like "handshake delaying payments" will find exact keyword matches even
    if the embedding model doesn't capture the semantic meaning perfectly.
    """
    try:
        sb = _get_supabase()
        query = request.get("query", "").strip()
        mode = request.get("mode", "chat")  # Default to chat mode
        user_id = request.get("userId")
        
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        
        print(f"\n=== UNIFIED SEARCH START (mode: {mode}) ===")
        print(f"[UnifiedSearch] Query: '{query}', User ID: {user_id}")
        
        # STEP 1: Parse query intent
        search_intent = _parse_search_intent(query)
        category_filter = search_intent.get('category_filter')
        search_terms = search_intent.get('search_terms', query)
        print(f"[UnifiedSearch] Search intent: {search_intent}")
        
        # STEP 2: Enhanced query processing and embedding generation
        model = _get_embedding_model()
        enhanced_query = _enhance_search_query(search_terms)
        query_embedding = model.encode(enhanced_query, normalize_embeddings=True)
        
        print(f"[UnifiedSearch] Enhanced query: '{enhanced_query}'")
        
        # STEP 3: HYBRID SEARCH - Vector + Keyword + AI Re-ranking
        print(f"[UnifiedSearch] Performing hybrid search...")
        
        # 3a. Vector Search with adaptive thresholds
        is_news_query = any(term in search_terms.lower() for term in ['news', 'update', 'recent', 'latest'])
        entity_indicators = ['anthropic', 'openai', 'google', 'handshake', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia']
        has_specific_entity = any(entity in search_terms.lower() for entity in entity_indicators)
        
        if has_specific_entity:
            match_threshold = 0.35  # Higher threshold for entity queries
            match_count = 15
            print(f"[UnifiedSearch] Entity-specific query detected, using threshold: {match_threshold}")
        else:
            match_threshold = 0.4 if is_news_query else 0.45
            match_count = 12
        
        vector_results = sb.rpc('match_emails', {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': match_threshold,
            'match_count': match_count
        }).execute()
        
        vector_candidates = getattr(vector_results, "data", []) or []
        print(f"[UnifiedSearch] Vector search: {len(vector_candidates)} candidates")
        
        # 3b. Keyword Search - This catches exact matches like "handshake"
        keyword_candidates = _perform_keyword_search(sb, query, user_id)
        print(f"[UnifiedSearch] Keyword search: {len(keyword_candidates)} candidates")
        
        # 3c. Combine candidates (deduplicate by thread_id)
        all_candidates = _combine_search_candidates(vector_candidates, keyword_candidates)
        print(f"[UnifiedSearch] Combined: {len(all_candidates)} unique candidates")
        
        if not all_candidates:
            search_results = []
            print(f"[UnifiedSearch] No candidates found from either search method")
        else:
            # 3d. Pre-filter to remove clearly irrelevant results
            filtered_candidates = _prefilter_candidates(all_candidates, query)
            print(f"[UnifiedSearch] After pre-filtering: {len(filtered_candidates)} candidates")
            
            # 3e. AI Re-ranking for final precision
            search_results = _ai_rerank_candidates(filtered_candidates, query)
            max_results = 8 if is_news_query else 6
            search_results = search_results[:max_results]
            
            print(f"[UnifiedSearch] Final results after AI re-ranking: {len(search_results)} emails")
        
        # Log top results for debugging
        if search_results:
            for i, result in enumerate(search_results[:3]):
                print(f"[UnifiedSearch] Result {i+1}: '{result.get('subject', 'N/A')}' (score: {result.get('relevance_score', 'N/A')})")
        
        # STEP 4: Decrypt emails if encryption key provided
        if x_encryption_key and search_results:
            print(f"[UnifiedSearch] Decrypting {len(search_results)} emails...")
            decrypted_results = []
            for result in search_results:
                if result.get('encrypted_content') and result.get('iv'):
                    try:
                        encrypted_content = result.get('encrypted_content')
                        iv = result.get('iv')
                        decrypted_json = _decrypt_with_user_key(encrypted_content, iv, x_encryption_key)
                        email_data = json.loads(decrypted_json)
                        result['subject'] = email_data.get('subject', 'No Subject')
                        result['sender'] = email_data.get('sender', 'Unknown Sender')
                        result['content'] = email_data.get('content', '')
                        decrypted_results.append(result)
                        print(f"[UnifiedSearch] Decrypted: '{result['subject']}'")
                    except Exception as e:
                        print(f"[UnifiedSearch] Decryption failed: {e}")
                        continue
                else:
                    decrypted_results.append(result)
            search_results = decrypted_results
            print(f"[UnifiedSearch] After decryption: {len(search_results)} emails available")
        
        # STEP 5: Return based on mode
        if mode == "search":
            # Return structured results like /search endpoint
            result_data = {
                "status": "success",
                "query": query,
                "enhanced_query": enhanced_query,
                "results": search_results,
                "count": len(search_results),
                "search_type": "hybrid_ai_reranked",
            }
            
            if category_filter:
                result_data["category_filter"] = category_filter
            
            print(f"[UnifiedSearch] Returning structured results: {len(search_results)} emails")
            print(f"=== UNIFIED SEARCH END ===\n")
            return result_data
        
        elif mode == "chat":
            # Return streaming conversational response like /chat endpoint
            email_context = _prepare_email_context(search_results, query)
            print(f"[UnifiedSearch] Prepared email context: {len(email_context)} characters")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or OpenAI is None:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            
            client = OpenAI(api_key=api_key)
            
            # Detect if user is asking about important emails
            query_lower = query.lower()
            is_asking_important = any(term in query_lower for term in ['important', 'critical', 'urgent', 'priority'])
            
            # Create system prompt
            system_prompt = (
                "You are a helpful email assistant that can answer questions about the user's emails. "
                "You have access to relevant email context and should provide conversational, helpful responses. "
                "When answering questions about specific time periods (like 'this week' or 'last month'), "
                "focus on the most recent and relevant emails. Be concise but informative, and if you don't "
                "have enough information, say so clearly. "
            )
            
            if is_asking_important:
                system_prompt += (
                    "\n\nIMPORTANT: The user asked about 'important' emails. Categorize the emails into: "
                    "\n\n**Important emails** (mention these first):"
                    "\n1) Personal emails from real people (not automated/marketing)"
                    "\n2) Work-related communications from colleagues/clients"
                    "\n3) Financial alerts (bills, payments, receipts)"
                    "\n4) Account notifications (password resets, security alerts)"
                    "\n5) Travel confirmations and tickets"
                    "\n\n**Less important** (newsletters/promotional - mention these briefly if present):"
                    "\n- Newsletters, promotional emails, marketing content, social media notifications, automated digests"
                    "\n\nIf there are NO important emails, say so clearly, but still mention what other emails they received. "
                    "For example: 'You received 5 emails today, but they are all newsletters and promotional content: [list them]'"
                )
            
            system_prompt += (
                "\n\nIMPORTANT: When mentioning emails, always include the Gmail URL as a clickable link using markdown format: "
                "[Email Subject](Gmail URL). This allows users to click directly to open the email in Gmail."
            )
            
            # Stream the response
            async def generate_stream():
                try:
                    print(f"[UnifiedSearch] Starting streaming response with gpt-4o-mini...")
                    
                    # Send emails data and metadata first
                    response_data = {
                        "emails": search_results,
                        "search_metadata": {
                            "query_type": "unified_hybrid",
                            "results_count": len(search_results),
                            "search_type": "hybrid_ai_reranked"
                        }
                    }
                    emails_data = json.dumps(response_data)
                    yield f"data: {emails_data}\n\n"
                    
                    # Stream AI response
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"User Question: {query}\n\nRelevant Email Context:\n{email_context}"}
                        ],
                        max_tokens=1000,
                        temperature=0.7,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content_chunk = chunk.choices[0].delta.content
                            chunk_data = json.dumps({"content": content_chunk})
                            yield f"data: {chunk_data}\n\n"
                    
                    # Send done signal
                    yield "data: [DONE]\n\n"
                    print(f"[UnifiedSearch] Streaming complete")
                    print(f"=== UNIFIED SEARCH END ===\n")
                    
                except Exception as e:
                    print(f"[UnifiedSearch] Streaming error: {str(e)}")
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'search' or 'chat'")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[UnifiedSearch] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unified search error: {str(e)}")


@app.post("/chat")
async def chat_with_emails(request: dict, x_encryption_key: Optional[str] = Header(None)):
    """Chat with your emails using natural language questions.

    Expected request body:
    { "message": "What emails did I receive this week about NYT news?", "userId": "..." (optional) }

    This endpoint:
    1. Analyzes the user's question to extract search terms and time filters
    2. Searches relevant emails using semantic search
    3. Uses OpenAI to generate a conversational response based on the found emails
    """
    try:
        sb = _get_supabase()

        # Debug: Log the full request
        print(f"[Chat] DEBUG: Full request received: {request}")

        message = request.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        user_id = request.get("userId")
        print(f"[Chat] Processing question: '{message}' (userId: {user_id})")
        
        # Analyze the question to extract search terms and time context
        search_terms, time_context = _analyze_question(message)
        print(f"[Chat] Extracted search terms: {search_terms}, time context: {time_context}")

        # Detect generic temporal queries that need special handling
        # BUT exclude queries that mention specific entities, companies, or topics
        message_lower = message.lower()
        has_specific_entities = any(entity in message_lower for entity in [
            'anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia',
            'nyt', 'new york times', 'ny times', 'times', 'yahoo', 'facebook', 'twitter', 'linkedin'
        ])

        # Check if query has specific topics/keywords (more than just temporal/email words)
        # Remove temporal and generic email words, see what's left
        temporal_and_generic = ['summarize', 'recent', 'latest', 'newest', 'today', 'yesterday',
                                'this', 'week', 'month', 'emails', 'email', 'my', 'me', 'show']
        content_words = [word for word in message_lower.split() if word not in temporal_and_generic and len(word) > 2]
        has_specific_content = len(content_words) > 0  # Has words like "fantasy", "football", etc.

        is_generic_temporal_query = (
            not has_specific_entities and  # Don't treat as generic if specific entities mentioned
            not has_specific_content and   # Don't treat as generic if specific content/topics mentioned
            any(term in message.lower() for term in [
                'today\'s emails', 'recent emails', 'latest emails', 'new emails',
                'what emails', 'show me emails', 'my emails', 'all emails'
            ]) and any(term in message.lower() for term in [
                'today', 'recent', 'latest', 'new', 'this week', 'yesterday'
            ])
        )
        
        # Build search query with time filters if needed
        search_query = " ".join(search_terms) if search_terms else "email"
        
        # For generic temporal queries, use a broader search approach
        if is_generic_temporal_query:
            print(f"[Chat] Detected generic temporal query, using broader search")
            enhanced_query = "email message recent latest"
        else:
            # Enhance the search query for better matching
            enhanced_query = _enhance_search_query(search_query)
        
        print(f"[Chat] Enhanced search query: '{enhanced_query}'")
        
        if time_context:
            enhanced_query = f"{enhanced_query} {time_context}"
        
        print(f"[Chat] Final search query: '{enhanced_query}'")
        
        # Perform semantic search using the same hybrid approach as /search endpoint
        print(f"[Chat] Using simple semantic search for chat...")
        
        # Load embedding model and generate query embedding
        model = _get_embedding_model()
        query_embedding = model.encode(message, normalize_embeddings=True)
        
        # Call Supabase RPC function for semantic search
        print(f"[Chat] Calling Supabase RPC 'match_emails'...")
        results = sb.rpc('match_emails', {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': 0.3,
            'match_count': 20
        }).execute()
        
        search_results = results.data if results.data else []
        print(f"[Chat] Semantic search found {len(search_results)} emails")

        # Keep query type flags for downstream filtering
        message_lower = message.lower()
        is_news_query = 'news' in message_lower
        is_latest_query = any(term in message_lower for term in ['latest', 'recent', 'new'])
        is_sports_query = 'sports' in message_lower or 'fantasy' in message_lower
        
        # Add created_at field to results if missing (for time filtering)
        for result in search_results:
            if 'created_at' not in result and 'thread_id' in result:
                try:
                    # Fetch the created_at from the emails table
                    email_data = sb.table("emails").select("created_at").eq("thread_id", result['thread_id']).limit(1).execute()
                    email_rows = getattr(email_data, "data", []) or []
                    if email_rows:
                        result['created_at'] = email_rows[0].get('created_at')
                except Exception as e:
                    print(f"[Chat] Warning: Could not fetch created_at for thread {result.get('thread_id')}: {e}")
        
        # Apply time filtering to results if time context was specified
        if time_context and search_results:
            filtered_results = _apply_time_filter(search_results, time_context)
            print(f"[Chat] After time filtering: {len(filtered_results)} emails")
            search_results = filtered_results

        # Decrypt encrypted emails if encryption key is provided
        if x_encryption_key and search_results:
            print(f"[Chat] Decrypting {len(search_results)} encrypted emails...")
            decrypted_results = []
            for result in search_results:
                # Check if email is encrypted
                if result.get('encrypted_content') and result.get('iv'):
                    try:
                        # Decrypt email content
                        encrypted_content = result.get('encrypted_content')
                        iv = result.get('iv')
                        decrypted_json = _decrypt_with_user_key(encrypted_content, iv, x_encryption_key)
                        email_data = json.loads(decrypted_json)

                        # Add decrypted fields to result
                        result['subject'] = email_data.get('subject', 'No Subject')
                        result['sender'] = email_data.get('sender', 'Unknown Sender')
                        result['content'] = email_data.get('content', '')

                        print(f"[Chat] Decrypted: '{result['subject']}' from {result['sender']}")
                        decrypted_results.append(result)
                    except Exception as e:
                        print(f"[Chat] Warning: Failed to decrypt email: {e}")
                        # Skip emails that fail to decrypt
                        continue
                else:
                    # Plain text email (backwards compatibility)
                    decrypted_results.append(result)

            search_results = decrypted_results
            print(f"[Chat] After decryption: {len(search_results)} emails available")

        # For generic temporal queries or "latest" queries, sort by recency instead of similarity
        if (is_generic_temporal_query or is_latest_query) and search_results:
            from datetime import datetime
            def get_date(result):
                created_at = result.get('created_at')
                if created_at:
                    try:
                        return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        pass
                return datetime.min.replace(tzinfo=datetime.now().tzinfo)

            search_results.sort(key=get_date, reverse=True)  # Most recent first
            print(f"[Chat] Sorted {len(search_results)} results by recency")

            # For "latest" queries, take only the top results
            if is_latest_query and len(search_results) > 5:
                search_results = search_results[:5]
                print(f"[Chat] Limited to top 5 most recent for 'latest' query")
        
        # Detect if user is asking about "important" emails (for LLM prompt enhancement)
        is_asking_important = any(term in message_lower for term in ['important', 'critical', 'urgent', 'priority'])

        # Apply sports-specific filtering
        if search_results and is_sports_query:
            original_count = len(search_results)
            # Filter for sports-related senders and subjects
            sports_keywords = ['fantasy', 'football', 'basketball', 'baseball', 'sports', 'nfl', 'nba', 'mlb', 'espn', 'yahoo fantasy', 'sleeper', 'draft', 'lineup']
            filtered_results = []
            for email in search_results:
                sender = (email.get('sender') or '').lower()
                subject = (email.get('subject') or '').lower()
                content = (email.get('content') or '').lower()
                # Keep if sender, subject, or content contains sports keywords
                if any(keyword in sender or keyword in subject or keyword in content for keyword in sports_keywords):
                    filtered_results.append(email)
            search_results = filtered_results
            if original_count != len(search_results):
                print(f"[Chat] Sports filter: reduced from {original_count} to {len(search_results)} emails")

        # Apply minimal filtering for chat to preserve context - only filter if query is very specific
        entity_indicators = ['anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia']
        has_specific_entity = any(entity in message_lower for entity in entity_indicators)

        # Only apply aggressive filtering for very specific entity queries, not for general news/time/important queries
        if search_results and has_specific_entity and not is_news_query and not is_asking_important and not is_sports_query:
            original_count = len(search_results)
            search_results = _filter_irrelevant_results(message, search_results)
            filtered_count = len(search_results)
            if original_count != filtered_count:
                print(f"[Chat] Filtered out {original_count - filtered_count} irrelevant results")
        elif not is_sports_query:
            print(f"[Chat] Skipping aggressive filtering for temporal/news/important query - keeping all {len(search_results)} results for LLM to categorize")
        

        # Prepare context for the LLM
        email_context = _prepare_email_context(search_results, message)
        print(f"[Chat] Prepared email context with {len(search_results)} results")
        print(f"[Chat] Context length: {len(email_context)} characters")

        # Debug: Print first few email subjects
        if search_results:
            print(f"[Chat] DEBUG: Found {len(search_results)} emails:")
            for i, email in enumerate(search_results[:5]):
                print(f"[Chat] DEBUG: {i+1}. '{email.get('subject', 'No subject')}' from {email.get('sender', 'Unknown')}")
        
        # Ensure OpenAI client is available
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            raise HTTPException(status_code=500, detail="OpenAI SDK not available or OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        # Detect if user is asking about "important" emails
        is_asking_important = any(term in message_lower for term in ['important', 'critical', 'urgent', 'priority'])

        # Create system prompt for email assistant
        system_prompt = (
            "You are a helpful email assistant that can answer questions about the user's emails. "
            "You have access to relevant email context and should provide conversational, helpful responses. "
            "When answering questions about specific time periods (like 'this week' or 'last month'), "
            "focus on the most recent and relevant emails. Be concise but informative, and if you don't "
            "have enough information, say so clearly. "
        )

        if is_asking_important:
            system_prompt += (
                "\n\nIMPORTANT: The user asked about 'important' emails. Categorize the emails into: "
                "\n\n**Important emails** (mention these first):"
                "\n1) Personal emails from real people (not automated/marketing)"
                "\n2) Work-related communications from colleagues/clients"
                "\n3) Financial alerts (bills, payments, receipts)"
                "\n4) Account notifications (password resets, security alerts)"
                "\n5) Travel confirmations and tickets"
                "\n\n**Less important** (newsletters/promotional - mention these briefly if present):"
                "\n- Newsletters, promotional emails, marketing content, social media notifications, automated digests"
                "\n\nIf there are NO important emails, say so clearly, but still mention what other emails they received. "
                "For example: 'You received 5 emails today, but they are all newsletters and promotional content: [list them]'"
            )

        system_prompt += (
            "\n\nIMPORTANT: When mentioning emails, always include the Gmail URL as a clickable link using markdown format: "
            "[Email Subject](Gmail URL). This allows users to click directly to open the email in Gmail."
        )
        
        # Use streaming for better UX
        async def generate_stream():
            try:
                print(f"[Chat] Starting streaming response with gpt-4o-mini...")

                # Send the emails data and search metadata first
                response_data = {
                    "emails": search_results,
                    "search_metadata": {
                        "query_type": "chat",
                        "results_count": len(search_results)
                    }
                }
                emails_data = json.dumps(response_data)
                yield f"data: {emails_data}\n\n"

                # Stream the AI response
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"User Question: {message}\n\nRelevant Email Context:\n{email_context}"}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    stream=True  # Enable streaming
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        # Send each chunk as server-sent event
                        chunk_data = json.dumps({"content": content_chunk})
                        yield f"data: {chunk_data}\n\n"

                # Send done signal
                yield "data: [DONE]\n\n"
                print(f"[Chat] Streaming complete")

            except Exception as e:
                print(f"[Chat] Streaming error: {str(e)}")
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Exception in /chat: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/search-feedback")
async def record_search_feedback(request: dict):
    """
    Record user feedback on search results for adaptive learning.

    Expected request body:
    {
        "search_query_id": "uuid-of-search-query",
        "google_user_id": "user's google ID",
        "email_id": "uuid-of-email (optional)",
        "thread_id": "gmail thread ID (optional)",
        "action": "clicked" | "relevant" | "irrelevant" | "skipped" | "dismissed",
        "similarity_score": 0.85 (optional),
        "rank_position": 1 (optional),
        "dwell_time_ms": 5000 (optional),
        "metadata": {} (optional JSON object)
    }

    This endpoint:
    1. Records user interaction with search results
    2. Updates sender affinity scores
    3. Updates user search preferences (CTR, etc.)
    """
    try:
        sb = _get_supabase()

        # Extract required fields
        search_query_id = request.get("search_query_id")
        google_user_id = request.get("google_user_id")
        action = request.get("action")

        if not action or action not in ['clicked', 'relevant', 'irrelevant', 'skipped', 'dismissed']:
            raise HTTPException(status_code=400, detail="Valid action is required (clicked, relevant, irrelevant, skipped, dismissed)")

        if not google_user_id:
            raise HTTPException(status_code=400, detail="google_user_id is required")

        print(f"[Feedback] Recording {action} action for user {google_user_id}")

        # Get user_id (UUID) from google_user_id for v2 schema compatibility
        user_id = None
        try:
            user_result = sb.table("users").select("id").eq("google_user_id", google_user_id).limit(1).execute()
            user_data = getattr(user_result, "data", []) or []
            if user_data:
                user_id = user_data[0].get('id')
        except Exception as e:
            print(f"[Feedback] Warning: Could not get user_id: {e}")

        # Insert feedback record with both user_id (FK) and google_user_id (denormalized)
        feedback_data = {
            'search_query_id': search_query_id,
            'google_user_id': google_user_id,
            'email_id': request.get('email_id'),
            'thread_id': request.get('thread_id'),
            'action': action,
            'similarity_score': request.get('similarity_score'),
            'rank_position': request.get('rank_position'),
            'dwell_time_ms': request.get('dwell_time_ms'),
            'metadata': json.dumps(request.get('metadata')) if request.get('metadata') else None,
        }

        # Add user_id if found (v2 schema compatibility)
        if user_id:
            feedback_data['user_id'] = user_id

        result = sb.table("search_feedback").insert(feedback_data).execute()
        feedback_id = None
        if hasattr(result, 'data') and result.data:
            feedback_id = result.data[0].get('id')

        print(f"[Feedback] Recorded feedback: {feedback_id}")

        # Update sender affinity if we have email information
        thread_id = request.get('thread_id')
        if thread_id and action in ['clicked', 'relevant']:
            try:
                # Get email details to extract sender
                email_result = sb.table("emails").select("sender, encrypted_content, iv").eq("thread_id", thread_id).eq("google_user_id", google_user_id).limit(1).execute()
                email_data = getattr(email_result, "data", []) or []

                if email_data:
                    email = email_data[0]
                    sender = email.get('sender')

                    # If sender is encrypted, we skip affinity update (would need decryption key)
                    if sender and '@' in sender:
                        sender_email = sender
                        sender_domain = sender.split('@')[1] if '@' in sender else None

                        # Upsert sender affinity
                        affinity_result = sb.table("sender_affinity").select("*").eq("google_user_id", google_user_id).eq("sender_email", sender_email).limit(1).execute()
                        affinity_data = getattr(affinity_result, "data", []) or []

                        if affinity_data:
                            # Update existing affinity
                            existing = affinity_data[0]
                            clicked_count = existing.get('clicked_count', 0)
                            marked_relevant = existing.get('marked_relevant', 0)
                            marked_irrelevant = existing.get('marked_irrelevant', 0)

                            if action == 'clicked':
                                clicked_count += 1
                            elif action == 'relevant':
                                marked_relevant += 1

                            # Calculate affinity score (simple formula: clicks + 2*relevant / total_emails)
                            total_emails = existing.get('total_emails', 0)
                            affinity_score = (clicked_count + 2 * marked_relevant) / max(total_emails, 1)
                            affinity_score = min(1.0, affinity_score)  # Cap at 1.0

                            sb.table("sender_affinity").update({
                                'clicked_count': clicked_count,
                                'marked_relevant': marked_relevant,
                                'affinity_score': affinity_score,
                                'last_interaction': datetime.now().isoformat(),
                            }).eq("google_user_id", google_user_id).eq("sender_email", sender_email).execute()

                            print(f"[Feedback] Updated sender affinity for {sender_email}: score={affinity_score:.4f}")
                        else:
                            # Create new affinity record with user_id for v2 schema
                            affinity_insert = {
                                'google_user_id': google_user_id,
                                'sender_email': sender_email,
                                'sender_domain': sender_domain,
                                'total_emails': 1,
                                'clicked_count': 1 if action == 'clicked' else 0,
                                'marked_relevant': 1 if action == 'relevant' else 0,
                                'marked_irrelevant': 0,
                                'affinity_score': 0.5,  # Initial score
                                'last_interaction': datetime.now().isoformat(),
                            }

                            # Add user_id if found (v2 schema compatibility)
                            if user_id:
                                affinity_insert['user_id'] = user_id

                            sb.table("sender_affinity").insert(affinity_insert).execute()

                            print(f"[Feedback] Created sender affinity for {sender_email}")

            except Exception as e:
                print(f"[Feedback] Warning: Could not update sender affinity: {e}")

        # Update user search preferences (CTR)
        if action == 'clicked':
            try:
                prefs_result = sb.table("user_search_preferences").select("*").eq("google_user_id", google_user_id).limit(1).execute()
                prefs_data = getattr(prefs_result, "data", []) or []

                if prefs_data:
                    prefs = prefs_data[0]
                    total_searches = prefs.get('total_searches', 0)
                    total_clicks = prefs.get('total_clicks', 0) + 1

                    # Calculate new CTR
                    avg_ctr = total_clicks / max(total_searches, 1)

                    sb.table("user_search_preferences").update({
                        'total_clicks': total_clicks,
                        'avg_ctr': avg_ctr,
                        'last_updated': datetime.now().isoformat(),
                    }).eq("google_user_id", google_user_id).execute()

                    print(f"[Feedback] Updated user CTR: {avg_ctr:.4f} ({total_clicks}/{total_searches})")

            except Exception as e:
                print(f"[Feedback] Warning: Could not update user preferences: {e}")

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": f"Feedback recorded: {action}"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Exception in /search-feedback: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


def _analyze_question(question: str) -> tuple[list[str], str]:
    """Analyze user question to extract search terms and time context."""
    question_lower = question.lower()
    
    # Extract time-related terms
    time_context = ""
    if any(term in question_lower for term in ['this week', 'this week\'s', 'past week']):
        time_context = "newer_than:7d"
    elif any(term in question_lower for term in ['last week', 'previous week']):
        time_context = "newer_than:14d older_than:7d"
    elif any(term in question_lower for term in ['this month', 'past month']):
        time_context = "newer_than:30d"
    elif any(term in question_lower for term in ['last month', 'previous month']):
        time_context = "newer_than:60d older_than:30d"
    elif any(term in question_lower for term in ['today', 'this morning', 'this afternoon']):
        time_context = "newer_than:1d"
    elif any(term in question_lower for term in ['yesterday']):
        time_context = "newer_than:2d older_than:1d"
    elif any(term in question_lower for term in ['recent', 'recently', 'latest', 'newest']):
        time_context = "newer_than:7d"  # "recent" defaults to last 7 days
    
    # Handle patterns like "about [entity] news" - prioritize the entity over generic "news"
    import re

    # Pattern to match "about [entity] news" and prioritize the entity
    about_pattern = r'about\s+([^?\s]+(?:\s+[^?\s]+)*?)\s+news'
    match = re.search(about_pattern, question_lower)
    if match:
        entity = match.group(1).strip()
        print(f"[Chat] DEBUG: Detected 'about {entity} news' pattern")
        # Replace the pattern with just the entity to avoid generic news matching
        question_lower = re.sub(about_pattern, entity, question_lower)
        print(f"[Chat] DEBUG: Cleaned query: '{question_lower}'")

    # Handle common abbreviations and variations AFTER pattern detection
    question_lower = question_lower.replace('nyt', 'new york times')
    question_lower = question_lower.replace('ny times', 'new york times')
    question_lower = question_lower.replace('times newsletter', 'new york times')
    question_lower = question_lower.replace('times news', 'new york times news')

    # Extract key search terms (remove common words but keep time and topic words)
    stop_words = {'what', 'did', 'i', 'receive', 'get', 'about', 'from', 'that', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
    # Note: Removed 'this', 'week', 'news' from stop words to preserve important query terms

    # Split and clean the question
    words = question_lower.replace('?', '').replace('.', '').split()
    search_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    return search_terms, time_context


def _apply_time_filter(search_results: list, time_context: str) -> list:
    """Apply time filtering to search results based on created_at timestamp."""
    from datetime import datetime, timezone, timedelta
    
    if not search_results:
        return search_results
    
    now = datetime.now(timezone.utc)
    filtered_results = []
    
    for result in search_results:
        created_at_str = result.get('created_at')
        if not created_at_str:
            continue
            
        try:
            # Parse the timestamp - handle both Z and +00:00 formats
            if created_at_str.endswith('Z'):
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            else:
                created_at = datetime.fromisoformat(created_at_str)
            
            # Apply time filtering based on the time context
            if time_context == "newer_than:7d":
                if created_at > now - timedelta(days=7):
                    filtered_results.append(result)
            elif time_context == "newer_than:14d older_than:7d":
                if now - timedelta(days=14) < created_at <= now - timedelta(days=7):
                    filtered_results.append(result)
            elif time_context == "newer_than:30d":
                if created_at > now - timedelta(days=30):
                    filtered_results.append(result)
            elif time_context == "newer_than:60d older_than:30d":
                if now - timedelta(days=60) < created_at <= now - timedelta(days=30):
                    filtered_results.append(result)
            elif time_context == "newer_than:1d":
                if created_at > now - timedelta(days=1):
                    filtered_results.append(result)
            elif time_context == "newer_than:2d older_than:1d":
                if now - timedelta(days=2) < created_at <= now - timedelta(days=1):
                    filtered_results.append(result)
            else:
                # If we don't recognize the time context, include the result
                filtered_results.append(result)
                
        except Exception as e:
            print(f"[Chat] Error parsing timestamp {created_at_str}: {e}")
            # If we can't parse the timestamp, include the result to be safe
            filtered_results.append(result)
    
    return filtered_results


def _generate_gmail_url(thread_id: str) -> str:
    """Generate Gmail URL for a thread ID."""
    # Convert thread ID to hex format if needed (Gmail uses hex)
    try:
        # If thread_id is numeric, convert to hex
        if thread_id.isdigit():
            hex_id = hex(int(thread_id))[2:]
        else:
            hex_id = thread_id
        return f"https://mail.google.com/mail/u/0/#inbox/{hex_id}"
    except:
        # Fallback: use thread_id as is
        return f"https://mail.google.com/mail/u/0/#inbox/{thread_id}"

def _perform_keyword_search(sb: Client, query: str, user_id: Optional[str] = None) -> list:
    """
    Perform full-text keyword search to find emails containing exact query terms.
    """
    print(f"[Keyword Search] Searching for exact terms in: '{query}'")
    
    # Extract meaningful keywords from query
    stop_words = {'all', 'show', 'me', 'find', 'get', 'news', 'about', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [term for term in query.lower().split() if term not in stop_words and len(term) > 2]
    
    if not keywords:
        print(f"[Keyword Search] No meaningful keywords found in query")
        return []
    
    print(f"[Keyword Search] Extracted keywords: {keywords}")
    
    # Build keyword search query - search in subject, sender, and content
    keyword_results = []
    for keyword in keywords:
        try:
            # Search in subject
            query_builder = sb.table("emails").select("*").ilike("subject", f"%{keyword}%")
            if user_id:
                query_builder = query_builder.eq("google_user_id", user_id)
            subject_results = query_builder.limit(20).execute()
            
            # Search in sender
            query_builder = sb.table("emails").select("*").ilike("sender", f"%{keyword}%")
            if user_id:
                query_builder = query_builder.eq("google_user_id", user_id)
            sender_results = query_builder.limit(20).execute()
            
            # Search in content (if available)
            query_builder = sb.table("emails").select("*").ilike("content", f"%{keyword}%")
            if user_id:
                query_builder = query_builder.eq("google_user_id", user_id)
            content_results = query_builder.limit(20).execute()
            
            # Combine results
            for result_set in [subject_results, sender_results, content_results]:
                results = getattr(result_set, "data", []) or []
                keyword_results.extend(results)
                
        except Exception as e:
            print(f"[Keyword Search] Error searching for keyword '{keyword}': {e}")
            continue
    
    # Deduplicate by thread_id
    seen_threads = set()
    unique_results = []
    for result in keyword_results:
        thread_id = result.get('thread_id')
        if thread_id and thread_id not in seen_threads:
            seen_threads.add(thread_id)
            unique_results.append(result)
    
    print(f"[Keyword Search] Found {len(unique_results)} unique results")
    return unique_results


def _combine_search_candidates(vector_results: list, keyword_results: list) -> list:
    """
    Combine vector and keyword search results into unique candidates.
    """
    print(f"[Hybrid Search] Combining {len(vector_results)} vector + {len(keyword_results)} keyword results")
    
    # Use thread_id as the unique identifier
    seen_threads = set()
    combined_results = []
    
    # Add vector results first (they have similarity scores)
    for result in vector_results:
        thread_id = result.get('thread_id')
        if thread_id and thread_id not in seen_threads:
            seen_threads.add(thread_id)
            result['source'] = 'vector'
            combined_results.append(result)
    
    # Add keyword results that weren't already found by vector search
    for result in keyword_results:
        thread_id = result.get('thread_id')
        if thread_id and thread_id not in seen_threads:
            seen_threads.add(thread_id)
            result['source'] = 'keyword'
            result['similarity'] = 0.0  # Default similarity for keyword-only results
            combined_results.append(result)
    
    print(f"[Hybrid Search] Combined to {len(combined_results)} unique candidates")
    return combined_results


def _calculate_keyword_boost(query: str, subject: str, sender: str, content: str) -> float:
    """
    Calculate keyword match boost based on exact term presence.
    Returns a boost score between 0.0 and 2.0.
    """
    query_lower = query.lower()
    subject_lower = subject.lower()
    sender_lower = sender.lower()
    content_lower = content.lower()

    # Extract meaningful keywords from query
    stop_words = {'all', 'show', 'me', 'find', 'get', 'news', 'about', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [term for term in query_lower.split() if term not in stop_words and len(term) > 2]

    if not keywords:
        return 0.0

    boost = 0.0
    total_possible_boost = 0.0

    for keyword in keywords:
        # Different weights for different locations
        if keyword in subject_lower:
            boost += 0.8  # Subject matches are very valuable
            total_possible_boost += 0.8
        elif keyword in sender_lower:
            boost += 0.5  # Sender matches are valuable
            total_possible_boost += 0.5
        elif keyword in content_lower:
            boost += 0.3  # Content matches are good
            total_possible_boost += 0.3
        else:
            total_possible_boost += 0.8  # Max possible for this keyword

    # Normalize boost based on keyword coverage
    if total_possible_boost > 0:
        coverage_ratio = boost / total_possible_boost
        # Scale final boost: full coverage = +2.0, partial coverage scales down
        final_boost = coverage_ratio * 2.0
    else:
        final_boost = 0.0

    return min(2.0, final_boost)


def _prefilter_candidates(candidates: list, query: str) -> list:
    """
    Enhanced pre-filtering with stricter entity matching and negative filtering.
    Removes clearly irrelevant candidates before expensive AI re-ranking.
    """
    if not candidates:
        return candidates

    print(f"[Pre-filtering] Starting with {len(candidates)} candidates")
    query_lower = query.lower()

    # Enhanced entity detection
    entity_indicators = ['anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia', 'claude']
    query_entities = [entity for entity in entity_indicators if entity in query_lower]
    has_specific_entity = len(query_entities) > 0

    # Negative filters - patterns that indicate irrelevant content
    negative_patterns = {
        'anthropic': ['breaking news', 'new york times', 'nyt', 'palestinian', 'trump', 'jimmy kimmel', 'half baked', 'startup ideas'],
        'openai': ['breaking news', 'new york times', 'trump', 'palestinian'],
        'google': ['breaking news', 'new york times'] if 'news' in query_lower else [],
        'h-1b': ['half baked', 'startup', 'birchbox', 'times sale', 'subscription offer'],
        'visa': ['subscription', 'newsletter signup', 'fantasy football'] if 'immigration' in query_lower or 'work' in query_lower else []
    }

    filtered = []

    for candidate in candidates:
        subject = candidate.get('subject', '').lower()
        sender = candidate.get('sender', '').lower()
        content = candidate.get('content', '')[:1000].lower()
        full_text = f"{subject} {sender} {content}"

        # For entity-specific queries, apply strict filtering
        if has_specific_entity:
            # Must have entity match for entity queries
            entity_match = any(entity in full_text for entity in query_entities)

            if not entity_match:
                # Skip if no entity match found
                print(f"[Pre-filtering] Skipping '{subject[:50]}...' - no entity match for {query_entities}")
                continue

            # Check negative patterns for this entity
            skip_due_to_negative = False
            for entity in query_entities:
                if entity in negative_patterns:
                    for pattern in negative_patterns[entity]:
                        if pattern in full_text:
                            print(f"[Pre-filtering] Skipping '{subject[:50]}...' - negative pattern '{pattern}' for entity '{entity}'")
                            skip_due_to_negative = True
                            break
                    if skip_due_to_negative:
                        break

            if skip_due_to_negative:
                continue

        else:
            # For non-entity queries, use keyword matching
            query_terms = [term for term in query_lower.split() if len(term) > 3]
            keyword_match = any(term in full_text for term in query_terms)

            if not keyword_match and candidate.get('source') != 'keyword':
                print(f"[Pre-filtering] Skipping '{subject[:50]}...' - no keyword match")
                continue

        # Additional quality filters
        if _is_low_quality_content(subject, sender, content):
            print(f"[Pre-filtering] Skipping '{subject[:50]}...' - low quality content")
            continue

        # Calculate relevance score for ranking within filtered results
        relevance_score = _calculate_prefilter_score(candidate, query_lower, query_entities)
        candidate['prefilter_score'] = relevance_score

        filtered.append(candidate)

    # Sort by prefilter score and limit to top candidates
    filtered.sort(key=lambda x: x.get('prefilter_score', 0), reverse=True)
    max_candidates = 10 if has_specific_entity else 15  # Fewer for entity queries

    result = filtered[:max_candidates]
    print(f"[Pre-filtering] Filtered {len(candidates)} -> {len(result)} candidates")

    return result


def _is_low_quality_content(subject: str, sender: str, content: str) -> bool:
    """
    Detect low-quality content that should be filtered out.
    """
    subject_lower = subject.lower()
    sender_lower = sender.lower()

    # Generic promotional content
    promo_indicators = ['sale', '$1/wk', 'best offer', 'subscribe', 'unsubscribe', 'promotional']
    if any(indicator in subject_lower for indicator in promo_indicators):
        return True

    # Empty or very short content
    if len(content.strip()) < 50:
        return True

    # Suspicious sender patterns
    suspicious_senders = ['noreply', 'no-reply', 'donotreply']
    if any(pattern in sender_lower for pattern in suspicious_senders) and 'times' in sender_lower:
        return True

    return False


def _calculate_prefilter_score(candidate: dict, query_lower: str, query_entities: list) -> float:
    """
    Calculate a quick relevance score for pre-filtering ranking.
    """
    subject = candidate.get('subject', '').lower()
    sender = candidate.get('sender', '').lower()
    content = candidate.get('content', '')[:500].lower()
    full_text = f"{subject} {sender} {content}"

    score = 0.0

    # Entity matching bonus (highest priority)
    if query_entities:
        entity_matches = sum(1 for entity in query_entities if entity in full_text)
        score += entity_matches * 5.0

        # Subject line entity match gets extra bonus
        subject_entity_matches = sum(1 for entity in query_entities if entity in subject)
        score += subject_entity_matches * 3.0

    # Keyword matching
    query_words = [word for word in query_lower.split() if len(word) > 3]
    if query_words:
        keyword_matches = sum(1 for word in query_words if word in full_text)
        score += keyword_matches * 2.0

        # Subject keyword matches get bonus
        subject_keyword_matches = sum(1 for word in query_words if word in subject)
        score += subject_keyword_matches * 1.5

    # Source bonus
    if candidate.get('source') == 'keyword':
        score += 2.0

    # Vector similarity bonus
    similarity = candidate.get('similarity', 0)
    if similarity > 0.5:
        score += similarity * 2.0

    return score


def _get_cache_key(query: str, email_id: str) -> str:
    """
    Generate a cache key for query-email pair.
    """
    import hashlib
    combined = f"{query.lower().strip()}:{email_id}"
    return hashlib.md5(combined.encode()).hexdigest()


def _get_cached_score(query: str, email_id: str) -> Optional[float]:
    """
    Get cached AI score for query-email pair if still valid.
    """
    cache_key = _get_cache_key(query, email_id)

    if cache_key in AI_SCORE_CACHE:
        cached_data = AI_SCORE_CACHE[cache_key]
        timestamp = cached_data.get('timestamp', 0)

        # Check if cache is still valid
        if time.time() - timestamp < CACHE_TTL:
            return cached_data.get('score')
        else:
            # Remove expired entry
            del AI_SCORE_CACHE[cache_key]

    return None


def _cache_score(query: str, email_id: str, score: float):
    """
    Cache AI score for query-email pair.
    """
    cache_key = _get_cache_key(query, email_id)
    AI_SCORE_CACHE[cache_key] = {
        'score': score,
        'timestamp': time.time()
    }


def _cleanup_expired_cache():
    """
    Remove expired entries from cache to prevent memory buildup.
    """
    current_time = time.time()
    expired_keys = [
        key for key, data in AI_SCORE_CACHE.items()
        if current_time - data.get('timestamp', 0) > CACHE_TTL
    ]

    for key in expired_keys:
        del AI_SCORE_CACHE[key]

    if expired_keys:
        print(f"[Cache] Cleaned up {len(expired_keys)} expired entries")


def _ai_rerank_candidates(candidates: list, original_query: str) -> list:
    """
    Use GPT-5 nano to re-rank candidates with caching for improved performance.
    """
    print(f"[AI Re-ranking] Re-ranking {len(candidates)} candidates for query: '{original_query}'")

    # Cleanup expired cache entries periodically
    _cleanup_expired_cache()

    # Check if OpenAI is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        print("[AI Re-ranking] OpenAI not available, using fallback scoring")
        return _fallback_score_candidates(candidates, original_query)

    client = OpenAI(api_key=api_key)
    reranked_candidates = []

    # Separate cached and uncached candidates
    cached_candidates = []
    uncached_candidates = []

    for candidate in candidates:
        email_id = candidate.get('thread_id', candidate.get('id', ''))
        cached_score = _get_cached_score(original_query, email_id)

        if cached_score is not None:
            candidate['_cached_ai_score'] = cached_score
            cached_candidates.append(candidate)
        else:
            uncached_candidates.append(candidate)

    print(f"[AI Re-ranking] Cache hits: {len(cached_candidates)}, Cache misses: {len(uncached_candidates)}")

    # Process uncached candidates in batches and store scores
    batch_size = 5
    uncached_scores = {}

    for batch_start in range(0, len(uncached_candidates), batch_size):
        batch = uncached_candidates[batch_start:batch_start + batch_size]
        batch_scores = _batch_score_emails(client, batch, original_query, batch_start)

        # Store scores and cache for future use
        for i, candidate in enumerate(batch):
            ai_score = batch_scores.get(i, 5.0)
            uncached_index = batch_start + i
            uncached_scores[uncached_index] = ai_score

            # Cache score for future queries
            email_id = candidate.get('thread_id', candidate.get('id', ''))
            if email_id:
                _cache_score(original_query, email_id, ai_score)

    # Process all candidates (cached + newly scored)
    all_candidates = cached_candidates + uncached_candidates

    for i, candidate in enumerate(all_candidates):
        # Get AI score (from cache or fresh scoring)
        if i < len(cached_candidates):
            # Cached candidate
            ai_score = candidate['_cached_ai_score']
            del candidate['_cached_ai_score']  # Clean up temp field
            cache_status = "cached"
        else:
            # Uncached candidate - get from batch processing results
            uncached_index = i - len(cached_candidates)
            ai_score = uncached_scores.get(uncached_index, 5.0)
            cache_status = "fresh"

        # Apply keyword match boosting
        subject = candidate.get('subject', '')
        sender = candidate.get('sender', '')
        content_preview = candidate.get('content', '')[:500]
        keyword_boost = _calculate_keyword_boost(original_query, subject, sender, content_preview)
        final_score = min(10.0, ai_score + keyword_boost)

        candidate['relevance_score'] = final_score
        candidate['ai_score'] = ai_score
        candidate['keyword_boost'] = keyword_boost
        reranked_candidates.append(candidate)

        print(f"[AI Re-ranking] Email {i + 1}: '{subject[:50]}...' -> AI Score: {ai_score} ({cache_status}), Keyword Boost: {keyword_boost}, Final: {final_score}")

    # Sort by relevance score (highest first)
    reranked_candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    cache_hit_rate = len(cached_candidates) / len(candidates) * 100 if candidates else 0
    print(f"[AI Re-ranking] Re-ranking complete. Top score: {reranked_candidates[0].get('relevance_score', 0) if reranked_candidates else 'N/A'}, Cache hit rate: {cache_hit_rate:.1f}%")
    return reranked_candidates


def _batch_score_emails(client: OpenAI, batch: list, query: str, batch_start: int) -> dict:
    """
    Score a batch of emails in a single API call for efficiency.
    Returns dict mapping batch index to AI score.
    """
    # Build batch prompt with simplified format for GPT-5 nano
    emails_text = []
    for i, candidate in enumerate(batch):
        subject = candidate.get('subject', 'No Subject')
        sender = candidate.get('sender', 'Unknown Sender')
        content_preview = candidate.get('content', '')[:300]  # Shorter for batch processing

        emails_text.append(f"Email {i+1}:")
        emails_text.append(f"Subject: {subject}")
        emails_text.append(f"From: {sender}")
        emails_text.append(f"Content: {content_preview}")
        emails_text.append("")

    # Simplified prompt optimized for GPT-5 nano
    batch_prompt = f"""Score these emails for relevance to query: "{query}"

{chr(10).join(emails_text)}

Rate each email 1-10 for relevance:
- 9-10: Perfect match (exact entity + topic)
- 7-8: Good match (entity present)
- 5-6: Moderate match
- 3-4: Weak match
- 1-2: No match

Respond with only JSON array of numbers: [score1, score2, score3, ...]"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": batch_prompt}],
            max_completion_tokens=50
        )

        response_text = response.choices[0].message.content.strip()
        print(f"[Batch Scoring] Batch {batch_start//5 + 1} response: {response_text}")

        # Parse JSON array response
        import json
        scores = json.loads(response_text)

        # Validate and return scores
        result = {}
        for i, score in enumerate(scores):
            if isinstance(score, (int, float)) and 1 <= score <= 10:
                result[i] = float(score)
            else:
                print(f"[Batch Scoring] Invalid score {score} for email {i+1}, using fallback")
                result[i] = _fallback_single_score(batch[i] if i < len(batch) else {}, query)

        return result

    except Exception as e:
        print(f"[Batch Scoring] Error in batch scoring: {e}")
        # Fallback to individual scoring for this batch
        result = {}
        for i, candidate in enumerate(batch):
            result[i] = _fallback_single_score(candidate, query)
        return result


def _fallback_score_candidates(candidates: list, query: str) -> list:
    """
    Fallback scoring system when AI is unavailable.
    Uses rule-based scoring for deterministic results.
    """
    print("[Fallback Scoring] Using rule-based scoring system")

    for i, candidate in enumerate(candidates):
        score = _fallback_single_score(candidate, query)

        # Apply keyword boost
        subject = candidate.get('subject', '')
        sender = candidate.get('sender', '')
        content_preview = candidate.get('content', '')[:500]
        keyword_boost = _calculate_keyword_boost(query, subject, sender, content_preview)
        final_score = min(10.0, score + keyword_boost)

        candidate['relevance_score'] = final_score
        candidate['ai_score'] = score
        candidate['keyword_boost'] = keyword_boost

        print(f"[Fallback Scoring] Email {i+1}: Rule Score: {score}, Keyword Boost: {keyword_boost}, Final: {final_score}")

    # Sort by relevance score
    candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return candidates


def _fallback_single_score(candidate: dict, query: str) -> float:
    """
    Calculate rule-based relevance score for a single email.
    """
    subject = candidate.get('subject', '').lower()
    sender = candidate.get('sender', '').lower()
    content = candidate.get('content', '')[:1000].lower()
    query_lower = query.lower()

    score = 1.0  # Base score

    # Entity matching (high weight)
    entity_indicators = ['anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia', 'claude']
    query_entities = [entity for entity in entity_indicators if entity in query_lower]

    if query_entities:
        full_text = f"{subject} {sender} {content}"
        entity_matches = sum(1 for entity in query_entities if entity in full_text)
        if entity_matches > 0:
            score += 4.0 * (entity_matches / len(query_entities))  # Up to +4 points

    # Keyword matching (medium weight)
    query_words = [word for word in query_lower.split() if len(word) > 3]
    if query_words:
        full_text = f"{subject} {sender} {content}"
        keyword_matches = sum(1 for word in query_words if word in full_text)
        score += 2.0 * (keyword_matches / len(query_words))  # Up to +2 points

    # Subject line bonus (high relevance indicator)
    if any(word in subject for word in query_words):
        score += 1.5

    # Sender relevance
    if any(word in sender for word in query_words):
        score += 1.0

    # Source bonus (keyword search results are more targeted)
    if candidate.get('source') == 'keyword':
        score += 0.5

    return min(10.0, score)


def _detect_unsubscribe_link(content: str, headers: dict = None) -> bool:
    """Detect if email has unsubscribe link or headers (indicates newsletter/marketing)."""
    if not content:
        return False

    content_lower = content.lower()

    # Check for unsubscribe links in content
    unsubscribe_patterns = [
        'unsubscribe', 'opt-out', 'opt out', 'manage preferences',
        'update your preferences', 'email preferences', 'stop receiving'
    ]

    if any(pattern in content_lower for pattern in unsubscribe_patterns):
        return True

    # Check for List-Unsubscribe header (standard for bulk email)
    if headers and 'list-unsubscribe' in [k.lower() for k in headers.keys()]:
        return True

    return False


def _is_automated_sender(sender: str) -> bool:
    """Detect if sender is automated/marketing (not a real person)."""
    if not sender:
        return False

    sender_lower = sender.lower()

    # Check for noreply/no-reply anywhere in the email (not just at start)
    if 'noreply' in sender_lower or 'no-reply' in sender_lower or 'no_reply' in sender_lower:
        return True
    if 'donotreply' in sender_lower or 'do-not-reply' in sender_lower or 'do_not_reply' in sender_lower:
        return True

    # Common automated sender patterns (must be at start of local part)
    automated_patterns = [
        'newsletter@', 'news@', 'marketing@',
        'notifications@', 'notify@', 'alerts@',
        'info@', 'support@', 'help@',
        'team@', 'hello@', 'hi@',
        'automated@', 'auto@',
        'direct@', 'fromthetimes@', 'mailer@',
        'careerservice@', 'careers@', 'jobs@'
    ]

    return any(pattern in sender_lower for pattern in automated_patterns)


def _has_bulk_headers(headers: dict) -> bool:
    """Check for bulk/marketing email headers."""
    if not headers:
        return False

    # Normalize header keys to lowercase
    headers_lower = {k.lower(): v for k, v in headers.items()}

    # Check for bulk email indicators
    if 'precedence' in headers_lower:
        precedence = str(headers_lower['precedence']).lower()
        if precedence in ['bulk', 'list', 'junk']:
            return True

    # Check for list headers
    list_headers = ['list-id', 'list-post', 'list-help', 'list-subscribe']
    if any(header in headers_lower for header in list_headers):
        return True

    return False


def _calculate_importance_score(email_data: dict) -> int:
    """
    Calculate importance score (0-100) based on email metadata.
    Higher score = more important/personal, Lower score = marketing/automated
    """
    score = 50  # baseline

    content = email_data.get('content', '')
    sender = email_data.get('sender', '')
    headers = email_data.get('headers', {})

    # Marketing/automated signals (negative)
    if _detect_unsubscribe_link(content, headers):
        score -= 20

    if _is_automated_sender(sender):
        score -= 15

    if _has_bulk_headers(headers):
        score -= 10

    # Positive signals
    # Short content often means personal email (not marketing copy)
    if content and len(content) < 500:
        score += 5

    # Personal domain patterns (simple heuristic)
    if sender and '@' in sender:
        domain = sender.split('@')[-1].lower()
        personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com']
        if any(personal_domain in domain for personal_domain in personal_domains):
            # But only if not automated sender
            if not _is_automated_sender(sender):
                score += 10

    return max(0, min(100, score))


def _prepare_email_context(search_results: list, original_question: str) -> str:
    """Prepare email context for the LLM by formatting relevant emails."""
    if not search_results:
        return "No relevant emails found for this question."
    
    context_parts = []
    context_parts.append(f"User asked: {original_question}")
    context_parts.append(f"Found {len(search_results)} relevant emails:")
    context_parts.append("")
    
    for i, result in enumerate(search_results[:10]):  # Limit to top 10 most relevant
        subject = result.get('subject', 'No Subject')
        sender = result.get('sender', 'Unknown Sender')
        content = result.get('content', '')
        similarity = result.get('similarity', 0)
        created_at = result.get('created_at', 'Unknown Date')
        thread_id = result.get('thread_id', '')
        importance_score = result.get('importance_score', 50)
        is_automated = result.get('is_automated', False)
        has_unsubscribe = result.get('has_unsubscribe', False)

        # Truncate content to avoid token limits
        content_preview = content[:500] + "..." if len(content) > 500 else content

        # Generate Gmail URL
        gmail_url = _generate_gmail_url(thread_id) if thread_id else ""

        # Determine email type based on importance metadata
        email_type = "Personal/Important"
        if is_automated or has_unsubscribe or importance_score < 40:
            email_type = "Newsletter/Marketing"

        context_parts.append(f"Email {i+1}:")
        context_parts.append(f"  From: {sender}")
        context_parts.append(f"  Subject: {subject}")
        context_parts.append(f"  Date: {created_at}")
        context_parts.append(f"  Type: {email_type} (Importance Score: {importance_score}/100)")
        context_parts.append(f"  Gmail URL: {gmail_url}")
        context_parts.append(f"  Relevance: {similarity:.2f}")
        context_parts.append(f"  Content: {content_preview}")
        context_parts.append("")
    
    return "\n".join(context_parts)


@app.post("/sync-inbox")
async def sync_inbox(request: dict, x_encryption_key: Optional[str] = Header(None)):
    """Fetch recent Gmail threads, embed content, and upsert into Supabase.

    Expected request body:
      { "userId": "..." (optional), "range": "24h|7d|30d" (optional) }

    - When a range is provided, we filter Gmail threads using the search query
      parameter (e.g., newer_than:1d) so we only fetch recent content.
    - We deduplicate by skipping threads already present in the `emails` table
      and only count newly indexed threads in the response.
    """
    try:
        print(f"[Sync] Starting inbox sync with request: {request}")
        sb = _get_supabase()

        user_id = request.get("userId")
        selected_range = (request.get("range") or "").strip().lower()
        # Resolve a default google_user_id if one wasn't passed
        if not user_id:
            try:
                res_uid = sb.table("users").select("google_user_id").limit(1).execute()  # type: ignore[attr-defined]
                rows_uid = getattr(res_uid, "data", []) or []
                if rows_uid:
                    user_id = rows_uid[0].get("google_user_id") or None
            except Exception as _e:
                print(f"[Sync] Could not resolve default google_user_id: {_e}")
        print(f"[Sync] User ID: {user_id}")
        print(f"[Sync] Selected range: {selected_range!r}")

        # Build Gmail service
        print("[Sync] Building Gmail service...")
        service = _build_gmail_service(user_id)
        print("[Sync] Gmail service built successfully")

        # Build Gmail query from range selector
        gmail_q = None
        if selected_range in ("24h", "1d", "day"):
            gmail_q = "newer_than:1d"
        elif selected_range in ("7d", "7days", "week"):
            gmail_q = "newer_than:7d"
        elif selected_range in ("30d", "30days", "month"):
            gmail_q = "newer_than:30d"

        # List all threads in the time range (with pagination)
        print("[Sync] Fetching threads list...")
        threads = []
        page_token = None

        while True:
            if gmail_q:
                if page_token:
                    threads_resp = service.users().threads().list(userId="me", maxResults=500, q=gmail_q, pageToken=page_token).execute()
                else:
                    threads_resp = service.users().threads().list(userId="me", maxResults=500, q=gmail_q).execute()
            else:
                if page_token:
                    threads_resp = service.users().threads().list(userId="me", maxResults=500, pageToken=page_token).execute()
                else:
                    threads_resp = service.users().threads().list(userId="me", maxResults=500).execute()

            batch_threads = threads_resp.get('threads', []) or []
            threads.extend(batch_threads)
            print(f"[Sync] Fetched {len(batch_threads)} threads (total so far: {len(threads)})")

            page_token = threads_resp.get('nextPageToken')
            if not page_token:
                break

        print(f"[Sync] Found {len(threads)} total threads")
        
        # Debug: Log thread IDs and basic info
        for i, thread in enumerate(threads[:5]):  # Log first 5 for debugging
            thread_id = thread.get('id', 'Unknown')
            print(f"[Sync] Thread {i+1}: ID={thread_id}")

        if not threads:
            print("[Sync] No threads found, returning 0")
            return {"status": "success", "indexed_count": 0}

        # Determine which threads are new by comparing with Supabase
        thread_ids = [t.get('id') for t in threads if t.get('id')]
        existing_ids: set[str] = set()
        if thread_ids:
            try:
                query = sb.table("emails").select("thread_id")  # type: ignore[attr-defined]
                if user_id:
                    query = query.eq("google_user_id", user_id)
                existing = query.in_("thread_id", thread_ids).execute()  # type: ignore[attr-defined]
                rows = getattr(existing, "data", []) or []
                existing_ids = {r.get("thread_id") for r in rows if r.get("thread_id")}
            except Exception as e:
                print(f"[Sync] Warning: failed to query existing thread ids: {e}")
        new_thread_ids = [tid for tid in thread_ids if tid not in existing_ids]

        print(f"[Sync] New threads to index: {len(new_thread_ids)}; existing skipped: {len(existing_ids)}")

        if not new_thread_ids:
            return {"status": "success", "indexed_count": 0, "skipped_existing": len(existing_ids)}

        print("[Sync] Loading embedding model...")
        model = _get_embedding_model()
        print("[Sync] Embedding model loaded")

        indexed = 0
        for i, tid in enumerate(new_thread_ids):
            if not tid:
                print(f"[Sync] Thread {i+1}/{len(new_thread_ids)}: No ID, skipping")
                continue
            
            print(f"[Sync] Processing thread {i+1}/{len(new_thread_ids)}: {tid}")
            try:
                thread_full = service.users().threads().get(userId="me", id=tid, format="full").execute()
                
                # Debug: Extract subject and sender for logging
                messages = thread_full.get("messages", [])
                if messages:
                    headers = messages[0].get("payload", {}).get("headers", [])
                    subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
                    sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown Sender")
                    print(f"[Sync] Thread {tid}: '{subject}' from {sender}")
                
                print(f"[Sync] Thread {tid}: Fetched full thread data ({len(messages)} messages in thread)")
                
                content = _fetch_thread_plaintext(service, tid)
                print(f"[Sync] Thread {tid}: Extracted content ({len(content)} chars)")
                
                meta = _parse_thread_metadata(thread_full)
                subject = meta.get('subject', '')
                sender = meta.get('from', '')

                # Extract email date from the first message
                email_date = None
                if messages:
                    # Get internalDate (milliseconds since epoch) from first message
                    internal_date_ms = messages[0].get('internalDate')
                    if internal_date_ms:
                        from datetime import datetime, timezone
                        email_date = datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc).isoformat()

                print(f"[Sync] Thread {tid}: Subject='{subject[:50]}...', Sender='{sender}', Date='{email_date}'")

                # Classify email into categories using LLM
                print(f"[Sync] Thread {tid}: Classifying email categories")
                categories = _classify_email_categories(subject, content, sender)

                # Calculate importance score based on metadata
                email_data = {
                    'content': content,
                    'sender': sender,
                    'headers': {}  # TODO: Extract headers from Gmail API if needed
                }
                importance_score = _calculate_importance_score(email_data)
                is_automated = _is_automated_sender(sender)
                has_unsubscribe = _detect_unsubscribe_link(content)
                print(f"[Sync] Thread {tid}: Importance score: {importance_score} (automated: {is_automated}, unsubscribe: {has_unsubscribe})")

                # Create enhanced text for embedding that includes keywords and context
                text_for_embedding = _create_enhanced_embedding_text(subject, content, sender)
                print(f"[Sync] Thread {tid}: Generating embedding for {len(text_for_embedding)} chars")
                emb = model.encode(text_for_embedding, normalize_embeddings=True).tolist()  # 384 dims
                print(f"[Sync] Thread {tid}: Generated embedding with {len(emb)} dimensions")

                # Upsert into Supabase emails table
                # Check if encryption is enabled
                if x_encryption_key:
                    # Zero-knowledge mode: Encrypt sensitive data
                    print(f"[Sync] Thread {tid}: Encrypting with user key (zero-knowledge mode)")

                    # Encrypt email content (subject, sender, content combined)
                    email_json = json.dumps({
                        "subject": subject,
                        "sender": sender,
                        "content": content
                    })
                    encrypted_content, content_iv = _encrypt_with_user_key(email_json, x_encryption_key)

                    # Encrypt embedding vector
                    embedding_json = json.dumps(emb)
                    encrypted_embedding, embedding_iv = _encrypt_with_user_key(embedding_json, x_encryption_key)

                    # Create payload with encrypted fields
                    payload = {
                        "google_user_id": user_id,
                        "thread_id_hash": _hash_thread_id(tid),  # Hashed thread ID
                        "encrypted_content": encrypted_content,
                        "encrypted_embedding": encrypted_embedding,
                        "iv": content_iv,  # Store IV for decryption
                        "sender_domain": _extract_domain(sender),  # Plain metadata
                        "categories": categories,  # Plain metadata
                        "importance_score": importance_score,  # Plain metadata
                        "is_automated": is_automated,  # Plain metadata
                        "has_unsubscribe": has_unsubscribe,  # Plain metadata
                    }
                else:
                    # Standard mode: Store plain text (backwards compatible)
                    print(f"[Sync] Thread {tid}: Storing in plain text mode")
                    payload = {
                        "google_user_id": user_id,
                        "thread_id": tid,
                        "subject": subject,
                        "sender": sender,
                        "content": content,
                        "embedding": emb,
                        "categories": categories,
                        "importance_score": importance_score,
                        "is_automated": is_automated,
                        "has_unsubscribe": has_unsubscribe,
                    }

                # Add created_at if we have it
                if email_date:
                    payload["created_at"] = email_date
                print(f"[Sync] Thread {tid}: Upserting to Supabase...")

                # Use different conflict resolution based on encryption mode
                if x_encryption_key:
                    # Encrypted mode: check for duplicates manually by thread_id_hash
                    thread_hash = _hash_thread_id(tid)
                    existing = sb.table("emails").select("id").eq("google_user_id", user_id).eq("thread_id_hash", thread_hash).execute()
                    if existing.data:
                        # Update existing
                        sb.table("emails").update(payload).eq("id", existing.data[0]["id"]).execute()
                    else:
                        # Insert new
                        sb.table("emails").insert(payload).execute()
                else:
                    # Plain text mode: use standard upsert
                    sb.table("emails").upsert(
                        payload,
                        on_conflict="google_user_id,thread_id",
                    ).execute()  # type: ignore[attr-defined]
                indexed += 1
                print(f"[Sync] Thread {tid}: Successfully indexed ({indexed}/{len(new_thread_ids)})")
            except Exception as e:
                print(f"[Sync] Thread {tid}: ERROR - {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[Sync] Completed: {indexed}/{len(new_thread_ids)} new threads indexed successfully (skipped {len(existing_ids)})")
        return {"status": "success", "indexed_count": indexed, "skipped_existing": len(existing_ids)}
    except HTTPException as he:
        print(f"[ERROR] HTTPException in /sync-inbox: {he.status_code} - {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Exception in /sync-inbox: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sync error: {str(e)}")


@app.get("/autocomplete")
async def autocomplete_suggestions(query: str = "", limit: int = 5):
    """Get autocomplete suggestions for instant search.

    Returns suggestions for:
    - Common senders (emails from @github.com)
    - Common subjects
    - Date ranges (last 7 days, last month, etc.)
    - Search patterns (emails about deployment, emails with attachments)
    """
    try:
        if not query or len(query) < 2:
            # Return default suggestions
            return {
                "suggestions": [
                    {"type": "pattern", "text": "emails from @github.com", "description": "Filter by sender domain"},
                    {"type": "pattern", "text": "emails about deployment", "description": "Search email content"},
                    {"type": "pattern", "text": "emails with attachments", "description": "Filter by attachments"},
                    {"type": "date", "text": "last 7 days", "description": "Recent emails"},
                    {"type": "date", "text": "last month", "description": "Older emails"}
                ]
            }

        sb = _get_supabase()
        suggestions = []
        query_lower = query.lower()

        # Check if query starts with common patterns
        if query_lower.startswith("from:") or query_lower.startswith("emails from"):
            # Suggest sender domains
            try:
                # Extract unique sender domains from recent emails
                recent_emails = sb.table("emails").select("sender").order("created_at", desc=True).limit(100).execute()
                rows = getattr(recent_emails, "data", []) or []

                domains = set()
                for row in rows:
                    sender = row.get("sender", "")
                    if "@" in sender:
                        domain = sender.split("@")[-1].strip(">")
                        domains.add(domain)

                # Filter domains matching query
                search_term = query_lower.replace("from:", "").replace("emails from", "").strip()
                matching_domains = [d for d in domains if search_term in d.lower()][:limit]

                for domain in matching_domains:
                    suggestions.append({
                        "type": "sender",
                        "text": f"emails from @{domain}",
                        "description": f"Emails from {domain}"
                    })
            except Exception as e:
                print(f"[Autocomplete] Error getting sender suggestions: {e}")

        elif query_lower.startswith("about") or query_lower.startswith("emails about"):
            # Suggest common topics from recent emails
            suggestions.extend([
                {"type": "topic", "text": "emails about deployment", "description": "Deployment related emails"},
                {"type": "topic", "text": "emails about meeting", "description": "Meeting invitations"},
                {"type": "topic", "text": "emails about invoice", "description": "Billing and invoices"},
            ])

        # Always include date range suggestions if query matches
        if any(term in query_lower for term in ["last", "recent", "today", "yesterday", "week", "month"]):
            suggestions.extend([
                {"type": "date", "text": "last 7 days", "description": "Emails from last week"},
                {"type": "date", "text": "last 30 days", "description": "Emails from last month"},
                {"type": "date", "text": "today", "description": "Today's emails"},
            ])

        # If no specific suggestions, try to find emails with matching subjects
        if not suggestions:
            try:
                matching_emails = sb.table("emails").select("subject, sender").ilike("subject", f"%{query}%").limit(5).execute()
                rows = getattr(matching_emails, "data", []) or []

                for row in rows[:limit]:
                    subject = row.get("subject", "")
                    sender = row.get("sender", "")
                    if subject:
                        suggestions.append({
                            "type": "subject",
                            "text": subject,
                            "description": f"From {sender}"
                        })
            except Exception as e:
                print(f"[Autocomplete] Error getting subject suggestions: {e}")

        return {"suggestions": suggestions[:limit]}

    except Exception as e:
        print(f"[ERROR] Exception in /autocomplete: {str(e)}")
        return {"suggestions": []}


def _score_email_relevance_with_ai(query: str, email: dict) -> float:
    """Use AI (gpt-4o-mini) to score how relevant an email is to a search query.

    Returns a similarity score between 0.0 and 1.0.
    Uses caching to avoid redundant API calls.
    """
    try:
        # Create cache key from query and email ID
        cache_key = f"{query.lower().strip()}:{email.get('thread_id', '')}"

        # Check cache first
        if cache_key in AI_SCORE_CACHE:
            cached_entry = AI_SCORE_CACHE[cache_key]
            # Check if cache entry is still valid (TTL)
            if time.time() - cached_entry['timestamp'] < CACHE_TTL:
                return cached_entry['score']

        # Prepare email content for scoring
        subject = email.get('subject', '(No Subject)')[:200]
        sender = email.get('sender', 'Unknown')[:100]
        content = email.get('content', '')[:300]
        created_at = email.get('created_at', '')

        # Format date nicely
        try:
            email_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - email_date).days
            date_str = f"{days_ago} days ago" if days_ago > 0 else "today"
        except:
            date_str = created_at

        # Create scoring prompt
        prompt = f"""Score how relevant this email is to the search query. Consider:
- Does the sender match or relate to the query?
- Does the subject match or relate to the query?
- Does the content match or relate to the query?
- Is the email recent (if recency matters for the query)?

Query: "{query}"

Email:
- Sender: {sender}
- Subject: {subject}
- Content: {content}
- Date: {date_str}

Return ONLY a decimal number between 0.0 and 1.0 representing the relevance score. No explanation, just the number."""

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not OpenAI:
            # Fallback to rule-based if no API key
            return 0.5

        client = OpenAI(api_key=api_key)

        # Call GPT-5 nano with minimal reasoning for fast scoring
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=50,  # Higher limit to account for reasoning tokens
            reasoning_effort="minimal"  # Use minimal reasoning for speed
        )

        score_text = response.choices[0].message.content.strip()
        print(f"[AI Scoring] Query: '{query}' | Sender: '{sender}' | Subject: '{subject[:50]}' | GPT-5 response: '{score_text}'")

        # Handle empty or invalid responses
        if not score_text:
            print(f"[AI Scoring] Empty response, using fallback score 0.5")
            return 0.5

        # Try to extract a number from the response
        import re
        number_match = re.search(r'0?\.\d+|[01]\.?\d*', score_text)
        if number_match:
            score = float(number_match.group())
        else:
            print(f"[AI Scoring] Could not parse number from '{score_text}', using fallback")
            return 0.5

        # Clamp to valid range
        score = max(0.0, min(1.0, score))

        # Cache the result
        AI_SCORE_CACHE[cache_key] = {
            'score': score,
            'timestamp': time.time()
        }

        return score

    except Exception as e:
        print(f"[AI Scoring] Error scoring email: {e}")
        # Fallback to 0.5 if scoring fails
        return 0.5


@app.post("/instant-search")
async def instant_search(request: dict):
    """Instant search with filters for date ranges, senders, and attachments.
    Uses AI-powered relevance scoring (GPT-5 nano) for accurate confidence scores.

    Expected request body:
    {
        "query": "search text",
        "filters": {
            "dateRange": "7d|30d|90d|custom",
            "senders": ["email@domain.com"],
            "hasAttachment": true/false,
            "startDate": "2024-01-01" (optional),
            "endDate": "2024-12-31" (optional)
        },
        "limit": 20
    }
    """
    try:
        print(f"[InstantSearch] Request: {request}")
        sb = _get_supabase()

        query = request.get("query", "").strip()
        filters = request.get("filters", {})
        limit = request.get("limit", 20)

        # Parse query for special patterns
        query_lower = query.lower()
        sender_filter = None
        search_text = query

        # Check for "from:" or "emails from" patterns
        if "from:" in query_lower or "emails from" in query_lower:
            import re

            # First try to match @domain pattern (e.g., "emails from @nytimes.com")
            match = re.search(r'(?:from:|emails from)\s*@([a-zA-Z0-9\-\.]+)', query_lower)
            if match:
                sender_filter = match.group(1)
                print(f"[InstantSearch] Detected sender filter (domain): {sender_filter}")
                # Remove the from pattern from search text
                search_text = re.sub(r'(?:from:|emails from)\s*@[a-zA-Z0-9\-\.]+', '', query, flags=re.IGNORECASE).strip()
            else:
                # Try to match sender name pattern (e.g., "emails from new york times")
                match = re.search(r'(?:from:|emails from)\s+(.+?)(?:\s+(?:about|with|containing|in|on)\s+|$)', query_lower)
                if match:
                    sender_filter = match.group(1).strip()
                    print(f"[InstantSearch] Detected sender filter (name): {sender_filter}")
                    # Remove the from pattern from search text
                    search_text = re.sub(r'(?:from:|emails from)\s+[^(about|with|containing|in|on)]+', '', query, flags=re.IGNORECASE).strip()

        # Build the search
        if sender_filter:
            # Use direct database query for sender filtering
            query_builder = sb.table("emails").select("*").ilike("sender", f"%{sender_filter}%")

            # Apply date filter at database level if present
            date_range = filters.get("dateRange")
            if date_range:
                from datetime import datetime, timedelta, timezone
                now = datetime.now(timezone.utc)

                if date_range == "7d":
                    cutoff = now - timedelta(days=7)
                elif date_range == "30d":
                    cutoff = now - timedelta(days=30)
                elif date_range == "90d":
                    cutoff = now - timedelta(days=90)
                else:
                    cutoff = None

                if cutoff:
                    cutoff_iso = cutoff.isoformat()
                    query_builder = query_builder.gte("created_at", cutoff_iso)
                    print(f"[InstantSearch] Applying date filter: created_at >= {cutoff_iso}")

            results = query_builder.order("created_at", desc=True).limit(100).execute()
            search_results = getattr(results, "data", []) or []
            print(f"[InstantSearch] Found {len(search_results)} emails from {sender_filter}")
        elif search_text:
            # Use semantic search for general query
            model = _get_embedding_model()
            query_embedding = model.encode(search_text, normalize_embeddings=True)

            results = sb.rpc('match_emails', {
                'query_embedding': query_embedding.tolist(),
                'match_threshold': 0.3,
                'match_count': 100  # Get more results for filtering
            }).execute()

            search_results = getattr(results, "data", []) or []
        else:
            # No query, just get recent emails
            results = sb.table("emails").select("*").order("created_at", desc=True).limit(100).execute()
            search_results = getattr(results, "data", []) or []

        # Apply filters
        filtered_results = search_results

        # Date range filter
        date_range = filters.get("dateRange")
        if date_range:
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)

            if date_range == "7d":
                cutoff = now - timedelta(days=7)
            elif date_range == "30d":
                cutoff = now - timedelta(days=30)
            elif date_range == "90d":
                cutoff = now - timedelta(days=90)
            elif date_range == "custom":
                start_date = filters.get("startDate")
                end_date = filters.get("endDate")
                # Custom date filtering logic here
                cutoff = None
            else:
                cutoff = None

            if cutoff:
                filtered_results = [
                    email for email in filtered_results
                    if email.get("created_at") and datetime.fromisoformat(email["created_at"].replace("Z", "+00:00")) >= cutoff
                ]

        # Sender filter
        senders = filters.get("senders", [])
        if senders:
            filtered_results = [
                email for email in filtered_results
                if any(sender.lower() in email.get("sender", "").lower() for sender in senders)
            ]

        # Attachment filter
        has_attachment = filters.get("hasAttachment")
        if has_attachment is not None:
            filtered_results = [
                email for email in filtered_results
                if email.get("has_attachment", False) == has_attachment
            ]

        # Limit results
        filtered_results = filtered_results[:limit]

        # Use AI to re-score all results for accurate relevance (hybrid approach)
        if query and filtered_results:
            print(f"[InstantSearch] Re-scoring {len(filtered_results)} results with AI...")
            for result in filtered_results:
                ai_score = _score_email_relevance_with_ai(query, result)
                result['similarity'] = ai_score

        print(f"[InstantSearch] Returning {len(filtered_results)} results")

        # Debug: Log similarity scores
        if filtered_results:
            print(f"[InstantSearch] Sample results with AI similarity scores:")
            for i, result in enumerate(filtered_results[:3]):
                similarity = result.get('similarity', 'N/A')
                subject = result.get('subject', 'No subject')[:50]
                sender = result.get('sender', 'Unknown')[:30]
                print(f"[InstantSearch]   {i+1}. '{subject}' from {sender} - AI similarity: {similarity}")

        # Sort by AI similarity score (highest first)
        if query and filtered_results:
            filtered_results = sorted(filtered_results, key=lambda x: x.get('similarity', 0), reverse=True)

        return {
            "results": filtered_results,
            "total": len(filtered_results)
        }

    except Exception as e:
        print(f"[ERROR] Exception in /instant-search: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
