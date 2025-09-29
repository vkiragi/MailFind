import os
import time
from fastapi import FastAPI, HTTPException, Form
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

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

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

    creds = Credentials(
        token=stored.get("token"),
        refresh_token=stored.get("refresh_token"),
        token_uri=stored.get("token_uri"),
        client_id=stored.get("client_id"),
        client_secret=stored.get("client_secret"),
        scopes=stored.get("scopes", SCOPES),
    )

    # Refresh if needed and persist new access token
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            updated = dict(stored)
            updated["token"] = creds.token
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
    if sender_domain:
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
    """Redirect to Google OAuth"""
    try:
        flow = get_flow()
        
        # Generate state to prevent CSRF
        state = secrets.token_urlsafe(32)
        oauth_states[state] = True
        
        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent',  # Force consent screen to ensure refresh token
            state=state
        )
        
        return RedirectResponse(url=authorization_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.get("/callback")
def callback(code: str, state: str):
    """Handle OAuth callback, store encrypted tokens in Supabase, and return user id."""
    try:
        # Verify state to prevent CSRF
        if state not in oauth_states:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        # Remove used state
        del oauth_states[state]

        sb = _get_supabase()

        flow = get_flow()
        flow.fetch_token(code=code)

        # Fetch Google user info
        access_token = flow.credentials.token
        userinfo = {}
        try:
            uresp = requests.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            if uresp.ok:
                userinfo = uresp.json() or {}
        except Exception:
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
        }
        
        print(f"[OAuth] Storing credentials: token={bool(creds_payload['token'])}, refresh_token={bool(creds_payload['refresh_token'])}, client_id={bool(creds_payload['client_id'])}, client_secret={bool(creds_payload['client_secret'])}")
        encrypted = _encrypt_dict(creds_payload)

        # Upsert into Supabase users table
        now_iso = datetime.now(timezone.utc).isoformat()
        existing = sb.table("users").select("id").eq("google_user_id", google_user_id).limit(1).execute()  # type: ignore[attr-defined]
        if (getattr(existing, "data", []) or []):
            # Update
            sb.table("users").update({
                "email": email,
                "encrypted_credentials": encrypted,
                "updated_at": now_iso,
            }).eq("google_user_id", google_user_id).execute()  # type: ignore[attr-defined]
        else:
            # Insert
            sb.table("users").insert({
                "google_user_id": google_user_id,
                "email": email,
                "encrypted_credentials": encrypted,
                "created_at": now_iso,
                "updated_at": now_iso,
            }).execute()  # type: ignore[attr-defined]

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


@app.post("/chat")
async def chat_with_emails(request: dict):
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
        # BUT exclude queries that mention specific entities or companies
        message_lower = message.lower()
        has_specific_entities = any(entity in message_lower for entity in [
            'anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia',
            'nyt', 'new york times', 'ny times', 'times', 'yahoo', 'facebook', 'twitter', 'linkedin'
        ])

        is_generic_temporal_query = (
            not has_specific_entities and  # Don't treat as generic if specific entities mentioned
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
        
        # Perform semantic search to get relevant emails
        model = _get_embedding_model()
        query_embedding = model.encode(enhanced_query, normalize_embeddings=True)
        
        # Use more lenient threshold for chat to get more context
        # For temporal queries and news-related queries, use very low threshold
        message_lower = message.lower()
        is_news_query = any(term in message_lower for term in ['news', 'nyt', 'new york times', 'times', 'newspaper', 'breaking'])

        if is_generic_temporal_query or is_news_query:
            threshold = 0.15  # Very low threshold to catch all potentially relevant emails
            count = 30        # Higher count for broader results
        else:
            threshold = 0.3   # More lenient threshold for chat
            count = 20        # More results for better context
            
        results = sb.rpc('match_emails', {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': threshold,
            'match_count': count
        }).execute()
        
        search_results = getattr(results, "data", []) or []
        print(f"[Chat] Found {len(search_results)} relevant emails (threshold: {threshold})")
        
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
        
        # For generic temporal queries, sort by recency instead of similarity
        if is_generic_temporal_query and search_results:
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
            print(f"[Chat] Sorted {len(search_results)} results by recency for temporal query")
        
        # Apply minimal filtering for chat to preserve context - only filter if query is very specific
        entity_indicators = ['anthropic', 'openai', 'google', 'microsoft', 'apple', 'amazon', 'meta', 'tesla', 'nvidia']
        has_specific_entity = any(entity in message_lower for entity in entity_indicators)

        # Only apply aggressive filtering for very specific entity queries, not for general news/time queries
        if search_results and has_specific_entity and not is_news_query:
            original_count = len(search_results)
            search_results = _filter_irrelevant_results(message, search_results)
            filtered_count = len(search_results)
            if original_count != filtered_count:
                print(f"[Chat] Filtered out {original_count - filtered_count} irrelevant results")
        else:
            print(f"[Chat] Skipping aggressive filtering for temporal/news query")
        

        # Prepare context for the LLM
        email_context = _prepare_email_context(search_results, message)
        print(f"[Chat] Prepared email context with {len(search_results)} results")
        print(f"[Chat] Context length: {len(email_context)} characters")
        
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
                "\n\nIMPORTANT: The user asked about 'important' emails. Focus ONLY on emails that are truly significant, such as: "
                "1) Personal emails from real people (not automated/marketing), "
                "2) Work-related communications from colleagues/clients, "
                "3) Financial alerts (bills, payments, receipts), "
                "4) Account notifications (password resets, security alerts), "
                "5) Travel confirmations and tickets. "
                "EXCLUDE: Newsletters, promotional emails, marketing content, social media notifications, "
                "automated digests, and subscription content. If all the emails provided are newsletters/promotions, "
                "say 'I found some emails from today, but they appear to be newsletters and promotional content rather than important personal or work emails.'"
            )

        system_prompt += (
            "\n\nIMPORTANT: When mentioning emails, always include the Gmail URL as a clickable link using markdown format: "
            "[Email Subject](Gmail URL). This allows users to click directly to open the email in Gmail."
        )
        
        # Temporary: Use simple non-streaming response for debugging
        try:
            print(f"[Chat] Making OpenAI API call...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Question: {message}\n\nRelevant Email Context:\n{email_context}"}
                ],
                max_tokens=500,
                temperature=0.7
            )

            content = response.choices[0].message.content
            print(f"[Chat] Got response: {content[:100]}...")

            if content:
                return {"response": content}
            else:
                return {"response": "No response generated."}

        except Exception as oe:
            print(f"[Chat] OpenAI error: {str(oe)}")
            return {"error": f"Chat error: {str(oe)}"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Exception in /chat: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


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
        
        # Truncate content to avoid token limits
        content_preview = content[:500] + "..." if len(content) > 500 else content
        
        # Generate Gmail URL
        gmail_url = _generate_gmail_url(thread_id) if thread_id else ""
        
        context_parts.append(f"Email {i+1}:")
        context_parts.append(f"  From: {sender}")
        context_parts.append(f"  Subject: {subject}")
        context_parts.append(f"  Date: {created_at}")
        context_parts.append(f"  Gmail URL: {gmail_url}")
        context_parts.append(f"  Relevance: {similarity:.2f}")
        context_parts.append(f"  Content: {content_preview}")
        context_parts.append("")
    
    return "\n".join(context_parts)


@app.post("/sync-inbox")
async def sync_inbox(request: dict):
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

        # List recent threads (limit to 50 for popup use)
        print("[Sync] Fetching threads list...")
        if gmail_q:
            threads_resp = service.users().threads().list(userId="me", maxResults=50, q=gmail_q).execute()
        else:
            threads_resp = service.users().threads().list(userId="me", maxResults=50).execute()
        threads = threads_resp.get('threads', []) or []
        print(f"[Sync] Found {len(threads)} threads")
        
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
                print(f"[Sync] Thread {tid}: Subject='{subject[:50]}...', Sender='{sender}'")

                # Classify email into categories using LLM
                print(f"[Sync] Thread {tid}: Classifying email categories")
                categories = _classify_email_categories(subject, content, sender)
                
                # Create enhanced text for embedding that includes keywords and context
                text_for_embedding = _create_enhanced_embedding_text(subject, content, sender)
                print(f"[Sync] Thread {tid}: Generating embedding for {len(text_for_embedding)} chars")
                emb = model.encode(text_for_embedding, normalize_embeddings=True).tolist()  # 384 dims
                print(f"[Sync] Thread {tid}: Generated embedding with {len(emb)} dimensions")

                # Upsert into Supabase emails table
                payload = {
                    "google_user_id": user_id,
                    "thread_id": tid,
                    "subject": subject,
                    "sender": sender,
                    "content": content,
                    "embedding": emb,
                    "categories": categories,
                }
                print(f"[Sync] Thread {tid}: Upserting to Supabase...")
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
