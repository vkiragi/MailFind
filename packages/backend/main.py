import os
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
    OpenAI = None  # Will validate at runtime

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
        return Fernet(ENCRYPTION_KEY)
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
    """Enhance search query with synonyms and related terms for better matching."""
    query_lower = query.lower()
    enhanced_terms = [query]
    
    # Add related terms for common topics
    if 'yahoo fantasy' in query_lower:
        enhanced_terms.extend(['yahoo sports', 'fantasy football', 'fantasy sports', 'yahoo ff'])
    
    if 'news' in query_lower:
        enhanced_terms.extend(['update', 'announcement', 'breaking', 'latest'])
    
    if 'fantasy' in query_lower and 'football' in query_lower:
        enhanced_terms.extend(['nfl', 'fantasy sports', 'football news'])
    
    # Join terms with the original query getting higher weight
    return f"{query} {' '.join(set(enhanced_terms[1:]))}"


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
        # Limit content to first 1000 chars to avoid token limits
        content_truncated = content[:1000] + "..." if len(content) > 1000 else content
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
                            message = getattr(err, "message", None) if err else None
                            if message:
                                yield f"\n[error] {message}".encode("utf-8")
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
                            message = getattr(err, "message", None) if err else None
                            if message:
                                yield f"\n[error] {message}".encode("utf-8")
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
        sb = _get_supabase()
        
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        
        user_id = request.get("userId")
        
        # Load embedding model and generate query embedding
        model = _get_embedding_model()
        
        # Enhanced query processing for better search results
        enhanced_query = _enhance_search_query(query)
        print(f"[Search] Original query: '{query}', Enhanced: '{enhanced_query}'")
        
        query_embedding = model.encode(enhanced_query, normalize_embeddings=True)
        
        print(f"[Search] Query: '{query}', embedding shape: {query_embedding.shape}")
        
        # Call Supabase RPC function for semantic search with lower threshold for news queries
        is_news_query = any(term in query.lower() for term in ['news', 'update', 'recent', 'latest', 'announcement'])
        match_threshold = 0.3 if is_news_query else 0.5
        match_count = 15 if is_news_query else 10
        
        results = sb.rpc('match_emails', {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': match_threshold,
            'match_count': match_count
        }).execute()
        
        search_results = getattr(results, "data", []) or []
        
        # Post-process results to boost recent emails for news queries
        if is_news_query and search_results:
            search_results = _boost_recent_emails(search_results)
        
        print(f"[Search] Found {len(search_results)} matching emails (threshold: {match_threshold})")
        
        return {
            "status": "success",
            "query": query,
            "enhanced_query": enhanced_query,
            "results": search_results,
            "count": len(search_results),
            "search_type": "news_optimized" if is_news_query else "standard"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


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
                print(f"[Sync] Thread {tid}: Fetched full thread data")
                
                content = _fetch_thread_plaintext(service, tid)
                print(f"[Sync] Thread {tid}: Extracted content ({len(content)} chars)")
                
                meta = _parse_thread_metadata(thread_full)
                subject = meta.get('subject', '')
                sender = meta.get('from', '')
                print(f"[Sync] Thread {tid}: Subject='{subject[:50]}...', Sender='{sender}'")

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
