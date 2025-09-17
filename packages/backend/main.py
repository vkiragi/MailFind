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
from cryptography.fernet import Fernet, InvalidToken  # type: ignore
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
oauth_states = {}\

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
    try:
        raw = f.decrypt(token_str.encode("utf-8"))
    except InvalidToken:
        # Likely ENCRYPTION_KEY changed or stored credentials are corrupted/expired
        raise HTTPException(status_code=401, detail="Stored credentials invalid. Please login again.")
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
    # Try to derive a consistent subject, sender, and timestamp from the last message
    messages = thread.get('messages', [])
    if not messages:
        return {"subject": "", "from": "", "timestamp": None}
    last = messages[-1]
    payload = last.get('payload', {})
    headers = _extract_headers(payload)
    subject = headers.get('subject', '')
    sender = headers.get('from', '')
    
    # Extract timestamp from the message
    timestamp = None
    try:
        # Gmail API provides internalDate as milliseconds since epoch
        internal_date = last.get('internalDate')
        if internal_date:
            from datetime import datetime, timezone
            timestamp = datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc).isoformat()
        else:
            # Fallback: try to parse Date header
            date_header = headers.get('date', '')
            if date_header:
                from email.utils import parsedate_to_datetime
                try:
                    parsed_date = parsedate_to_datetime(date_header)
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                    timestamp = parsed_date.isoformat()
                except Exception:
                    pass
    except Exception as e:
        print(f"[Metadata] Error parsing timestamp: {e}")
    
    return {"subject": subject, "from": sender, "timestamp": timestamp}

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
    """Logout clears pending oauth states. User data is preserved."""
    try:
        global oauth_states
        oauth_states.clear()
        
        users_count = 0
        try:
            sb = _get_supabase()
            res = sb.table("users").select("id", count="exact").execute()  # type: ignore[attr-defined]
            users_count = getattr(res, "count", 0) or 0
        except Exception:
            users_count = 0
        
        print("[Logout] Cleared OAuth states - user credentials preserved")
        return {
            "message": "Logout completed - cleared active sessions (user data preserved)",
            "authenticated_users": users_count,
            "active_states": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout error: {str(e)}")

@app.post("/clear-all-users")
def clear_all_users():
    """Development endpoint: Clear all stored user credentials from database.
    WARNING: This is destructive and removes all user data."""
    try:
        sb = _get_supabase()
        # Delete all users (keeping the placeholder UUID if it exists)
        sb.table("users").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("[Clear Users] Cleared all stored user credentials from Supabase")
        
        users_count = 0
        try:
            res = sb.table("users").select("id", count="exact").execute()  # type: ignore[attr-defined]
            users_count = getattr(res, "count", 0) or 0
        except Exception:
            users_count = 0
            
        return {
            "message": "All user credentials cleared from database",
            "authenticated_users": users_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear users error: {str(e)}")

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


def _build_gmail_query_string(parsed_entities: dict) -> str:
    """Build clean, simple Gmail API query string from parsed entities."""
    query_parts = []
    
    # Add sender filter - clean format
    if parsed_entities.get("sender"):
        sender = parsed_entities["sender"]
        # Clean sender format - remove extra spaces and variations
        clean_sender = sender.strip().lower()
        query_parts.append(f"from:{clean_sender}")
    
    # Add recipient filter - clean format  
    if parsed_entities.get("recipient"):
        recipient = parsed_entities["recipient"].strip().lower()
        query_parts.append(f"to:{recipient}")
    
    # Add date range filter - use Gmail's preferred format
    if parsed_entities.get("date_range"):
        date_range = parsed_entities["date_range"]
        start_date = date_range.get("start")
        end_date = date_range.get("end")
        
        if start_date:
            # Convert YYYY-MM-DD to YYYY/MM/DD for Gmail
            gmail_start = start_date.replace("-", "/")
            query_parts.append(f"after:{gmail_start}")
        if end_date:
            gmail_end = end_date.replace("-", "/")
            query_parts.append(f"before:{gmail_end}")
    
    # Add status filter - clean format
    if parsed_entities.get("status"):
        status = parsed_entities["status"].lower()
        if status == "unread":
            query_parts.append("is:unread")
        elif status == "read":
            query_parts.append("is:read")
        elif status == "starred":
            query_parts.append("is:starred")
    
    # Add priority filter
    if parsed_entities.get("priority"):
        priority = parsed_entities["priority"].lower()
        if priority in ["high", "urgent", "important"]:
            query_parts.append("is:important")
    
    # Add search terms - simple format without redundant OR logic
    search_terms = parsed_entities.get("search_terms", [])
    if search_terms:
        # Simple space-separated terms in quotes
        for term in search_terms:
            if term.strip():  # Only add non-empty terms
                query_parts.append(f'"{term.strip()}"')
    
    # Default to inbox if no filters (not "all" which can be too broad)
    if not query_parts:
        query_parts.append("in:inbox")
    
    # Build final clean query
    gmail_query = " ".join(query_parts)
    print(f"[Gmail Query] Built clean query: '{gmail_query}'")
    return gmail_query


async def _fetch_emails_for_summarization(parsed_entities: dict, user_id: str = None) -> list:
    """Fetch email content from Gmail API for summarization."""
    try:
        gmail_service = _build_gmail_service(user_id)
        if not gmail_service:
            raise HTTPException(status_code=401, detail="Gmail service not available")
        
        # Build Gmail query string
        gmail_query = _build_gmail_query_string(parsed_entities)
        print(f"[Summarize] Gmail query: {gmail_query}")
        
        # List messages matching the query
        messages_result = gmail_service.users().messages().list(
            userId="me",
            q=gmail_query,
            maxResults=50  # Limit for summarization performance
        ).execute()
        
        messages = messages_result.get("messages", [])
        if not messages:
            return []
        
        print(f"[Summarize] Found {len(messages)} messages to summarize")
        
        # Fetch full content for each message
        email_contents = []
        for msg in messages[:20]:  # Limit to 20 emails for performance
            try:
                message = gmail_service.users().messages().get(
                    userId="me", 
                    id=msg["id"],
                    format="full"
                ).execute()
                
                # Extract email metadata and content
                headers = _extract_headers(message.get("payload", {}))
                subject = headers.get("subject", "No Subject")
                sender = headers.get("from", "Unknown Sender")
                date = headers.get("date", "Unknown Date")
                
                # Extract body content
                content = _extract_email_content(message.get("payload", {}))
                
                if content.strip():  # Only include emails with content
                    email_contents.append({
                        "subject": subject,
                        "sender": sender,
                        "date": date,
                        "content": content[:2000]  # Limit content length for API efficiency
                    })
                    
            except Exception as e:
                print(f"[Summarize] Error fetching message {msg['id']}: {e}")
                continue
        
        print(f"[Summarize] Successfully fetched {len(email_contents)} emails for summarization")
        return email_contents
        
    except Exception as e:
        print(f"[Summarize] Error fetching emails: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch emails for summarization: {str(e)}")


async def _generate_email_summary(email_contents: list, parsed_entities: dict) -> str:
    """Generate a summary of multiple emails using GPT-5-nano."""
    try:
        # Ensure OpenAI client is available
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API not available for summarization")
        
        client = OpenAI(api_key=api_key)
        
        # Prepare combined email text
        combined_text = ""
        for i, email in enumerate(email_contents, 1):
            combined_text += f"\n--- EMAIL {i} ---\n"
            combined_text += f"From: {email['sender']}\n"
            combined_text += f"Subject: {email['subject']}\n"
            combined_text += f"Date: {email['date']}\n"
            combined_text += f"Content: {email['content']}\n"
        
        # Create specialized summarization prompt
        system_prompt = """You are a world-class executive assistant with exceptional email management skills. 

Your task is to review multiple emails and provide a concise, actionable summary that highlights:
1. Most important updates and announcements
2. Key decisions that were made
3. Action items or tasks that require attention
4. Deadlines or time-sensitive information
5. Important communications from key people

IGNORE:
- Marketing emails and promotional content
- Automated notifications with no actionable content
- Spam or irrelevant messages

FORMAT your response as:
## Summary
[Brief overview of the main themes and topics]

## Key Updates
- [Important update 1]
- [Important update 2]

## Action Items
- [Action item 1 with deadline if mentioned]
- [Action item 2 with deadline if mentioned]

## Important Communications
- [Key message from important sender]

Keep the summary concise but comprehensive. Focus on what the user needs to know and act upon."""
        
        user_prompt = f"Please summarize the following {len(email_contents)} emails:\n\n{combined_text}"
        
        print(f"[Summarize] Generating summary for {len(email_contents)} emails using GPT-5-nano")
        
        # Use GPT-5-nano for summarization
        try:
            with client.responses.stream(
                model="gpt-5-nano",
                input=f"System: {system_prompt}\n\nUser: {user_prompt}",
            ) as stream:
                summary_text = ""
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            summary_text += delta
                
                print(f"[Summarize] Generated summary: {len(summary_text)} characters")
                return summary_text.strip()
                
        except Exception as api_error:
            print(f"[Summarize] GPT-5-nano API error: {api_error}")
            # Fallback to GPT-4o-mini
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                summary_text = response.choices[0].message.content
                print(f"[Summarize] Fallback summary generated: {len(summary_text)} characters")
                return summary_text.strip()
                
            except Exception as fallback_error:
                print(f"[Summarize] Fallback summarization failed: {fallback_error}")
                raise HTTPException(status_code=500, detail="Failed to generate email summary")
        
    except Exception as e:
        print(f"[Summarize] Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


async def _generate_conversational_answer(search_results: list, parsed_entities: dict, original_query: str) -> str:
    """Generate a conversational answer based on search results, like a chatbot."""
    try:
        # Ensure OpenAI client is available
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            # Fallback to simple response
            if not search_results:
                return "I didn't find any emails matching your query."
            else:
                return f"I found {len(search_results)} relevant emails for you."
        
        client = OpenAI(api_key=api_key)
        
        # Prepare context about the search results
        if not search_results:
            context = "No emails were found matching the user's query."
        else:
            context = f"Found {len(search_results)} relevant emails:\n\n"
            for i, result in enumerate(search_results[:5], 1):  # Limit to top 5 for context
                context += f"{i}. From: {result.get('sender', 'Unknown')}\n"
                context += f"   Subject: {result.get('subject', 'No subject')}\n"
                if result.get('created_at'):
                    context += f"   Date: {result.get('created_at')}\n"
                context += "\n"
        
        # Create conversational response prompt
        system_prompt = """You are a helpful AI email assistant. Based on the user's query and the search results, provide a natural, conversational response as if you're a knowledgeable assistant.

Guidelines:
- Be conversational and friendly, like ChatGPT or Gemini
- If emails were found, mention the key details (sender, subject, count)
- If no emails were found, suggest why or offer alternatives
- Keep responses concise but informative
- Use natural language, not technical jargon
- Focus on what's most relevant to the user's question

Examples of good responses:
- "I found 3 recent emails from MongoDB! The latest is about a webinar on queries and aggregation, plus a consultation offer."
- "No MongoDB emails in the last 30 days, but I did find some from earlier this year about database optimization."
- "Yes! You got 2 emails from your boss this week - one about the quarterly review and another about the team meeting."
- "I don't see any recent updates from GitHub, but I found some older notifications about repository activity."
"""
        
        user_prompt = f"""User asked: "{original_query}"

Search results context:
{context}

Please provide a helpful, conversational response about what was found."""
        
        print(f"[Answer] Generating conversational response for: '{original_query}'")
        
        # Use GPT-5-nano for conversational response
        try:
            with client.responses.stream(
                model="gpt-5-nano",
                input=f"System: {system_prompt}\n\nUser: {user_prompt}",
            ) as stream:
                response_text = ""
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            response_text += delta
                
                print(f"[Answer] Generated conversational response: {len(response_text)} characters")
                return response_text.strip()
                
        except Exception as api_error:
            print(f"[Answer] GPT-5-nano API error: {api_error}")
            # Fallback to GPT-4o-mini
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content
                print(f"[Answer] Fallback conversational response: {len(response_text)} characters")
                return response_text.strip()
                
            except Exception as fallback_error:
                print(f"[Answer] Fallback also failed: {fallback_error}")
                # Simple fallback
                if not search_results:
                    return f"I didn't find any emails about {parsed_entities.get('search_terms', ['your query'])[0] if parsed_entities.get('search_terms') else 'that topic'}."
                else:
                    return f"I found {len(search_results)} emails that might interest you! The most recent one is from {search_results[0].get('sender', 'an unknown sender')}."
        
    except Exception as e:
        print(f"[Answer] Error generating conversational answer: {e}")
        return "I had trouble processing your request, but I'm here to help with your emails!"


@app.post("/search")
async def search_emails(request: dict):
    """Advanced semantic search with AI query parsing and sophisticated multi-step filtering.

    Expected request body: { "query": "search terms", "userId": "..." (optional) }
    
    Process:
    1. Parse complex natural language query with GPT-5-nano to extract entities
    2. Build dynamic, multi-step Supabase query with structured filters  
    3. Perform semantic vector search on filtered results
    4. Apply additional post-processing filters and ranking
    5. Return highly accurate, contextually relevant results
    """
    try:
        sb = _get_supabase()
        
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        
        user_id = request.get("userId")
        
        print(f"[Search] Processing natural language query: '{query}'")
        
        # Step 1: Parse query with GPT-5-nano to extract comprehensive entities
        parsed_entities = await _parse_query_with_ai(query)
        print(f"[Search] Raw extracted entities: {parsed_entities}")
        
        # Step 1.5: Validate and enhance parsed entities
        parsed_entities = _validate_and_enhance_parsed_entities(parsed_entities)
        print(f"[Search] Enhanced entities: {parsed_entities}")
        
        # Step 1.7: Check if this is a summarization request
        action = parsed_entities.get("action", "search")
        if action == "summarize":
            print("[Search] Detected summarization request - switching to summarization workflow")
            
            # Try to fetch emails from Gmail API based on parsed filters
            try:
                email_contents = await _fetch_emails_for_summarization(parsed_entities, user_id)
                print(f"[Search] Gmail API returned {len(email_contents)} emails")
                
                # If Gmail API succeeded but found 0 emails, still try Supabase fallback
                if not email_contents:
                    print("[Search] Gmail API found 0 emails, trying Supabase fallback")
                    raise Exception("Gmail API found no emails, trying fallback")
                    
            except Exception as gmail_error:
                print(f"[Search] Gmail API failed for summarization: {gmail_error}")
                # Fallback to Supabase data for summarization
                print("[Search] Falling back to Supabase data for summarization")
                
                # REUSE THE EXACT SAME LOGIC AS REGULAR SEARCH
                # Step 2: Build sophisticated dynamic Supabase query with multiple filters
                base_query = sb.table("emails").select("*")
                applied_filters = []
                
                # Apply sender filter with flexible matching (same as regular search)
                if parsed_entities.get("sender"):
                    sender = parsed_entities["sender"]
                    sender_variations = parsed_entities.get("sender_variations", [sender])
                    
                    # Use the most specific sender variation for the primary filter
                    primary_sender = sender_variations[0] if sender_variations else sender
                    base_query = base_query.filter("sender", "ilike", f"%{primary_sender}%")
                    applied_filters.append(f"sender contains '{primary_sender}'")
                    print(f"[Summarize] Applied sender filter: {primary_sender}")
                
                # Apply date range filter (same as regular search)
                if parsed_entities.get("date_range"):
                    date_range = parsed_entities["date_range"]
                    if date_range.get("start"):
                        base_query = base_query.filter("created_at", "gte", date_range["start"])
                        applied_filters.append(f"after {date_range['start']}")
                    if date_range.get("end"):
                        base_query = base_query.filter("created_at", "lte", date_range["end"])
                        applied_filters.append(f"before {date_range['end']}")
                    print(f"[Summarize] Applied date filter: {date_range}")
                
                # Apply recency ordering if specified
                if parsed_entities.get("recency_priority"):
                    base_query = base_query.order('created_at', desc=True)
                    print("[Summarize] Applied recency ordering")
                
                # Execute the structured query (same as regular search)
                print(f"[Summarize] Executing Supabase query with filters: {applied_filters}")
                
                # If no filters were applied, get recent emails broadly
                if not applied_filters:
                    print("[Summarize] No specific filters, getting recent emails broadly")
                    # Don't order by created_at since many emails have null timestamps
                    # Just get emails without any ordering
                
                filtered_result = base_query.limit(20).execute()
                filtered_emails = filtered_result.data or []
                print(f"[Summarize] Found {len(filtered_emails)} emails with structured filters")
                
                # Convert Supabase format to summarization format
                email_contents = []
                for email in filtered_emails:
                    email_contents.append({
                        "subject": email.get("subject", "No Subject"),
                        "sender": email.get("sender", "Unknown Sender"), 
                        "date": email.get("created_at", "Unknown Date"),
                        "content": email.get("content", "")[:2000]
                    })
                
                print(f"[Summarize] Prepared {len(email_contents)} emails for summarization")
            
            if not email_contents:
                return {
                    "status": "success",
                    "action": "summarize",
                    "query": query,
                    "parsed_entities": parsed_entities,
                    "summary": "No emails found matching your criteria.",
                    "emails_found": 0
                }
            
            # Generate summary using GPT-5-nano
            summary = await _generate_email_summary(email_contents, parsed_entities)
            
            return {
                "status": "success",
                "action": "summarize",
                "query": query,
                "parsed_entities": parsed_entities,
                "summary": summary,
                "emails_found": len(email_contents),
                "emails_summarized": [
                    {
                        "subject": email["subject"],
                        "sender": email["sender"],
                        "date": email["date"]
                    } for email in email_contents
                ]
            }
        
        # Step 1.8: Check if this is a conversational answer request
        if action == "answer":
            print("[Search] Detected conversational answer request - generating natural language response")
            
            # Perform regular search first to get relevant emails
            # (Use the same logic as regular search but generate conversational response)
            pass  # Will implement the search logic below and then generate conversational response
        
        # Step 2: Build sophisticated dynamic Supabase query with multiple filters (for search/answer actions)
        base_query = sb.table("emails").select("*")
        applied_filters = []
        
        # Apply sender filter with flexible matching
        if parsed_entities.get("sender"):
            sender = parsed_entities["sender"]
            sender_variations = parsed_entities.get("sender_variations", [sender])
            
            # Use the most specific sender variation for the primary filter
            primary_sender = sender_variations[0] if sender_variations else sender
            base_query = base_query.filter("sender", "ilike", f"%{primary_sender}%")
            applied_filters.append(f"sender contains '{primary_sender}'")
            print(f"[Search] Applied sender filter: {primary_sender}")
            
            # Store variations for later use in semantic search boosting
            parsed_entities["_sender_boost_terms"] = sender_variations
        
        # Apply recipient filter if specified  
        if parsed_entities.get("recipient"):
            recipient = parsed_entities["recipient"]
            # Note: This would require a recipient column in the emails table
            # base_query = base_query.filter("recipient", "ilike", f"%{recipient}%")
            applied_filters.append(f"recipient contains '{recipient}'")
            print(f"[Search] Recipient filter noted (requires schema update): {recipient}")
        
        # Apply sophisticated date range filtering
        if parsed_entities.get("date_range"):
            date_range = parsed_entities["date_range"]
            if date_range.get("start"):
                base_query = base_query.filter("created_at", "gte", f"{date_range['start']}T00:00:00Z")
                applied_filters.append(f"date >= {date_range['start']}")
                print(f"[Search] Applied start date filter: {date_range['start']}")
            if date_range.get("end"):
                base_query = base_query.filter("created_at", "lte", f"{date_range['end']}T23:59:59Z")
                applied_filters.append(f"date <= {date_range['end']}")
                print(f"[Search] Applied end date filter: {date_range['end']}")
        
        # Apply file type and attachment filters
        if parsed_entities.get("file_type"):
            file_type = parsed_entities["file_type"].lower()
            # Search for file extensions in content
            base_query = base_query.filter("content", "ilike", f"%.{file_type}%")
            applied_filters.append(f"contains .{file_type} files")
            print(f"[Search] Applied file type filter: {file_type}")
        
        # Apply subject keyword filters
        if parsed_entities.get("subject_keywords"):
            subject_keywords = parsed_entities["subject_keywords"]
            for keyword in subject_keywords:
                base_query = base_query.filter("subject", "ilike", f"%{keyword}%")
                applied_filters.append(f"subject contains '{keyword}'")
            print(f"[Search] Applied subject filters: {subject_keywords}")
        
        # Apply priority filters (search in content for priority indicators)
        if parsed_entities.get("priority"):
            priority = parsed_entities["priority"]
            priority_terms = ["urgent", "important", "high priority", "asap"]
            if priority.lower() in priority_terms:
                base_query = base_query.filter("content", "ilike", f"%{priority}%")
                applied_filters.append(f"priority: {priority}")
                print(f"[Search] Applied priority filter: {priority}")
        
        # Execute the structured query to get filtered emails
        try:
            print(f"[Search] Executing structured query with filters: {applied_filters}")
            
            # Add ordering by created_at for recency queries
            if parsed_entities.get('recency_priority'):
                base_query = base_query.order('created_at', desc=True)
                print(f"[Search] Added recency ordering (created_at DESC)")
            
            structured_results = base_query.limit(200).execute()  # Increased limit for better results
            filtered_emails = getattr(structured_results, "data", []) or []
            print(f"[Search] Structured filtering found {len(filtered_emails)} emails")
        except Exception as e:
            print(f"[Search] Structured query failed, using fallback: {str(e)}")
            # Fallback to basic query if complex filtering fails
            try:
                fallback_query = sb.table("emails").select("*")
                if parsed_entities.get("sender"):
                    fallback_query = fallback_query.filter("sender", "ilike", f"%{parsed_entities['sender']}%")
                structured_results = fallback_query.limit(100).execute()
                filtered_emails = getattr(structured_results, "data", []) or []
                print(f"[Search] Fallback query found {len(filtered_emails)} emails")
            except Exception as fallback_error:
                print(f"[Search] Fallback also failed: {fallback_error}")
                filtered_emails = []
        
        # Step 3: Perform enhanced semantic search 
        search_terms = parsed_entities.get("search_terms", [query])
        if not search_terms:
            search_terms = [query]
        
        # Add email type to search terms if specified
        if parsed_entities.get("email_type"):
            search_terms.append(parsed_entities["email_type"])
        
        print(f"[Search] Performing semantic search for terms: {search_terms}")
        
        # Generate embeddings and perform vector search
        model = _get_embedding_model()
        semantic_results = []
        seen_thread_ids = set()
        
        # Combine original query with extracted search terms for better semantic matching
        all_search_terms = list(set([query] + search_terms))  # Remove duplicates
        
        for term in all_search_terms:
            try:
                embedding = model.encode(term, normalize_embeddings=True)
                
                # Perform vector search - prefer RPC function for better performance
                try:
                    vector_results = sb.rpc('match_emails', {
                        'query_embedding': embedding.tolist(),
                        'match_threshold': 0.4,  # Lowered threshold for more results
                        'match_count': 20  # Increased count
                    }).execute()
                    
                    term_results = getattr(vector_results, "data", []) or []
                    
                    # If we have structured filters, prioritize results that match them
                    if filtered_emails:
                        filtered_thread_ids = {email.get('thread_id') for email in filtered_emails}
                        # Boost similarity score for emails that match structured filters
                        for result in term_results:
                            if result.get('thread_id') in filtered_thread_ids:
                                result['similarity'] = min(1.0, result.get('similarity', 0) + 0.3)  # Boost score
                                result['matched_filters'] = True
                            else:
                                result['matched_filters'] = False
                    
                    # Add results, avoiding duplicates
                    for result in term_results:
                        thread_id = result.get('thread_id')
                        if thread_id not in seen_thread_ids:
                            result['matched_term'] = term
                            result['search_type'] = 'vector_search'
                            semantic_results.append(result)
                            seen_thread_ids.add(thread_id)
                    
                    print(f"[Search] Term '{term}' found {len(term_results)} vector matches")
                    
                except Exception as vector_error:
                    print(f"[Search] Vector search failed for term '{term}': {vector_error}")
                    
                    # Manual similarity calculation fallback for filtered emails
                    if filtered_emails:
                        for email in filtered_emails:
                            thread_id = email.get('thread_id')
                            if thread_id not in seen_thread_ids:
                                # Simple keyword matching as fallback
                                content = (email.get('content', '') + ' ' + email.get('subject', '')).lower()
                                term_lower = term.lower()
                                if term_lower in content:
                                    email['similarity'] = 0.7  # High score for exact matches
                                    email['matched_term'] = term
                                    email['search_type'] = 'keyword_match'
                                    email['matched_filters'] = True
                                    semantic_results.append(email)
                                    seen_thread_ids.add(thread_id)
                        
                        print(f"[Search] Fallback keyword matching for '{term}' on filtered emails")
                    
            except Exception as e:
                print(f"[Search] Error processing search term '{term}': {str(e)}")
                continue
        
        # Step 4: Advanced post-processing and ranking
        print(f"[Search] Post-processing {len(semantic_results)} results")
        
        # Apply additional filters based on parsed entities
        if parsed_entities.get("attachment_required"):
            # Filter for emails likely to have attachments
            attachment_keywords = ['attachment', 'attached', 'please find', 'pdf', 'doc', 'file']
            semantic_results = [
                result for result in semantic_results
                if any(keyword in result.get('content', '').lower() for keyword in attachment_keywords)
            ]
            print(f"[Search] Filtered for attachments: {len(semantic_results)} results remain")
        
        # Calculate enhanced relevance scores for better ranking
        for result in semantic_results:
            relevance_score = _calculate_enhanced_relevance_score(result, parsed_entities, query)
            result['enhanced_relevance'] = relevance_score
        
        # Sort by enhanced relevance score (content relevance prioritized over sender matching)
        semantic_results.sort(key=lambda x: x.get('enhanced_relevance', 0), reverse=True)
        
        # Limit final results
        final_results = semantic_results[:20]  # Increased limit for better user experience
        
        # If no good results found and we have a sender filter, try a broader search
        # Also trigger if we have sender-specific query but no sender matches in top results
        needs_broad_search = False
        if not final_results:
            needs_broad_search = True
        elif final_results:
            max_relevance = max(r.get('enhanced_relevance', 0) for r in final_results)
            # Check if any of the top results match the sender we're looking for
            sender_in_results = any(
                parsed_entities.get('sender', '').lower() in r.get('sender', '').lower() 
                for r in final_results[:3]
            ) if parsed_entities.get('sender') else True
            
            # Trigger broad search if low relevance OR no sender match in top results
            if max_relevance < 100 or not sender_in_results:
                needs_broad_search = True
                print(f"[Search] Triggering broad search: max_relevance={max_relevance:.1f}, sender_in_results={sender_in_results}")
        
        if needs_broad_search:
            if parsed_entities.get('sender'):
                print(f"[Search] Low relevance results, trying broader sender search...")
                try:
                    # Try a broader sender search without other filters
                    broad_query = sb.table("emails").select("*").filter("sender", "ilike", f"%{parsed_entities['sender']}%")
                    
                    # Add recency ordering for recency queries
                    if parsed_entities.get('recency_priority'):
                        broad_query = broad_query.order('created_at', desc=True)
                        print(f"[Search] Added recency ordering to broad search")
                    
                    broad_results = getattr(broad_query.limit(10).execute(), "data", []) or []
                    
                    if broad_results:
                        print(f"[Search] Found {len(broad_results)} emails in broader sender search")
                        # Add these and calculate proper enhanced relevance scores
                        for result in broad_results:
                            if result.get('thread_id') not in seen_thread_ids:
                                result['similarity'] = 0.6  # Moderate similarity
                                result['search_type'] = 'broad_sender_search'
                                result['matched_term'] = parsed_entities['sender']
                                
                                # Calculate proper enhanced relevance score for broad search results
                                relevance_score = _calculate_enhanced_relevance_score(result, parsed_entities, query)
                                result['enhanced_relevance'] = relevance_score
                                
                                final_results.append(result)
                                
                        # Re-sort and limit
                        final_results.sort(key=lambda x: x.get('enhanced_relevance', 0), reverse=True)
                        final_results = final_results[:20]
                        
                except Exception as e:
                    print(f"[Search] Broader search failed: {str(e)}")
        
        print(f"[Search] Final ranked results: {len(final_results)} emails")
        
        # Step 4.5: Apply result count limiting if specified
        original_count = len(final_results)
        requested_count = parsed_entities.get("result_count")
        
        if requested_count and isinstance(requested_count, int) and requested_count > 0:
            if len(final_results) > requested_count:
                final_results = final_results[:requested_count]
                print(f"[Search] Limited results from {original_count} to {requested_count} as requested")
            else:
                print(f"[Search] User requested {requested_count} results, but only found {len(final_results)}")
        
        # Step 5.5: Generate conversational response if this is an "answer" action
        if action == "answer":
            print("[Search] Generating conversational answer based on search results")
            conversational_answer = await _generate_conversational_answer(final_results, parsed_entities, query)
            
            return {
                "status": "success",
                "action": "answer",
                "query": query,
                "answer": conversational_answer,
                "results_found": len(final_results),
                "parsed_entities": parsed_entities,
                "search_metadata": {
                    "ai_parsing": "gpt-5-nano",
                    "conversational_response": True,
                    "emails_analyzed": len(final_results)
                }
            }
        
        # Step 5: Prepare enhanced response with detailed metadata (for regular search)
        response_data = {
            "status": "success",
            "query": query,
            "parsed_entities": parsed_entities,
            "applied_filters": applied_filters,
            "structured_filter_count": len(filtered_emails) if filtered_emails else 0,
            "search_terms": search_terms,
            "total_candidates": len(semantic_results),
            "results": final_results,
            "count": len(final_results),
            "original_count": original_count,  # Show how many results were found before limiting
            "requested_count": requested_count,  # Show what the user requested
            "search_metadata": {
                "ai_parsing": "gpt-5-nano",
                "vector_search": True,
                "structured_filters": len(applied_filters),
                "semantic_terms": len(all_search_terms),
                "result_limiting": requested_count is not None
            }
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Search] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Advanced search error: {str(e)}")


async def _parse_query_with_ai(query: str) -> dict:
    """Use OpenAI gpt-5-nano to parse complex natural language queries and extract structured entities."""
    try:
        # Ensure OpenAI client is available
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            print("[Search] OpenAI not available, using basic parsing")
            return {"search_terms": [query]}

        client = OpenAI(api_key=api_key)
        
        # Get current date for relative date parsing
        from datetime import datetime, timedelta
        today = datetime.now()
        current_year = today.year
        current_month = today.month
        
        system_prompt = f"""You are an expert email search query parser. Analyze the user's natural language query and extract key entities into a structured JSON object.

Current date: {today.strftime('%Y-%m-%d')} (for relative date parsing)

Extract these entities:
- 'action': String - "search" (default), "summarize" (if user wants a summary/digest), or "answer" (if user asks a question and wants a conversational response)
- 'search_terms': List of ALL relevant keywords/phrases for semantic search (include names, topics, and content-specific terms)
- 'sender': Sender name/email (can be partial, e.g., "Bob", "accounting", "wellfound", "linkedin")  
- 'recipient': Recipient name/email (if specified)
- 'subject_keywords': Keywords that should appear in email subject (prioritize specific content terms over generic words)
- 'date_range': Object with 'start' and 'end' in YYYY-MM-DD format
- 'file_type': File extension (pdf, docx, xlsx, pptx, etc.)
- 'attachment_required': Boolean - true if query specifically mentions attachments
- 'priority': String - "high", "urgent", "important" if mentioned
- 'email_type': String - "meeting", "invoice", "receipt", "report", etc.
- 'status': String - "unread", "read", "starred" if mentioned (for Gmail API filtering)
- 'recency_priority': Boolean - true if query mentions "newest", "latest", "recent", "new", "last" (without specific timeframe)
- 'result_count': Integer - specific number of results requested (e.g., "13", "top 5", "last 3", "first 10")

IMPORTANT: 
1. For content-specific queries (e.g., "about charlie kirk"), extract the specific content terms as the primary search_terms and subject_keywords.
2. For recency queries (e.g., "newest wellfound email"), set recency_priority to true and focus on the sender/service name.
3. Company/service names should be extracted as both search_terms and sender (e.g., "wellfound", "linkedin", "github").

Date parsing examples:
- "last week" → {{"start": "{(today - timedelta(weeks=1)).strftime('%Y-%m-%d')}", "end": "{today.strftime('%Y-%m-%d')}"}}
- "last month" → {{"start": "{(today.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')}", "end": "{(today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')}"}}
- "last September" → {{"start": "{current_year-1 if current_month < 9 else current_year}-09-01", "end": "{current_year-1 if current_month < 9 else current_year}-09-30"}}
- "yesterday" → {{"start": "{(today - timedelta(days=1)).strftime('%Y-%m-%d')}", "end": "{(today - timedelta(days=1)).strftime('%Y-%m-%d')}"}}
- "this year" → {{"start": "{current_year}-01-01", "end": "{current_year}-12-31"}}

Examples:
Query: "Any news about MongoDB?"
Response: {{"action": "answer", "search_terms": ["MongoDB", "news", "updates"], "sender": "MongoDB"}}

Query: "What's the latest from GitHub?"
Response: {{"action": "answer", "search_terms": ["GitHub", "latest", "updates"], "sender": "GitHub"}}

Query: "Did I get any emails from my boss this week?"
Response: {{"action": "answer", "search_terms": ["boss"], "sender": "boss", "date_range": {{"start": "{(today - timedelta(weeks=1)).strftime('%Y-%m-%d')}", "end": "{today.strftime('%Y-%m-%d')}"}}}}

Query: "Summarize my unread emails from this week"
Response: {{"action": "summarize", "search_terms": ["unread"], "status": "unread", "date_range": {{"start": "{(today - timedelta(weeks=1)).strftime('%Y-%m-%d')}", "end": "{today.strftime('%Y-%m-%d')}"}}}}

Query: "Give me a digest of all emails from accounting this month"
Response: {{"action": "summarize", "search_terms": ["accounting"], "sender": "accounting", "date_range": {{"start": "{today.replace(day=1).strftime('%Y-%m-%d')}", "end": "{today.strftime('%Y-%m-%d')}"}}}}

Query: "Give me my last 13 DoorDash orders"
Response: {{"action": "search", "search_terms": ["DoorDash", "orders", "delivery"], "sender": "DoorDash", "email_type": "receipt", "recency_priority": true, "result_count": 13}}

Query: "Show me top 5 emails from LinkedIn"
Response: {{"action": "search", "search_terms": ["LinkedIn"], "sender": "LinkedIn", "result_count": 5}}

Query: "Find my last 3 emails about meetings"
Response: {{"action": "search", "search_terms": ["meeting", "appointment"], "email_type": "meeting", "subject_keywords": ["meeting"], "recency_priority": true, "result_count": 3}}

Query: "Find the PDF from Bob last September"
Response: {{"search_terms": ["PDF", "document"], "sender": "Bob", "date_range": {{"start": "2024-09-01", "end": "2024-09-30"}}, "file_type": "pdf", "attachment_required": true}}

Query: "urgent emails from accounting about invoices this month"  
Response: {{"search_terms": ["invoices", "billing"], "sender": "accounting", "priority": "urgent", "email_type": "invoice", "date_range": {{"start": "{today.replace(day=1).strftime('%Y-%m-%d')}", "end": "{today.strftime('%Y-%m-%d')}"}}}}

Query: "meeting invites from Sarah last week"
Response: {{"search_terms": ["meeting", "invite", "appointment"], "sender": "Sarah", "email_type": "meeting", "subject_keywords": ["meeting", "invite"], "date_range": {{"start": "{(today - timedelta(weeks=1)).strftime('%Y-%m-%d')}", "end": "{today.strftime('%Y-%m-%d')}"}}}}

Query: "find the email from new york times about charlie kirk"
Response: {{"search_terms": ["charlie kirk", "charlie", "kirk"], "sender": "new york times", "subject_keywords": ["charlie kirk", "charlie", "kirk"]}}

Query: "newest wellfound email"
Response: {{"search_terms": ["wellfound"], "sender": "wellfound", "recency_priority": true, "subject_keywords": ["wellfound"]}}

Query: "latest email from linkedin"
Response: {{"search_terms": ["linkedin"], "sender": "linkedin", "recency_priority": true}}

Query: "Excel reports with quarterly data"
Response: {{"search_terms": ["quarterly", "data", "report"], "file_type": "xlsx", "attachment_required": true, "email_type": "report", "subject_keywords": ["quarterly", "report"]}}

Only include entities that are clearly mentioned or strongly implied. Return valid JSON only."""
        
        user_prompt = f"Parse this email search query: '{query}'"
        
        print(f"[Search] Parsing complex query with GPT-5-nano: '{query}'")
        
        # Use Responses API with gpt-5-nano for structured parsing
        try:
            with client.responses.stream(
                model="gpt-5-nano",
                input=f"System: {system_prompt}\n\nUser: {user_prompt}",
            ) as stream:
                response_text = ""
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            response_text += delta
                
                print(f"[Search] GPT-5-nano response: {response_text}")
                
                # Parse JSON response
                try:
                    import json
                    # Clean response text - remove any markdown formatting
                    clean_response = response_text.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    
                    parsed_entities = json.loads(clean_response)
                    print(f"[Search] Successfully parsed entities: {parsed_entities}")
                    return parsed_entities
                except json.JSONDecodeError as je:
                    print(f"[Search] JSON parse error: {je}")
                    print(f"[Search] Raw response: {response_text}")
                    # Fallback to basic parsing
                    return {"search_terms": [query]}
                    
        except Exception as api_error:
            print(f"[Search] GPT-5-nano API error: {api_error}")
            # Try fallback with chat completion if Responses API fails
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # More reliable model for fallback
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"[Search] Fallback response: {response_text}")
                
                import json
                # Clean response text
                clean_response = response_text.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                parsed_entities = json.loads(clean_response)
                print(f"[Search] Fallback parsing successful: {parsed_entities}")
                return parsed_entities
                
            except Exception as fallback_error:
                print(f"[Search] Fallback parsing also failed: {fallback_error}")
                return {"search_terms": [query]}
        
    except Exception as e:
        print(f"[Search] Error in query parsing: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"search_terms": [query]}


def _validate_and_enhance_parsed_entities(entities: dict) -> dict:
    """Validate and enhance parsed entities with additional logic."""
    try:
        from datetime import datetime, timedelta
        
        # Validate and fix date ranges
        if entities.get("date_range"):
            date_range = entities["date_range"]
            today = datetime.now()
            
            # Handle common date parsing issues
            if date_range.get("start") and date_range.get("end"):
                try:
                    start_date = datetime.fromisoformat(date_range["start"])
                    end_date = datetime.fromisoformat(date_range["end"])
                    
                    # Ensure start is before end
                    if start_date > end_date:
                        date_range["start"], date_range["end"] = date_range["end"], date_range["start"]
                        print(f"[Search] Fixed date range order: {date_range}")
                        
                    # Ensure dates are not in the future (unless explicitly specified)
                    if start_date > today:
                        print(f"[Search] Warning: Start date {date_range['start']} is in the future")
                        
                except ValueError as ve:
                    print(f"[Search] Invalid date format in range: {ve}")
                    # Remove invalid date range
                    del entities["date_range"]
        
        # Enhance search terms based on email type
        if entities.get("email_type") and entities.get("search_terms"):
            email_type = entities["email_type"].lower()
            search_terms = entities["search_terms"]
            
            # Add related terms based on email type
            type_mappings = {
                "meeting": ["appointment", "calendar", "schedule", "zoom", "teams"],
                "invoice": ["billing", "payment", "amount", "due", "receipt"],
                "report": ["analysis", "summary", "data", "metrics", "quarterly"],
                "receipt": ["purchase", "transaction", "payment", "confirmation"],
            }
            
            if email_type in type_mappings:
                additional_terms = type_mappings[email_type]
                # Add terms that aren't already present
                for term in additional_terms:
                    if term not in [t.lower() for t in search_terms]:
                        search_terms.append(term)
                
                entities["search_terms"] = search_terms
                print(f"[Search] Enhanced search terms for {email_type}: {search_terms}")
        
        # Normalize sender names (handle common variations)
        if entities.get("sender"):
            sender = entities["sender"].strip()
            # Handle common name variations
            if len(sender.split()) == 1 and sender.isalpha():
                # Single name - could be first name, add common email patterns
                entities["sender_variations"] = [
                    sender,
                    f"{sender.lower()}@",  # Start of email
                    f"{sender.title()}",   # Title case
                ]
                print(f"[Search] Added sender variations: {entities['sender_variations']}")
        
        return entities
        
    except Exception as e:
        print(f"[Search] Error validating entities: {str(e)}")
        return entities


def _calculate_enhanced_relevance_score(result: dict, parsed_entities: dict, original_query: str) -> float:
    """Calculate enhanced relevance score that prioritizes content relevance over sender matching."""
    try:
        score = 0.0
        
        # Base semantic similarity (most important factor)
        base_similarity = result.get('similarity', 0)
        score += base_similarity * 100  # Scale to 0-100 range
        
        # Content relevance boost - check if search terms appear in content/subject
        search_terms = parsed_entities.get('search_terms', [])
        content = (result.get('content', '') + ' ' + result.get('subject', '')).lower()
        
        # Boost for exact keyword matches in content
        for term in search_terms:
            if term.lower() in content:
                score += 25  # Significant boost for keyword matches
                print(f"[Ranking] Boosted score by 25 for keyword '{term}' in content")
        
        # Boost for original query terms in content (handles cases where parsing might miss terms)
        original_words = original_query.lower().split()
        for word in original_words:
            if len(word) > 3 and word in content:  # Skip short words like "the", "and"
                score += 15
                print(f"[Ranking] Boosted score by 15 for original word '{word}' in content")
        
        # Subject line relevance (higher weight than general content)
        subject = result.get('subject', '').lower()
        for term in search_terms:
            if term.lower() in subject:
                score += 35  # Higher boost for subject matches
                print(f"[Ranking] Boosted score by 35 for keyword '{term}' in subject")
        
        # Sender matching bonus (higher priority for sender-specific queries)
        if parsed_entities.get('sender'):
            sender_name = parsed_entities['sender'].lower()
            result_sender = result.get('sender', '').lower()
            
            # Strong match if sender name appears in email sender
            if sender_name in result_sender:
                sender_bonus = 50  # High bonus for direct sender matches
                score += sender_bonus
                print(f"[Ranking] Added strong sender bonus: {sender_bonus} ('{sender_name}' found in '{result_sender}')")
            elif result.get('matched_filters'):
                sender_bonus = 15  # Moderate bonus for filter matches
                score += sender_bonus
                print(f"[Ranking] Added sender filter bonus: {sender_bonus}")
        
        # Penalty for promotional/marketing emails (common patterns)
        promotional_keywords = [
            'sale', 'offer', 'discount', 'save', 'deal', 'limited time', 
            'subscribe', 'unsubscribe', 'newsletter', 'promotion', '$',
            'free', 'buy now', 'click here'
        ]
        promotional_count = sum(1 for keyword in promotional_keywords if keyword in content)
        if promotional_count >= 3:  # Multiple promotional indicators
            penalty = min(promotional_count * 5, 30)  # Cap penalty at 30
            score -= penalty
            print(f"[Ranking] Applied promotional penalty: -{penalty} (found {promotional_count} promotional keywords)")
        
        # Recency priority boost (for "newest", "latest" queries)
        if parsed_entities.get('recency_priority'):
            try:
                # Use actual created_at timestamp for proper recency scoring
                created_at = result.get('created_at')
                if created_at:
                    from datetime import datetime, timezone
                    try:
                        # Parse the timestamp
                        if isinstance(created_at, str):
                            # Handle different timestamp formats
                            if 'T' in created_at:
                                email_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            else:
                                email_time = datetime.fromisoformat(created_at)
                        else:
                            email_time = created_at
                        
                        # Calculate days since email
                        now = datetime.now(timezone.utc)
                        if email_time.tzinfo is None:
                            email_time = email_time.replace(tzinfo=timezone.utc)
                        
                        days_ago = (now - email_time).total_seconds() / 86400  # Convert to days
                        
                        # Higher bonus for more recent emails (max 50 points for today, decreasing)
                        if days_ago <= 1:
                            recency_bonus = 50  # Today
                        elif days_ago <= 7:
                            recency_bonus = 40 - (days_ago * 5)  # This week, decreasing
                        elif days_ago <= 30:
                            recency_bonus = 15 - (days_ago * 0.3)  # This month, decreasing
                        else:
                            recency_bonus = 5  # Older emails get small bonus
                        
                        recency_bonus = max(5, recency_bonus)  # Minimum 5 points
                        score += recency_bonus
                        print(f"[Ranking] Added recency bonus: {recency_bonus:.1f} ({days_ago:.1f} days ago)")
                        
                    except (ValueError, TypeError) as e:
                        print(f"[Ranking] Error parsing created_at '{created_at}': {e}")
                        # Fallback: give small bonus for recency queries even if we can't parse date
                        score += 10
                else:
                    # Fallback: Use thread_id as rough recency indicator when no timestamp available
                    thread_id = result.get('thread_id', '')
                    if thread_id and len(thread_id) > 10:
                        try:
                            # Convert hex thread_id to int for rough recency scoring
                            thread_num = int(thread_id, 16)
                            # Newer thread_ids tend to be higher (rough heuristic)
                            # Normalize to 0-25 range
                            recency_bonus = min(25, (thread_num % 10000) / 400)
                            score += recency_bonus
                            print(f"[Ranking] Added thread-based recency bonus: {recency_bonus:.1f} (thread_id: {thread_id})")
                        except ValueError:
                            score += 10  # Small fallback bonus
                    else:
                        score += 10  # Small fallback bonus
                        
            except Exception as e:
                print(f"[Ranking] Error in recency calculation: {e}")
                # Fallback: give small bonus for recency queries
                score += 10
        
        # Boost for breaking news or important content
        important_keywords = ['breaking', 'urgent', 'important', 'alert', 'update']
        for keyword in important_keywords:
            if keyword in subject:
                score += 20
                print(f"[Ranking] Boosted score by 20 for important keyword '{keyword}' in subject")
        
        # Length penalty for very short content (likely promotional)
        content_length = len(result.get('content', ''))
        if content_length < 200:
            length_penalty = (200 - content_length) / 10  # Gradual penalty
            score -= min(length_penalty, 20)  # Cap penalty
            print(f"[Ranking] Applied length penalty: -{min(length_penalty, 20)} (content length: {content_length})")
        
        # Ensure score is non-negative
        final_score = max(score, 0)
        
        print(f"[Ranking] Email '{result.get('subject', '')[:50]}...' - Final relevance score: {final_score:.2f}")
        return final_score
        
    except Exception as e:
        print(f"[Ranking] Error calculating relevance score: {str(e)}")
        # Fallback to base similarity
        return result.get('similarity', 0) * 100


async def _expand_query_with_ai(original_query: str) -> list[str]:
    """Use OpenAI to expand the search query with related terms."""
    try:
        # Ensure OpenAI client is available
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            print("[Search] OpenAI not available, using original query only")
            return [original_query]

        client = OpenAI(api_key=api_key)
        
        expansion_prompt = f"""Given the search query: "{original_query}"

Generate 4-6 related search terms that would help find relevant emails. Include:
- Synonyms and alternative phrasings
- Related concepts and topics
- Common variations and abbreviations
- Context-specific terms

Return ONLY a comma-separated list of terms, no explanations.

Example:
Query: "meeting schedule"
Response: meeting schedule, appointment, calendar, meeting time, scheduled meeting, conference call

Query: "{original_query}"
Response:"""

        print(f"[Search] Expanding query with OpenAI...")
        
        # Use chat completion for query expansion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a more reliable model for this task
            messages=[
                {"role": "system", "content": "You are a helpful assistant that expands search queries with related terms."},
                {"role": "user", "content": expansion_prompt}
            ],
            max_tokens=100,
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        expanded_text = response.choices[0].message.content.strip()
        print(f"[Search] OpenAI response: {expanded_text}")
        
        # Parse the comma-separated terms
        expanded_terms = [term.strip() for term in expanded_text.split(',') if term.strip()]
        
        # Always include the original query first
        final_keywords = [original_query]
        
        # Add expanded terms, avoiding duplicates
        for term in expanded_terms:
            if term.lower() != original_query.lower() and term not in final_keywords:
                final_keywords.append(term)
        
        # Limit to maximum 6 keywords to avoid too many API calls
        return final_keywords[:6]
        
    except Exception as e:
        print(f"[Search] Error in AI query expansion: {str(e)}")
        # Fallback to original query if expansion fails
        return [original_query]


@app.post("/sync-inbox")
async def sync_inbox(request: dict):
    """Fetch Gmail threads with optional time-based filtering, embed content, and upsert into Supabase emails table.

    Expected request body: { 
        "userId": "..." (optional),
        "time_range_days": int (optional) - number of days to look back (e.g., 1, 7, 30)
    }
    """
    try:
        print(f"[Sync] Starting inbox sync with request: {request}")
        sb = _get_supabase()

        user_id = request.get("userId")
        time_range_days = request.get("time_range_days")
        
        print(f"[Sync] User ID: {user_id}")
        print(f"[Sync] Time range: {time_range_days} days" if time_range_days else "[Sync] Time range: default (last 50 messages)")

        # Build Gmail service
        print("[Sync] Building Gmail service...")
        service = _build_gmail_service(user_id)
        print("[Sync] Gmail service built successfully")

        # Build query parameters for Gmail API
        query_params = {"userId": "me"}
        
        if time_range_days:
            # Use Gmail search query to filter by time range
            query_params["q"] = f"in:inbox newer_than:{time_range_days}d"
            query_params["maxResults"] = 200  # Increase limit for time-based searches
            print(f"[Sync] Using Gmail query: {query_params['q']}")
        else:
            # Default behavior - last 50 messages
            query_params["maxResults"] = 50
        
        # List threads with time filtering
        print("[Sync] Fetching threads list...")
        threads_resp = service.users().threads().list(**query_params).execute()
        threads = threads_resp.get('threads', []) or []
        print(f"[Sync] Found {len(threads)} threads")

        if not threads:
            print("[Sync] No threads found, returning 0")
            return {
                "status": "success", 
                "indexed_count": 0,
                "time_range_days": time_range_days,
                "query_used": query_params.get("q", "default")
            }

        print("[Sync] Loading embedding model...")
        model = _get_embedding_model()
        print("[Sync] Embedding model loaded")

        indexed = 0
        for i, t in enumerate(threads):
            tid = t.get('id')
            if not tid:
                print(f"[Sync] Thread {i+1}/{len(threads)}: No ID, skipping")
                continue
            
            print(f"[Sync] Processing thread {i+1}/{len(threads)}: {tid}")
            try:
                thread_full = service.users().threads().get(userId="me", id=tid, format="full").execute()
                print(f"[Sync] Thread {tid}: Fetched full thread data")
                
                content = _fetch_thread_plaintext(service, tid)
                print(f"[Sync] Thread {tid}: Extracted content ({len(content)} chars)")
                
                meta = _parse_thread_metadata(thread_full)
                subject = meta.get('subject', '')
                sender = meta.get('from', '')
                email_timestamp = meta.get('timestamp')
                print(f"[Sync] Thread {tid}: Subject='{subject[:50]}...', Sender='{sender}', Timestamp='{email_timestamp}'")

                text_for_embedding = (subject + "\n\n" + content).strip() or content
                print(f"[Sync] Thread {tid}: Generating embedding for {len(text_for_embedding)} chars")
                emb = model.encode(text_for_embedding, normalize_embeddings=True).tolist()  # 384 dims
                print(f"[Sync] Thread {tid}: Generated embedding with {len(emb)} dimensions")

                # Upsert into Supabase emails table
                from datetime import datetime, timezone
                payload = {
                    "google_user_id": user_id,  # may be None; acceptable if you use RLS appropriately
                    "thread_id": tid,
                    "subject": subject,
                    "sender": sender,
                    "content": content,
                    "embedding": emb,
                    "created_at": email_timestamp or datetime.now(timezone.utc).isoformat(),  # Use actual email time or fallback
                }
                print(f"[Sync] Thread {tid}: Upserting to Supabase...")
                sb.table("emails").upsert(payload, on_conflict="thread_id").execute()  # type: ignore[attr-defined]
                indexed += 1
                print(f"[Sync] Thread {tid}: Successfully indexed ({indexed}/{len(threads)})")
            except Exception as e:
                print(f"[Sync] Thread {tid}: ERROR - {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[Sync] Completed: {indexed}/{len(threads)} threads indexed successfully")
        return {
            "status": "success", 
            "indexed_count": indexed,
            "total_threads_found": len(threads),
            "time_range_days": time_range_days,
            "query_used": query_params.get("q", "default")
        }
    except HTTPException as he:
        print(f"[ERROR] HTTPException in /sync-inbox: {he.status_code} - {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Exception in /sync-inbox: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sync error: {str(e)}")

@app.get("/debug-env")
def debug_env():
    """Debug endpoint to check environment variables"""
    import os
    return {
        "GOOGLE_CLIENT_ID": bool(os.getenv("GOOGLE_CLIENT_ID")),
        "GOOGLE_CLIENT_SECRET": bool(os.getenv("GOOGLE_CLIENT_SECRET")),
        "GOOGLE_CLIENT_ID_VALUE": os.getenv("GOOGLE_CLIENT_ID"),
        "working_directory": os.getcwd(),
        "env_file_exists": os.path.exists(".env")
    }


@app.get("/debug-emails-count")
def debug_emails_count():
    """Count emails rows in Supabase for verification."""
    try:
        sb = _get_supabase()
        res = sb.table("emails").select("thread_id", count="exact").execute()  # type: ignore[attr-defined]
        count = getattr(res, "count", 0) or 0
        return {"emails_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emails count error: {str(e)}")

@app.post("/clear-emails")
def clear_emails():
    """Clear all emails from the database. Use with caution!"""
    try:
        sb = _get_supabase()
        print("[Clear] Starting to clear all emails from database...")
        
        # Delete all emails
        result = sb.table("emails").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()  # type: ignore[attr-defined]
        
        # Count remaining emails
        count_res = sb.table("emails").select("thread_id", count="exact").execute()  # type: ignore[attr-defined]
        remaining_count = getattr(count_res, "count", 0) or 0
        
        print(f"[Clear] Cleared emails. Remaining count: {remaining_count}")
        
        return {
            "status": "success", 
            "message": "All emails cleared from database",
            "remaining_count": remaining_count
        }
    except Exception as e:
        print(f"[Clear] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear emails error: {str(e)}")

@app.post("/add-created-at-column")
def add_created_at_column():
    """Add created_at timestamp column to emails table."""
    try:
        sb = _get_supabase()
        print("[Schema] Adding created_at column to emails table...")
        
        # Execute SQL to add the column
        # Note: Supabase Python client doesn't have direct DDL support,
        # so we'll use the rpc() method to execute raw SQL
        sql_query = """
        ALTER TABLE emails 
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
        """
        
        result = sb.rpc('exec_sql', {'sql': sql_query}).execute()
        
        print("[Schema] Successfully added created_at column")
        
        return {
            "status": "success",
            "message": "created_at column added to emails table"
        }
    except Exception as e:
        print(f"[Schema] Error adding column: {str(e)}")
        # Try alternative approach using direct SQL execution
        try:
            # Alternative: Use PostgREST's direct SQL execution if available
            result = sb.postgrest.rpc('exec_sql', {'sql': sql_query}).execute()
            return {
                "status": "success",
                "message": "created_at column added to emails table (alternative method)"
            }
        except Exception as e2:
            print(f"[Schema] Alternative method also failed: {str(e2)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Could not add column. You may need to add it manually in Supabase dashboard: ALTER TABLE emails ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();"
            )
