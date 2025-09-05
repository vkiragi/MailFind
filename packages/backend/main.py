import os
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import secrets
import json
import os
from typing import Optional

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Will validate at runtime

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

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

# In-memory storage for OAuth states and credentials (temporary)
oauth_states = {}
user_credentials = {}

# OAuth configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
REDIRECT_URI = "http://localhost:8000/callback"

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
            state=state
        )
        
        return RedirectResponse(url=authorization_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.get("/callback")
def callback(code: str, state: str):
    """Handle OAuth callback and token exchange"""
    try:
        # Verify state to prevent CSRF
        if state not in oauth_states:
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        # Remove used state
        del oauth_states[state]
        
        flow = get_flow()
        flow.fetch_token(code=code)
        
        # Store credentials (in memory for now)
        user_id = f"user_{len(user_credentials) + 1}"
        user_credentials[user_id] = {
            "token": flow.credentials.token,
            "refresh_token": flow.credentials.refresh_token,
            "token_uri": flow.credentials.token_uri,
            "client_id": flow.credentials.client_id,
            "client_secret": flow.credentials.client_secret,
            "scopes": flow.credentials.scopes
        }
        
        return {
            "message": "Authentication successful",
            "user_id": user_id,
            "scopes": flow.credentials.scopes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Callback error: {str(e)}")

@app.get("/auth/status")
def auth_status():
    """Check authentication status"""
    return {
        "authenticated_users": len(user_credentials),
        "active_states": len(oauth_states)
    }

@app.post("/logout")
def logout():
    """Logout user and clear credentials"""
    try:
        # For now, clear all credentials (in a real app, you'd clear specific user's credentials)
        global user_credentials, oauth_states
        
        # Clear all stored credentials
        user_credentials.clear()
        oauth_states.clear()
        
        return {
            "message": "Logout successful",
            "authenticated_users": 0,
            "active_states": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout error: {str(e)}")

@app.post("/summarize")
def summarize_email_thread(request: dict):
    """Summarize an email thread using OpenAI.

    Expected request body:
    { "threadId": "...", "emailText": "..." }

    Until Gmail API integration is complete, `emailText` is optional.
    If it's not provided, we will generate a best-effort placeholder summary
    based on the thread id.
    """
    try:
        # Extract threadId from JSON body
        thread_id = request.get("threadId")
        email_text: Optional[str] = request.get("emailText")
        
        if not thread_id:
            raise HTTPException(status_code=400, detail="threadId is required")
        
        print(f"Received summarization request for thread ID: {thread_id}")

        # Ensure OpenAI client is available and API key present
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            raise HTTPException(status_code=500, detail="OpenAI SDK not available or OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        # If we don't yet have the real email text (Gmail API to be integrated),
        # create a short prompt that references the thread id. Once Gmail API is
        # implemented, supply the actual concatenated email body text here.
        prompt_text = email_text if email_text else (
            f"You are an assistant that summarizes Gmail threads. We don't have the raw message bodies yet. "
            f"Provide a concise placeholder summary and likely next actions for thread id {thread_id}."
        )

        system_msg = (
            "You are a helpful assistant that writes concise, factual summaries of email threads. "
            "Output 3-5 sentences followed by up to 3 bullet point action items. If details are missing, "
            "note assumptions explicitly."
        )

        try:
            # Use Responses API for gpt-5-nano; avoid unsupported params
            resp = client.responses.create(
                model="gpt-5-nano",
                input=f"System: {system_msg}\n\nUser: {prompt_text}",
            )
            summary_text = ""
            try:
                summary_text = resp.output_text.strip()  # type: ignore[attr-defined]
            except Exception:
                if getattr(resp, "choices", None):
                    summary_text = resp.choices[0].message["content"].strip()  # type: ignore[index]
        except Exception as oe:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(oe)}")

        return {
            "thread_id": thread_id,
            "summary": summary_text,
            "model": "gpt-5-nano",
            "status": "ok"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.get("/summarize/{thread_id}")
def get_summary_status(thread_id: str):
    """Get summary status for a specific thread"""
    # TODO: Implement summary retrieval from database
    return {
        "thread_id": thread_id,
        "status": "mock_status",
        "message": "Summary endpoint working - ready for Gmail API integration"
    }
