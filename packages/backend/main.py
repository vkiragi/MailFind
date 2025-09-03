import os
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import secrets
import json

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend dev server
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

@app.post("/summarize")
def summarize_email_thread(thread_id: str = Form(...)):
    """Summarize an email thread (currently returns mock data)"""
    try:
        # TODO: Step 1.3 - Replace with real Gmail API integration
        # TODO: Step 4.1 - Replace with real OpenAI integration
        
        # Mock summary data for now
        mock_summary = {
            "thread_id": thread_id,
            "summary": "This is a mock summary of the email thread. The actual implementation will fetch the email content via Gmail API and generate a summary using OpenAI.",
            "key_points": [
                "Mock key point 1: Important discussion about project timeline",
                "Mock key point 2: Team members assigned to specific tasks", 
                "Mock key point 3: Next meeting scheduled for Friday"
            ],
            "participants": ["sender@example.com", "recipient@example.com"],
            "timestamp": "2025-01-02T23:56:00Z",
            "status": "mock_data"
        }
        
        return mock_summary
        
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
