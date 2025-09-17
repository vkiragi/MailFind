import os
from dotenv import load_dotenv
from pathlib import Path

# Same loading logic as main.py
load_dotenv()
try:
    root_env_path = Path(__file__).resolve().parents[2] / ".env"
    if root_env_path.exists():
        load_dotenv(dotenv_path=str(root_env_path), override=False)
        print(f"Loaded from root: {root_env_path}")
    else:
        print("No root .env found")
except Exception as e:
    print(f"Exception loading root .env: {e}")

def get_flow():
    """Create OAuth2 flow"""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    print(f"CLIENT_ID: {client_id}")
    print(f"CLIENT_SECRET exists: {bool(client_secret)}")
    
    if not client_id or not client_secret:
        print("ERROR: OAuth credentials not configured")
        return False
    else:
        print("SUCCESS: OAuth credentials are configured")
        return True

if __name__ == "__main__":
    result = get_flow()
    print(f"Result: {result}")
