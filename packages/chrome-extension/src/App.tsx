import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [authStatus, setAuthStatus] = useState<any>(null)

  // Check authentication status on component mount
  const checkAuthStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/auth/status', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const status = await response.json();
        setAuthStatus(status);
        // If there are authenticated users, consider user as authenticated
        setIsAuthenticated(status.authenticated_users > 0);
      }
    } catch (error) {
      console.error('Failed to check auth status:', error);
    }
  };

  // Check auth status when component mounts
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const handleLogin = async () => {
    setIsLoading(true)
    try {
      console.log('Initiating OAuth flow...');
      
      // Open Google OAuth in a new window
      const oauthWindow = window.open('http://localhost:8000/login', '_blank', 'width=600,height=600');
      
      if (oauthWindow) {
        // Poll for OAuth completion by checking auth status
        const checkAuth = setInterval(async () => {
          try {
            const response = await fetch('http://localhost:8000/auth/status', {
              credentials: 'include'
            });
            
            if (response.ok) {
              const status = await response.json();
              if (status.authenticated_users > 0) {
                // OAuth completed successfully
                clearInterval(checkAuth);
                setIsAuthenticated(true);
                oauthWindow.close();
                console.log('OAuth completed successfully!');
              }
            }
          } catch (error) {
            console.log('Checking auth status...', error);
          }
        }, 1000); // Check every second
        
        // Timeout after 2 minutes
        setTimeout(() => {
          clearInterval(checkAuth);
          if (!isAuthenticated) {
            oauthWindow.close();
            alert('OAuth timed out. Please try again.');
          }
        }, 120000);
        
      } else {
        throw new Error('Popup blocked. Please allow popups for this site.');
      }
    } catch (error) {
      console.error('Login failed:', error);
      alert('Login failed: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsLoading(false);
    }
  }

  const handleSummarize = async () => {
    setIsLoading(true)
    try {
      // Call backend /summarize endpoint
      const response = await fetch('http://localhost:8000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'thread_id=test123', // TODO: Get actual thread ID from Gmail
        credentials: 'include'
      });
      
      if (response.ok) {
        const summary = await response.json();
        console.log('Summary received:', summary);
        alert(`Summary: ${summary.summary}`);
      } else {
        throw new Error('Failed to get summary');
      }
    } catch (error) {
      console.error('Summarization failed:', error);
      alert('Failed to get summary. Please check if the backend is running.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="w-80 p-4 bg-white">
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">📧 MailFind</h1>
        <p className="text-sm text-gray-600">AI-powered email summarization</p>
      </div>

      {!isAuthenticated ? (
        <div className="space-y-4">
          <p className="text-sm text-gray-700">
            Connect your Gmail account to start summarizing emails
          </p>
          <button
            onClick={handleLogin}
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors"
          >
            {isLoading ? 'Connecting...' : 'Connect with Google'}
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded-md p-3">
            <p className="text-sm text-green-800">
              ✅ Connected to Gmail
            </p>
          </div>
          <button
            onClick={handleSummarize}
            disabled={isLoading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors"
          >
            {isLoading ? 'Summarizing...' : 'Summarize Current Email'}
          </button>
        </div>
      )}

      <div className="mt-6 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          Click the "📧 Summarize" button in Gmail to summarize email threads
        </p>
        
        {authStatus && (
          <div className="mt-4 p-3 bg-gray-50 rounded-md">
            <p className="text-xs text-gray-600 text-center">
              Backend Status: {authStatus.authenticated_users} users, {authStatus.active_states} active states
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
