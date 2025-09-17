import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  // const [authStatus, setAuthStatus] = useState<any>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [syncLoading, setSyncLoading] = useState(false)

  // Check authentication status on component mount
  const checkAuthStatus = async () => {
    try {
      console.log('🔍 [Auth] Checking authentication status...');
      const response = await fetch('http://localhost:8000/auth/status', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const status = await response.json();
        console.log('📊 [Auth] Current auth status:', status);
        // setAuthStatus(status);
        // If there are authenticated users, consider user as authenticated
        const wasAuthenticated = status.authenticated_users > 0;
        setIsAuthenticated(wasAuthenticated);
        console.log(`🔐 [Auth] Authentication state: ${wasAuthenticated ? 'Authenticated' : 'Not authenticated'}`);
      } else {
        console.warn(`⚠️ [Auth] Auth status check failed: ${response.status}`);
      }
    } catch (error) {
      console.error('❌ [Auth] Failed to check auth status:', error);
    }
  };

  // Check auth status when component mounts
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const handleLogin = async () => {
    setIsLoading(true)
    console.log('🚀 [OAuth] Starting OAuth flow...');
    
    try {
      // Open Google OAuth in a new window
      console.log('📱 [OAuth] Opening popup window...');
      const oauthWindow = window.open('http://localhost:8000/login', '_blank', 'width=600,height=600');
      
      if (oauthWindow) {
        console.log('✅ [OAuth] Popup window opened successfully');
        
        // Poll for OAuth completion by checking auth status
        let pollCount = 0;
        const checkAuth = setInterval(async () => {
          pollCount++;
          try {
            console.log(`🔍 [OAuth] Polling auth status (attempt ${pollCount})...`);
            const response = await fetch('http://localhost:8000/auth/status', {
              credentials: 'include'
            });
            
            if (response.ok) {
              const status = await response.json();
              console.log(`📊 [OAuth] Auth status:`, status);
              
              if (status.authenticated_users > 0) {
                // OAuth completed successfully
                console.log('🎉 [OAuth] Authentication successful! User authenticated.');
                clearInterval(checkAuth);
                setIsAuthenticated(true);
                oauthWindow.close();
                console.log('🔒 [OAuth] Popup window closed');
              } else {
                console.log(`⏳ [OAuth] Still waiting... (${status.authenticated_users} users, ${status.active_states} states)`);
              }
            } else {
              console.warn(`⚠️ [OAuth] Auth status check failed: ${response.status}`);
            }
          } catch (error) {
            console.error(`❌ [OAuth] Error checking auth status (attempt ${pollCount}):`, error);
          }
        }, 1000); // Check every second
        
        // Timeout after 2 minutes
        setTimeout(() => {
          console.log('⏰ [OAuth] Timeout reached (2 minutes)');
          clearInterval(checkAuth);
          if (!isAuthenticated) {
            console.warn('⚠️ [OAuth] OAuth timed out, closing popup');
            oauthWindow.close();
            alert('OAuth timed out. Please try again.');
          }
        }, 120000);
        
      } else {
        throw new Error('Popup blocked. Please allow popups for this site.');
      }
    } catch (error) {
      console.error('💥 [OAuth] Login failed:', error);
      alert('Login failed: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsLoading(false);
      console.log('🏁 [OAuth] Login flow completed');
    }
  }



  const handleLogout = async () => {
    setIsLoading(true)
    console.log('🚪 [Logout] Starting logout process...');
    
    try {
      // Call backend /logout endpoint
      console.log('🌐 [Logout] Calling backend /logout endpoint...');
      const response = await fetch('http://localhost:8000/logout', {
        method: 'POST',
        credentials: 'include'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ [Logout] Logout successful:', result);
        
        // Update local state
        setIsAuthenticated(false);
        // setAuthStatus(null);
        
        // Refresh auth status
        await checkAuthStatus();
        
        console.log('🔐 [Logout] User logged out successfully');
      } else {
        console.error(`❌ [Logout] Backend error: ${response.status} ${response.statusText}`);
        throw new Error(`Failed to logout: ${response.status}`);
      }
    } catch (error) {
      console.error('💥 [Logout] Logout failed:', error);
      alert('Failed to logout. Please try again.');
    } finally {
      setIsLoading(false);
      console.log('🏁 [Logout] Logout flow completed');
    }
  }

  const handleSyncInbox = async () => {
    setSyncLoading(true)
    console.log('📥 [Sync] Starting inbox sync...');
    
    try {
      const response = await fetch('http://localhost:8000/sync-inbox', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
        credentials: 'include'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ [Sync] Sync successful:', result);
        alert(`✅ Synced ${result.indexed_count} emails successfully!`);
      } else {
        console.error(`❌ [Sync] Backend error: ${response.status} ${response.statusText}`);
        throw new Error(`Failed to sync: ${response.status}`);
      }
    } catch (error) {
      console.error('💥 [Sync] Sync failed:', error);
      alert('Failed to sync inbox. Please try again.');
    } finally {
      setSyncLoading(false);
      console.log('🏁 [Sync] Sync flow completed');
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true)
    console.log('🔍 [Search] Starting search for:', searchQuery);
    
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: searchQuery }),
        credentials: 'include'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ [Search] Search successful:', result);
        setSearchResults(result.results || []);
      } else {
        console.error(`❌ [Search] Backend error: ${response.status} ${response.statusText}`);
        throw new Error(`Search failed: ${response.status}`);
      }
    } catch (error) {
      console.error('💥 [Search] Search failed:', error);
      alert('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
      console.log('🏁 [Search] Search flow completed');
    }
  }

  return (
    <div className="w-96 bg-slate-900 text-slate-100 p-6 flex flex-col gap-y-5 font-sans">
      {/* Header Section */}
      <div className="flex flex-col items-center">
        <h1 className="text-2xl font-bold text-center text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-violet-500">
          MailFind
        </h1>
        <p className="text-sm text-slate-400 text-center -mt-1">
          AI-powered email assistant
        </p>
      </div>

      {!isAuthenticated ? (
        <div className="flex flex-col gap-y-4">
          <p className="text-sm text-slate-300 text-center">
            Connect your Gmail account to start using AI-powered email search
          </p>
          <button
            onClick={handleLogin}
            disabled={isLoading}
            className="bg-violet-600 text-white font-bold py-2 rounded-lg hover:bg-violet-500 transition-colors disabled:bg-slate-600"
          >
            {isLoading ? 'Connecting...' : 'Connect with Google'}
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-y-5">
          {/* Connection Status */}
          <div className="flex items-center justify-center gap-x-2 text-sm text-green-400 bg-green-900/50 rounded-full px-3 py-1">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            Connected to Gmail
          </div>
          
          {/* Sync Inbox Section */}
          <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl flex flex-col gap-y-3">
            <div className="flex items-center gap-x-2 font-semibold">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
              </svg>
              Sync Inbox
            </div>
            <div className="grid grid-cols-3 gap-x-2">
              <button className="bg-slate-700 text-slate-300 text-xs font-semibold py-2 rounded-lg hover:bg-slate-600 transition-colors">
                24h
              </button>
              <button className="bg-slate-700 text-slate-300 text-xs font-semibold py-2 rounded-lg hover:bg-slate-600 transition-colors">
                7d
              </button>
              <button className="bg-slate-700 text-slate-300 text-xs font-semibold py-2 rounded-lg hover:bg-slate-600 transition-colors">
                30d
              </button>
            </div>
            <button
              onClick={handleSyncInbox}
              disabled={syncLoading}
              className="bg-violet-600 text-white font-bold py-2 rounded-lg hover:bg-violet-500 transition-colors disabled:bg-slate-600"
            >
              {syncLoading ? 'Syncing...' : 'Sync Inbox'}
            </button>
          </div>

          {/* AI Assistant (Search) Section */}
          <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl flex flex-col gap-y-3">
            <div className="flex items-center gap-x-2 font-semibold">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              AI Email Assistant
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Ask about your emails..."
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
            <button
              onClick={handleSearch}
              disabled={isSearching || !searchQuery.trim()}
              className="bg-violet-600 text-white font-bold py-2 rounded-lg hover:bg-violet-500 transition-colors disabled:bg-slate-600"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl flex flex-col gap-y-3">
              <div className="font-semibold">Search Results ({searchResults.length})</div>
              <div className="max-h-60 overflow-y-auto space-y-2">
                {searchResults.map((result, index) => (
                  <div key={index} className="bg-slate-800 border border-slate-700 rounded-lg p-3">
                    <div className="text-xs font-medium text-slate-200 mb-1">
                      {result.sender || 'Unknown Sender'}
                    </div>
                    <div className="text-xs text-slate-300 mb-2">
                      {result.subject || 'No Subject'}
                    </div>
                    {result.similarity && (
                      <div className="text-xs text-violet-400">
                        Similarity: {Math.round(result.similarity * 100)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Logout Button */}
          <button
            onClick={handleLogout}
            disabled={isLoading}
            className="flex items-center justify-center gap-x-2 text-slate-400 text-sm hover:text-white transition-colors disabled:text-slate-500"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
            {isLoading ? 'Logging out...' : 'Logout'}
          </button>
        </div>
      )}
    </div>
  )
}

export default App
