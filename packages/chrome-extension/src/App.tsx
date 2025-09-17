import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [authStatus, setAuthStatus] = useState<any>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchSummary, setSearchSummary] = useState<string>('')
  const [searchAction, setSearchAction] = useState<string>('')
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
        setAuthStatus(status);
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
        setAuthStatus(null);
        
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

  const handleSyncInbox = async (timeRangeDays: number) => {
    setSyncLoading(true)
    const timeLabel = `last ${timeRangeDays} day${timeRangeDays > 1 ? 's' : ''}`;
    console.log(`📥 [Sync] Starting inbox sync (${timeLabel})...`);
    
    try {
      const requestBody = { time_range_days: timeRangeDays };
      
      const response = await fetch('http://localhost:8000/sync-inbox', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        credentials: 'include'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ [Sync] Sync successful:', result);
        
        // Create more informative success message
        const timeInfo = ` from the last ${result.time_range_days} day${result.time_range_days > 1 ? 's' : ''}`;
        const queryInfo = result.query_used !== 'default' ? 
          `\nQuery: ${result.query_used}` : '';
        
        alert(`✅ Synced ${result.indexed_count} emails successfully${timeInfo}!${queryInfo}`);
      } else {
        console.error(`❌ [Sync] Backend error: ${response.status} ${response.statusText}`);
        if (response.status === 401) {
          try {
            const body = await response.json().catch(() => ({} as any));
            const msg = body?.detail || 'Authentication required. Please log in again.';
            console.warn('⚠️ [Sync] 401 received:', msg);
          } catch (_) {}
          alert('Login required. Opening Google sign-in...');
          window.open('http://localhost:8000/login', '_blank', 'width=600,height=600');
          // Re-check auth shortly after to update UI
          setTimeout(() => {
            checkAuthStatus();
          }, 3000);
          return;
        }
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
        
        // Handle different response types
        if (result.action === 'summarize') {
          // Summarization response
          setSearchAction('summarize');
          setSearchSummary(result.summary || 'No summary available');
          setSearchResults([]); // Clear search results
        } else {
          // Regular search response
          setSearchAction('search');
          setSearchResults(result.results || []);
          setSearchSummary(''); // Clear summary
        }
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
                 
                 {/* Sync Inbox Section */}
                 <div className="space-y-3">
                   <h3 className="text-sm font-medium text-gray-700">📥 Sync Inbox</h3>
                   <p className="text-xs text-gray-600">Index your emails for semantic search</p>
                   
                   {/* Time-based Indexing Buttons */}
                   <div className="grid grid-cols-2 gap-2">
                     <button
                       onClick={() => handleSyncInbox(1)}
                       disabled={syncLoading}
                       className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-3 rounded-md transition-colors text-xs"
                     >
                       {syncLoading ? '⏳' : '📅 Last 24h'}
                     </button>
                     <button
                       onClick={() => handleSyncInbox(7)}
                       disabled={syncLoading}
                       className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-medium py-2 px-3 rounded-md transition-colors text-xs"
                     >
                       {syncLoading ? '⏳' : '📅 Last 7d'}
                     </button>
                   </div>
                   
                   {/* 30-day sync button */}
                   <button
                     onClick={() => handleSyncInbox(30)}
                     disabled={syncLoading}
                     className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors text-sm"
                   >
                     {syncLoading ? 'Syncing...' : '📅 Last 30 Days'}
                   </button>
                 </div>

                 {/* Search Section */}
                 <div className="space-y-2">
                   <h3 className="text-sm font-medium text-gray-700">🔍 Search Emails</h3>
                   <div className="flex space-x-2">
                     <input
                       type="text"
                       value={searchQuery}
                       onChange={(e) => setSearchQuery(e.target.value)}
                       onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                       placeholder="Search your emails..."
                       className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                     />
                     <button
                       onClick={handleSearch}
                       disabled={isSearching || !searchQuery.trim()}
                       className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium px-4 py-2 rounded-md transition-colors"
                     >
                       {isSearching ? '...' : '🔍'}
                     </button>
                   </div>
                 </div>

                {/* Search Results or Summary */}
                {searchAction === 'summarize' && searchSummary && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-700">📄 Email Summary</h4>
                    <div className="max-h-60 overflow-y-auto bg-blue-50 border border-blue-200 rounded-md p-3">
                      <div className="text-xs text-gray-800 whitespace-pre-wrap">
                        {searchSummary}
                      </div>
                    </div>
                  </div>
                )}
                
                {searchAction === 'search' && searchResults.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-700">🔍 Search Results ({searchResults.length})</h4>
                    <div className="max-h-60 overflow-y-auto space-y-2">
                      {searchResults.map((result, index) => (
                        <div key={index} className="bg-gray-50 border border-gray-200 rounded-md p-3">
                          <div className="text-xs font-medium text-gray-800 mb-1">
                            {result.sender || 'Unknown Sender'}
                          </div>
                          <div className="text-xs text-gray-700 mb-2">
                            {result.subject || 'No Subject'}
                          </div>
                          {result.similarity && (
                            <div className="text-xs text-blue-600">
                              Similarity: {Math.round(result.similarity * 100)}%
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                 
                 <button
                   onClick={handleLogout}
                   disabled={isLoading}
                   className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors"
                 >
                   {isLoading ? 'Logging out...' : '🚪 Logout'}
                 </button>
               </div>
             )}

      <div className="mt-6 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          ✅ Use the "📧 Summarize" button in Gmail to summarize email threads
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
