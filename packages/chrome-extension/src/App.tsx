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
        } else if (result.action === 'answer') {
          // Conversational answer response
          setSearchAction('answer');
          setSearchSummary(result.answer || 'No answer available');
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
    <div className="w-80 p-5 bg-slate-800 min-h-[500px] flex flex-col">
      {/* Header Section */}
      <div className="text-center mb-4">
        <h1 className="text-xl font-bold text-white mb-1">Mailfind</h1>
        <p className="text-xs text-slate-400">AI-Powered Email Intelligence</p>
      </div>

      {!isAuthenticated ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="bg-slate-700 rounded-lg p-6 space-y-4 w-full">
            <p className="text-sm text-slate-300 text-center">
              Connect your Gmail account to start using AI-powered email intelligence
            </p>
            <button
              onClick={handleLogin}
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2.5 px-4 rounded-md transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              {isLoading ? 'Connecting...' : '🔗 Connect with Google'}
            </button>
          </div>
        </div>
                   ) : (
               <div className="flex-1 flex flex-col gap-3">
                 {/* Status Card */}
                 <div className="bg-gradient-to-r from-green-900/30 to-slate-700 rounded-lg p-2.5 border border-green-800/30">
                   <p className="text-xs text-green-400 flex items-center gap-2">
                     <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                     Connected to Gmail
                   </p>
                 </div>
                 
                 {/* Sync Inbox Card */}
                 <div className="bg-slate-700/50 backdrop-blur rounded-lg p-3 space-y-2.5 border border-slate-600/30">
                   <div>
                     <h3 className="text-sm font-semibold text-white">Sync Inbox</h3>
                     <p className="text-xs text-slate-400 mt-0.5">Index your emails for AI search</p>
                   </div>
                   
                   {/* Time-based Indexing Buttons */}
                   <div className="grid grid-cols-3 gap-1.5">
                     <button
                       onClick={() => handleSyncInbox(1)}
                       disabled={syncLoading}
                       className="border border-slate-600 text-slate-300 hover:bg-slate-600 hover:border-slate-500 disabled:opacity-50 font-medium py-1.5 px-2 rounded-md transition-all text-xs"
                     >
                       {syncLoading ? '⏳' : '24h'}
                     </button>
                     <button
                       onClick={() => handleSyncInbox(7)}
                       disabled={syncLoading}
                       className="border border-slate-600 text-slate-300 hover:bg-slate-600 hover:border-slate-500 disabled:opacity-50 font-medium py-1.5 px-2 rounded-md transition-all text-xs"
                     >
                       {syncLoading ? '⏳' : '7d'}
                     </button>
                     <button
                       onClick={() => handleSyncInbox(30)}
                       disabled={syncLoading}
                       className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium py-1.5 px-2 rounded-md transition-all text-xs shadow-sm"
                     >
                       {syncLoading ? '⏳' : '30d'}
                     </button>
                   </div>
                 </div>

                {/* AI Assistant Card */}
                 <div className="bg-slate-700/50 backdrop-blur rounded-lg p-3 space-y-2.5 border border-slate-600/30">
                   <div>
                     <h3 className="text-sm font-semibold text-white">AI Assistant</h3>
                     <p className="text-xs text-slate-400 mt-0.5">Search and analyze your emails</p>
                   </div>
                   <div className="flex space-x-2">
                     <input
                       type="text"
                       value={searchQuery}
                       onChange={(e) => setSearchQuery(e.target.value)}
                       onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                       placeholder="Ask me anything..."
                       className="flex-1 px-3 py-2 bg-slate-800/50 text-white rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 placeholder-slate-500 border border-slate-600/30 focus:border-blue-500/50 transition-all"
                     />
                     <button
                       onClick={handleSearch}
                       disabled={isSearching || !searchQuery.trim()}
                       className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium px-3 py-2 rounded-md transition-all shadow-sm"
                     >
                       {isSearching ? '⏳' : '→'}
                     </button>
                   </div>
                   
                   {/* Example queries */}
                   <div className="text-xs text-slate-500 italic">
                     Try: "DoorDash orders" or "Summarize today"
                   </div>
                 </div>

               {/* Search Results, Summary, or Conversational Answer */}
                {searchAction === 'summarize' && searchSummary && (
                  <div className="bg-slate-700/50 backdrop-blur rounded-lg p-3 space-y-2 border border-slate-600/30">
                    <h4 className="text-sm font-semibold text-white">Summary</h4>
                    <div className="max-h-40 overflow-y-auto bg-slate-800/30 rounded-md p-2.5">
                      <div className="text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                        {searchSummary}
                      </div>
                    </div>
                  </div>
                )}
                
                {searchAction === 'answer' && searchSummary && (
                  <div className="bg-slate-700/50 backdrop-blur rounded-lg p-3 space-y-2 border border-slate-600/30">
                    <h4 className="text-sm font-semibold text-white">Answer</h4>
                    <div className="max-h-40 overflow-y-auto bg-slate-800/30 rounded-md p-2.5">
                      <div className="text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                        {searchSummary}
                      </div>
                    </div>
                  </div>
                )}
                
                {searchAction === 'search' && searchResults.length > 0 && (
                  <div className="bg-slate-700/50 backdrop-blur rounded-lg p-3 space-y-2 border border-slate-600/30">
                    <h4 className="text-sm font-semibold text-white">Results ({searchResults.length})</h4>
                    <div className="max-h-40 overflow-y-auto space-y-1.5">
                      {searchResults.map((result, index) => (
                        <div key={index} className="bg-slate-800/30 rounded-md p-2 hover:bg-slate-800/50 transition-colors cursor-pointer">
                          <div className="text-xs font-medium text-slate-200 truncate">
                            {result.sender || 'Unknown'}
                          </div>
                          <div className="text-xs text-slate-400 truncate">
                            {result.subject || 'No Subject'}
                          </div>
                          {result.similarity && (
                            <div className="text-xs text-blue-400 mt-1">
                              {Math.round(result.similarity * 100)}% match
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                 
                 {/* Spacer to push logout to bottom */}
                 <div className="flex-1"></div>
                 
                 {/* Logout Button - Small and Subtle */}
                 <button
                   onClick={handleLogout}
                   disabled={isLoading}
                   className="mx-auto flex items-center justify-center px-3 py-1.5 rounded-md text-xs text-slate-500 hover:text-red-400 hover:bg-slate-700/50 transition-all duration-200"
                 >
                   <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                   </svg>
                   {isLoading ? 'Logging out...' : 'Sign out'}
                 </button>
               </div>
             )}

      {/* Footer Info */}
      <div className="mt-auto pt-3 border-t border-slate-700/50">
        <p className="text-xs text-slate-500 text-center">
          Use the Summarize button in Gmail
        </p>
        
        {authStatus && authStatus.authenticated_users > 0 && (
          <div className="mt-2">
            <p className="text-xs text-slate-600 text-center">
              • Active Session
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App