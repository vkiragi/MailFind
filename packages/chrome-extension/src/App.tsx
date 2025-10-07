import { useState, useEffect } from 'react'
import './App.css'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { Toaster } from '@/components/ui/toaster'
import InstantSearch from '@/components/InstantSearch'

// Notification types
type NotificationType = 'success' | 'error' | 'info' | 'warning'

// Component to render text with clickable markdown links
const MarkdownText = ({ text }: { text: string }) => {
  const renderTextWithLinks = (text: string) => {
    // Regex to match markdown links: [text](url)
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(text)) !== null) {
      // Add text before the link
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }

      // Add the clickable link
      const linkText = match[1];
      const linkUrl = match[2];
      parts.push(
        <a
          key={match.index}
          href={linkUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-400 hover:text-blue-300 underline cursor-pointer"
          onClick={(e) => {
            e.preventDefault();
            window.open(linkUrl, '_blank');
          }}
        >
          {linkText}
        </a>
      );

      lastIndex = linkRegex.lastIndex;
    }

    // Add remaining text after the last link
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts.length > 0 ? parts : [text];
  };

  return (
    <div className="whitespace-pre-wrap">
      {renderTextWithLinks(text)}
    </div>
  );
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [syncLoading, setSyncLoading] = useState(false)
  const [syncRange, setSyncRange] = useState<'24h' | '7d' | '30d'>('24h')
  const [chatMessages, setChatMessages] = useState<Array<{role: 'user' | 'assistant', content: string, emails?: any[]}>>([])
  const [chatInput, setChatInput] = useState('')
  const [isChatting, setIsChatting] = useState(false)
  const [hasAutoSynced, setHasAutoSynced] = useState(false)
  const [isAutoSyncing, setIsAutoSyncing] = useState(false)
  const [chatHeight, setChatHeight] = useState(240) // Initial height in pixels
  const [activeTab, setActiveTab] = useState<'search' | 'chat' | 'settings' | 'analytics'>('search')

  // Settings state
  const [autoSyncEnabled, setAutoSyncEnabled] = useState(false)
  const [syncFrequency, setSyncFrequency] = useState<'15min' | '30min' | '1hr' | '6hr'>('1hr')
  const [userEmail, setUserEmail] = useState<string>('')
  const [settingsLoading, setSettingsLoading] = useState(false)

  // Toast notifications
  const { toast } = useToast()

  // Notification function using shadcn toast
  const addNotification = (type: NotificationType, message: string) => {
    toast({
      title: type === 'success' ? '✅ Success' : 
             type === 'error' ? '❌ Error' :
             type === 'warning' ? '⚠️ Warning' : 'ℹ️ Info',
      description: message,
      variant: type === 'error' || type === 'warning' ? 'destructive' : 'default',
    })
  }

  // Load user settings
  const loadSettings = async () => {
    try {
      const response = await fetch('http://localhost:8000/settings', {
        credentials: 'include'
      });

      if (response.ok) {
        const settings = await response.json();
        setAutoSyncEnabled(settings.autoSyncEnabled || false);
        setSyncFrequency(settings.syncFrequency || '1hr');
        setUserEmail(settings.userEmail || '');
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  // Save user settings
  const saveSettings = async () => {
    setSettingsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          autoSyncEnabled,
          syncFrequency
        })
      });

      if (response.ok) {
        addNotification('success', 'Settings saved successfully');
      } else {
        addNotification('error', 'Failed to save settings');
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
      addNotification('error', 'Failed to save settings');
    } finally {
      setSettingsLoading(false);
    }
  };

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

        // Load settings if authenticated
        if (wasAuthenticated) {
          await loadSettings();
        }

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

  // Auto-sync when authenticated and haven't synced yet
  useEffect(() => {
    if (isAuthenticated && !hasAutoSynced) {
      // Add a small delay to ensure everything is ready
      const timer = setTimeout(() => {
        autoSyncInbox();
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [isAuthenticated, hasAutoSynced]);

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
            addNotification('error', 'OAuth timed out. Please try again.');
          }
        }, 120000);
        
      } else {
        throw new Error('Popup blocked. Please allow popups for this site.');
      }
    } catch (error) {
      console.error('💥 [OAuth] Login failed:', error);
      addNotification('error', 'Login failed: ' + (error instanceof Error ? error.message : String(error)));
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
      addNotification('error', 'Failed to logout. Please try again.');
    } finally {
      setIsLoading(false);
      console.log('🏁 [Logout] Logout flow completed');
    }
  }

  const performSync = async (range: '24h' | '7d' | '30d', showAlert: boolean = true) => {
    console.log(`📥 [Sync] Starting inbox sync for ${range}...`);
    
    try {
      const response = await fetch('http://localhost:8000/sync-inbox', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ range }),
        credentials: 'include'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ [Sync] Sync successful:', result);
        const skipped = typeof result.skipped_existing === 'number' ? result.skipped_existing : 0;
        
        if (showAlert) {
          addNotification('success', `Indexed ${result.indexed_count} new emails${skipped ? ` (skipped ${skipped} existing)` : ''}.`);
        } else {
          console.log(`✅ [Auto-Sync] Indexed ${result.indexed_count} new emails${skipped ? ` (skipped ${skipped} existing)` : ''}.`);
        }
        return result;
      } else {
        console.error(`❌ [Sync] Backend error: ${response.status} ${response.statusText}`);
        throw new Error(`Failed to sync: ${response.status}`);
      }
    } catch (error) {
      console.error('💥 [Sync] Sync failed:', error);
      if (showAlert) {
        addNotification('error', 'Failed to sync inbox. Please try again.');
      }
      throw error;
    }
  }

  const handleSyncInbox = async () => {
    setSyncLoading(true)
    try {
      await performSync(syncRange, true);
    } finally {
      setSyncLoading(false);
      console.log('🏁 [Sync] Manual sync flow completed');
    }
  }

  const autoSyncInbox = async () => {
    if (hasAutoSynced) return; // Prevent multiple auto-syncs
    
    setIsAutoSyncing(true);
    console.log('🔄 [Auto-Sync] Starting automatic sync for last 24 hours...');
    try {
      await performSync('24h', true); // Show alert for auto-sync so user can see it's working
      setHasAutoSynced(true);
      console.log('🏁 [Auto-Sync] Automatic sync completed');
    } catch (error) {
      console.error('💥 [Auto-Sync] Automatic sync failed:', error);
      // Don't show error alert for auto-sync, just log it
    } finally {
      setIsAutoSyncing(false);
    }
  }

  const handleChat = async () => {
    if (!chatInput.trim()) return;
    
    setIsChatting(true)
    const userMessage = chatInput.trim()
    setChatInput('')
    
    // Add user message to chat
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
    
    console.log('💬 [Chat] Starting chat with:', userMessage);
    
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
        credentials: 'include'
      });
      
      if (response.ok) {
        // Parse JSON response
        const data = await response.json()
        const assistantMessage = data.response || data.error || 'No response'
        const emails = data.emails || []

        // Add assistant message with emails
        setChatMessages(prev => [...prev, { role: 'assistant', content: assistantMessage, emails }])

        console.log('✅ [Chat] Chat successful with', emails.length, 'emails');
      } else {
        console.error(`❌ [Chat] Backend error: ${response.status} ${response.statusText}`);
        throw new Error(`Chat failed: ${response.status}`);
      }
    } catch (error) {
      console.error('💥 [Chat] Chat failed:', error);
      // Add error message to chat
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
    } finally {
      setIsChatting(false);
      console.log('🏁 [Chat] Chat flow completed');
    }
  }


  return (
    <div className="w-full h-full min-h-screen bg-slate-900 text-slate-100 p-4 flex flex-col gap-y-4 font-sans">
      {/* Header Section */}
      <div className="flex flex-col items-center flex-shrink-0">
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
          <Button
            onClick={handleLogin}
            disabled={isLoading}
            className="bg-violet-600 hover:bg-violet-500"
          >
            {isLoading ? 'Connecting...' : 'Connect with Google'}
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-y-3 flex-1 min-h-0">
          {/* Connection Status */}
          <div className="flex items-center justify-center gap-x-2 text-xs text-green-400 bg-green-900/50 rounded-full px-3 py-1 flex-shrink-0">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            Connected to Gmail
          </div>

          {/* Auto-sync Status */}
          {isAutoSyncing && (
            <div className="flex items-center justify-center gap-x-2 text-xs text-blue-400 bg-blue-900/50 rounded-full px-3 py-1 flex-shrink-0">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              Auto-syncing last 24 hours...
            </div>
          )}

          {/* Tab Navigation */}
          <div className="grid grid-cols-4 gap-2 flex-shrink-0">
            <button
              onClick={() => setActiveTab('search')}
              className={`flex flex-col items-center justify-center gap-y-1 py-2 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'search'
                  ? 'bg-violet-600/20 text-violet-400 ring-2 ring-violet-400/50'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <span>Search</span>
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex flex-col items-center justify-center gap-y-1 py-2 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'chat'
                  ? 'bg-violet-600/20 text-violet-400 ring-2 ring-violet-400/50'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <span>Chat</span>
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`flex flex-col items-center justify-center gap-y-1 py-2 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'settings'
                  ? 'bg-violet-600/20 text-violet-400 ring-2 ring-violet-400/50'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span>Settings</span>
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className={`flex flex-col items-center justify-center gap-y-1 py-2 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'analytics'
                  ? 'bg-violet-600/20 text-violet-400 ring-2 ring-violet-400/50'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span>Analytics</span>
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex flex-col gap-y-3 flex-1 overflow-y-auto min-h-0">
            {/* Search Tab */}
            {activeTab === 'search' && (
              <div className="space-y-4">
                {/* Instant Search Section */}
                <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                  <InstantSearch />
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
                    <button onClick={() => setSyncRange('24h')} className={`text-xs font-semibold py-2 rounded-lg transition-colors ${syncRange==='24h' ? 'bg-slate-600 text-slate-100' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
                      24h
                    </button>
                    <button onClick={() => setSyncRange('7d')} className={`text-xs font-semibold py-2 rounded-lg transition-colors ${syncRange==='7d' ? 'bg-slate-600 text-slate-100' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
                      7d
                    </button>
                    <button onClick={() => setSyncRange('30d')} className={`text-xs font-semibold py-2 rounded-lg transition-colors ${syncRange==='30d' ? 'bg-slate-600 text-slate-100' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
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
              </div>
            )}

            {/* Chat Tab */}
            {activeTab === 'chat' && (
              <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl flex flex-col gap-y-3">

            {/* Chat Interface */}
            <>
                {/* Chat Messages */}
                {chatMessages.length > 0 && (
                  <div className="relative">
                    <div
                      className="overflow-y-auto space-y-3 bg-slate-800/50 rounded-lg p-3"
                      style={{ height: `${chatHeight}px` }}
                    >
                    {chatMessages.map((message, index) => (
                      <div key={index}>
                        <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[90%] rounded-lg p-3 text-sm overflow-hidden ${
                            message.role === 'user'
                              ? 'bg-violet-600 text-white'
                              : 'bg-slate-700 text-slate-100'
                          }`}>
                            {message.role === 'user' ? (
                              <div className="whitespace-pre-wrap">{message.content}</div>
                            ) : (
                              <>
                                <MarkdownText text={message.content} />

                                {/* Display email cards inline for assistant messages */}
                                {message.emails && message.emails.length > 0 && (
                                  <div className="mt-3 pt-3 border-t border-slate-600 space-y-2">
                                    <div className="text-xs text-slate-400 mb-2">📧 {message.emails.length} email{message.emails.length > 1 ? 's' : ''} {message.emails.length > 5 ? '(showing top 5)' : ''}</div>
                                    {message.emails.slice(0, 5).map((email: any, emailIndex: number) => (
                                      <a
                                        key={emailIndex}
                                        href={`https://mail.google.com/mail/u/0/#inbox/${email.thread_id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="block bg-slate-800/50 hover:bg-slate-600 border border-slate-600 rounded-lg p-2 transition-colors overflow-hidden cursor-pointer"
                                        style={{ pointerEvents: 'auto' }}
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          window.open(`https://mail.google.com/mail/u/0/#inbox/${email.thread_id}`, '_blank');
                                        }}
                                      >
                                        <div className="text-sm font-medium text-slate-100 truncate">
                                          {email.subject || '(No Subject)'}
                                        </div>
                                        <div className="text-xs text-slate-400 mt-1 truncate">{email.sender}</div>
                                      </a>
                                    ))}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                    {isChatting && (
                      <div className="flex justify-start">
                        <div className="bg-slate-700 text-slate-100 rounded-lg p-3 text-sm">
                          <div className="flex items-center gap-x-2">
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div>
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                          </div>
                        </div>
                      </div>
                    )}
                    </div>
                    {/* Resize Handle */}
                    <div
                      className="absolute bottom-0 left-0 right-0 h-2 cursor-ns-resize hover:bg-violet-500/30 transition-colors"
                      onMouseDown={(e) => {
                        e.preventDefault()
                        const startY = e.clientY
                        const startHeight = chatHeight

                        const handleMouseMove = (e: MouseEvent) => {
                          const delta = e.clientY - startY
                          const newHeight = Math.max(150, Math.min(600, startHeight + delta))
                          setChatHeight(newHeight)
                        }

                        const handleMouseUp = () => {
                          document.removeEventListener('mousemove', handleMouseMove)
                          document.removeEventListener('mouseup', handleMouseUp)
                        }

                        document.addEventListener('mousemove', handleMouseMove)
                        document.addEventListener('mouseup', handleMouseUp)
                      }}
                    >
                      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-12 h-1 bg-slate-600 rounded-full"></div>
                    </div>
                  </div>
                )}

                {/* Example Questions */}
                {chatMessages.length === 0 && (
                  <div className="space-y-2">
                    <div className="text-xs text-slate-400 mb-2">Try asking:</div>
                    <div className="grid grid-cols-1 gap-2">
                      <button
                        onClick={() => setChatInput("What emails did I receive this week about NYT news?")}
                        className="text-left text-xs bg-slate-700 hover:bg-slate-600 rounded-lg p-2 transition-colors"
                      >
                        📰 "What emails did I receive this week about NYT news?"
                      </button>
                      <button
                        onClick={() => setChatInput("Summarize my recent fantasy football emails")}
                        className="text-left text-xs bg-slate-700 hover:bg-slate-600 rounded-lg p-2 transition-colors"
                      >
                        🏈 "Summarize my recent fantasy football emails"
                      </button>
                      <button
                        onClick={() => setChatInput("What important emails did I get today?")}
                        className="text-left text-xs bg-slate-700 hover:bg-slate-600 rounded-lg p-2 transition-colors"
                      >
                        📧 "What important emails did I get today?"
                      </button>
                    </div>
                  </div>
                )}

                {/* Chat Input */}
                <div className="flex gap-x-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleChat()}
                    placeholder="Ask about your emails... (e.g., 'What emails did I get this week about NYT news?')"
                    className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                  <button
                    onClick={handleChat}
                    disabled={isChatting || !chatInput.trim()}
                    className="bg-violet-600 text-white font-bold px-4 py-2 rounded-lg hover:bg-violet-500 transition-colors disabled:bg-slate-600"
                  >
                    {isChatting ? '...' : 'Send'}
                  </button>
                </div>
              </>
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="space-y-3">
                {/* Account Information */}
                <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                  <div className="flex items-center gap-x-2 font-semibold mb-3">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    Account
                  </div>
                  <div className="space-y-2">
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <div className="text-xs text-slate-400 mb-1">Email Address</div>
                      <div className="text-sm text-slate-200">{userEmail || 'Loading...'}</div>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <div className="text-xs text-slate-400 mb-1">Connection Status</div>
                      <div className="flex items-center gap-x-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                        <span className="text-sm text-green-400">Connected to Gmail</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Sync Settings */}
                <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                  <div className="flex items-center gap-x-2 font-semibold mb-3">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Sync Preferences
                  </div>
                  <div className="space-y-3">
                    {/* Auto-sync Toggle */}
                    <div className="flex items-center justify-between bg-slate-700/30 rounded-lg p-3">
                      <div className="flex-1">
                        <div className="text-sm text-slate-200 mb-1">Auto-sync</div>
                        <div className="text-xs text-slate-400">Automatically sync new emails</div>
                      </div>
                      <button
                        onClick={() => setAutoSyncEnabled(!autoSyncEnabled)}
                        className={`relative w-11 h-6 rounded-full transition-colors ${
                          autoSyncEnabled ? 'bg-violet-600' : 'bg-slate-600'
                        }`}
                      >
                        <div
                          className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                            autoSyncEnabled ? 'translate-x-5' : 'translate-x-0'
                          }`}
                        />
                      </button>
                    </div>

                    {/* Sync Frequency */}
                    {autoSyncEnabled && (
                      <div className="bg-slate-700/30 rounded-lg p-3">
                        <div className="text-sm text-slate-200 mb-2">Sync Frequency</div>
                        <div className="grid grid-cols-2 gap-2">
                          <button
                            onClick={() => setSyncFrequency('15min')}
                            className={`text-xs py-2 rounded-lg transition-colors ${
                              syncFrequency === '15min'
                                ? 'bg-violet-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                          >
                            15 min
                          </button>
                          <button
                            onClick={() => setSyncFrequency('30min')}
                            className={`text-xs py-2 rounded-lg transition-colors ${
                              syncFrequency === '30min'
                                ? 'bg-violet-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                          >
                            30 min
                          </button>
                          <button
                            onClick={() => setSyncFrequency('1hr')}
                            className={`text-xs py-2 rounded-lg transition-colors ${
                              syncFrequency === '1hr'
                                ? 'bg-violet-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                          >
                            1 hour
                          </button>
                          <button
                            onClick={() => setSyncFrequency('6hr')}
                            className={`text-xs py-2 rounded-lg transition-colors ${
                              syncFrequency === '6hr'
                                ? 'bg-violet-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                          >
                            6 hours
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Save Settings Button */}
                    <button
                      onClick={saveSettings}
                      disabled={settingsLoading}
                      className="w-full bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium py-2 rounded-lg transition-colors disabled:opacity-50"
                    >
                      {settingsLoading ? 'Saving...' : 'Save Preferences'}
                    </button>
                  </div>
                </div>

                {/* Logout */}
                <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                  <button
                    onClick={handleLogout}
                    disabled={isLoading}
                    className="w-full flex items-center justify-center gap-x-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 text-sm py-2 rounded-lg transition-colors disabled:opacity-50"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                    {isLoading ? 'Logging out...' : 'Logout'}
                  </button>
                </div>
              </div>
            )}

            {/* Analytics Tab */}
            {activeTab === 'analytics' && (
              <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                <div className="space-y-4">
                  <div className="flex items-center gap-x-2 font-semibold mb-3">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Email Analytics
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gradient-to-br from-violet-600/20 to-violet-600/5 rounded-lg p-3 border border-violet-600/30">
                      <div className="text-2xl font-bold text-violet-400">---</div>
                      <div className="text-xs text-slate-400 mt-1">Total Emails</div>
                    </div>

                    <div className="bg-gradient-to-br from-blue-600/20 to-blue-600/5 rounded-lg p-3 border border-blue-600/30">
                      <div className="text-2xl font-bold text-blue-400">---</div>
                      <div className="text-xs text-slate-400 mt-1">This Week</div>
                    </div>

                    <div className="bg-gradient-to-br from-green-600/20 to-green-600/5 rounded-lg p-3 border border-green-600/30">
                      <div className="text-2xl font-bold text-green-400">---</div>
                      <div className="text-xs text-slate-400 mt-1">Important</div>
                    </div>

                    <div className="bg-gradient-to-br from-yellow-600/20 to-yellow-600/5 rounded-lg p-3 border border-yellow-600/30">
                      <div className="text-2xl font-bold text-yellow-400">---</div>
                      <div className="text-xs text-slate-400 mt-1">Top Sender</div>
                    </div>
                  </div>

                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-sm text-slate-400">Coming soon: Email trends, sender statistics, and more insights</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <Toaster />
    </div>
  )
}

export default App
