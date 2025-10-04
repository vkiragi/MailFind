import { useState, useEffect } from 'react'
import './App.css'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { Toaster } from '@/components/ui/toaster'

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
    <div className="w-full min-h-screen bg-slate-900 text-slate-100 p-6 flex flex-col gap-y-5 font-sans relative">
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
          <Button
            onClick={handleLogin}
            disabled={isLoading}
            className="bg-violet-600 hover:bg-violet-500"
          >
            {isLoading ? 'Connecting...' : 'Connect with Google'}
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-y-5">
          {/* Connection Status */}
          <div className="flex items-center justify-center gap-x-2 text-sm text-green-400 bg-green-900/50 rounded-full px-3 py-1">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            Connected to Gmail
          </div>
          
          {/* Auto-sync Status */}
          {isAutoSyncing && (
            <div className="flex items-center justify-center gap-x-2 text-sm text-blue-400 bg-blue-900/50 rounded-full px-3 py-1">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              Auto-syncing last 24 hours...
            </div>
          )}
          
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

          {/* AI Assistant Section */}
          <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl flex flex-col gap-y-3">
            <div className="flex items-center gap-x-2 font-semibold">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              AI Email Chat
            </div>

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
                                    <div className="text-xs text-slate-400 mb-2">📧 {message.emails.length} email{message.emails.length > 1 ? 's' : ''}</div>
                                    {message.emails.map((email: any, emailIndex: number) => (
                                      <a
                                        key={emailIndex}
                                        href={`https://mail.google.com/mail/u/0/#inbox/${email.thread_id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="block bg-slate-800/50 hover:bg-slate-600 border border-slate-600 rounded-lg p-2 transition-colors overflow-hidden"
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
                    onKeyPress={(e) => e.key === 'Enter' && handleChat()}
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
          
          {/* Toast notifications - positioned below logout */}
          <div className="relative">
            <Toaster />
          </div>
        </div>
      )}
    </div>
  )
}

export default App
