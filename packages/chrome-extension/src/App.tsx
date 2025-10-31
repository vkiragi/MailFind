import { useState, useEffect } from 'react'
import './App.css'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { Toaster } from '@/components/ui/toaster'
import InstantSearch from '@/components/InstantSearch'
import { PasswordSetup } from '@/components/PasswordSetup'
import { UnlockModal } from '@/components/UnlockModal'
import {
  isPasswordProtected,
  isUnlocked,
  unlock,
  lock,
  getEncryptionKeyAsBase64,
  setupPasswordProtection,
  hasLegacyEncryption,
  migrateLegacyKey,
  getPasswordHint,
  resetAutoLock
} from '@/utils/security'

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

  // Security state
  const [isLocked, setIsLocked] = useState(true)
  const [showPasswordSetup, setShowPasswordSetup] = useState(false)
  const [showUnlockModal, setShowUnlockModal] = useState(false)
  const [passwordHint, setPasswordHint] = useState<string | null>(null)
  const [isMigration, setIsMigration] = useState(false)

  // Analytics state
  const [analyticsData, setAnalyticsData] = useState<{
    totalEmails: number
    weeklyEmails: number
    importantEmails: number
    topSenders: Array<{sender: string, count: number}>
    dailyVolume: Array<{date: string, count: number}>
    categoryBreakdown: Array<{category: string, count: number}>
    emailsWithAttachments: number
    inboxHealthScore: number
    noiseRatio: {
      personal: number
      automated: number
      personalPercentage: number
      automatedPercentage: number
    }
    unsubscribeCandidates: Array<{sender: string, count: number}>
    vipSenders: Array<{sender: string, count: number}>
    timeWasters: Array<{sender: string, count: number}>
    peakHour: {hour: number, count: number, label: string}
    hourlyDistribution: Array<{hour: number, count: number}>
    busiestDay: {day: string, count: number, dayNum: number}
    dayOfWeekDistribution: Array<{day: string, dayNum: number, count: number}>
    weekOverWeek: {lastWeek: number, thisWeek: number, change: number, percentageChange: number, trend: string}
  } | null>(null)
  const [analyticsLoading, setAnalyticsLoading] = useState(false)

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

  // Check security status
  const checkSecurityStatus = async () => {
    console.log('🔐 [Security] Checking security status...');

    try {
      // Check if password protection is set up
      const hasPassword = await isPasswordProtected();

      if (!hasPassword) {
        // Check for legacy encryption
        const hasLegacy = await hasLegacyEncryption();

        if (hasLegacy) {
          console.log('🔄 [Security] Legacy encryption found, showing migration flow');
          setIsMigration(true);
          setShowPasswordSetup(true);
          setIsLocked(true);
        } else {
          console.log('🆕 [Security] No encryption found, showing initial setup');
          setIsMigration(false);
          setShowPasswordSetup(true);
          setIsLocked(true);
        }
      } else {
        // Password protection exists, check if unlocked
        const unlocked = isUnlocked();

        if (!unlocked) {
          console.log('🔒 [Security] Locked, showing unlock modal');
          setIsLocked(true);
          setShowUnlockModal(true);

          // Load password hint
          const hint = await getPasswordHint();
          setPasswordHint(hint);
        } else {
          console.log('🔓 [Security] Already unlocked');
          setIsLocked(false);
          setShowUnlockModal(false);

          // Initialize auto-lock
          await resetAutoLock();
        }
      }
    } catch (error) {
      console.error('❌ [Security] Failed to check security status:', error);
    }
  };

  // Load analytics data
  const loadAnalytics = async () => {
    setAnalyticsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analytics', {
        credentials: 'include'
      });

      if (response.ok) {
        const data = await response.json();
        setAnalyticsData(data);
      } else {
        console.error('Failed to load analytics:', response.status);
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setAnalyticsLoading(false);
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

  // Security handlers
  const handlePasswordSetup = async (password: string, hint?: string) => {
    try {
      if (isMigration) {
        // Migrate legacy key
        const success = await migrateLegacyKey(password, hint);
        if (success) {
          console.log('✅ [Security] Legacy key migrated successfully');
          setShowPasswordSetup(false);
          setIsLocked(false);
          setIsMigration(false);

          // Initialize auto-lock
          await resetAutoLock();

          addNotification('success', 'Password protection enabled successfully');
        } else {
          console.error('❌ [Security] Migration failed');
          addNotification('error', 'Failed to migrate encryption key');
        }
      } else {
        // Initial setup
        await setupPasswordProtection(password, hint);
        console.log('✅ [Security] Password protection setup successfully');
        setShowPasswordSetup(false);
        setIsLocked(false);

        // Initialize auto-lock
        await resetAutoLock();

        addNotification('success', 'Password protection enabled successfully');
      }
    } catch (error) {
      console.error('❌ [Security] Password setup failed:', error);
      addNotification('error', 'Failed to setup password protection');
      // Don't rethrow - let the PasswordSetup component handle the error
    }
  };

  const handleUnlock = async (password: string): Promise<boolean> => {
    try {
      const success = await unlock(password);
      if (success) {
        console.log('✅ [Security] Unlocked successfully');
        setIsLocked(false);
        setShowUnlockModal(false);

        // Initialize auto-lock
        await resetAutoLock();

        addNotification('success', 'Unlocked successfully');
        return true;
      } else {
        console.log('❌ [Security] Unlock failed - incorrect password');
        return false;
      }
    } catch (error) {
      console.error('❌ [Security] Unlock error:', error);
      return false;
    }
  };

  const handleLock = () => {
    lock();
    setIsLocked(true);
    setShowUnlockModal(true);
    addNotification('info', 'MailFind has been locked');
  };

  // Check auth status when component mounts
  useEffect(() => {
    checkAuthStatus();
    checkSecurityStatus();
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

  // Load analytics when analytics tab is selected
  useEffect(() => {
    if (isAuthenticated && activeTab === 'analytics') {
      loadAnalytics();
    }
  }, [isAuthenticated, activeTab]);

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
      // Get encryption key from security module (zero-knowledge mode)
      let encryptionKey: string | null = null
      try {
        encryptionKey = await getEncryptionKeyAsBase64()
        if (encryptionKey) {
          console.log('🔐 [Sync] Encryption enabled - using zero-knowledge mode')
          console.log(`🔐 [Sync] Key length: ${encryptionKey.length} bytes`)

          // Reset auto-lock on activity
          await resetAutoLock()
        } else {
          console.warn('⚠️ [Sync] No encryption key available (locked)')
        }
      } catch (error) {
        console.warn('⚠️ [Sync] Failed to get encryption key:', error)
        encryptionKey = null
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }

      if (encryptionKey) {
        headers['X-Encryption-Key'] = encryptionKey
      }

      const response = await fetch('http://localhost:8000/sync-inbox', {
        method: 'POST',
        headers,
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

    console.log('💬 [Chat] Starting streaming chat with:', userMessage);

    try {
      // Get encryption key from security module for decryption
      let encryptionKey: string | null = null
      try {
        encryptionKey = await getEncryptionKeyAsBase64()
        if (encryptionKey) {
          console.log('🔐 [Chat] Encryption key retrieved for decryption')

          // Reset auto-lock on activity
          await resetAutoLock()
        } else {
          console.warn('⚠️ [Chat] No encryption key available (locked)')
        }
      } catch (error) {
        console.warn('⚠️ [Chat] Failed to get encryption key:', error)
        encryptionKey = null
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }

      if (encryptionKey) {
        headers['X-Encryption-Key'] = encryptionKey
      }

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers,
        body: JSON.stringify({ message: userMessage }),
        credentials: 'include'
      });

      if (response.ok && response.body) {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let streamedContent = ''
        let emails: any[] = []
        let assistantMessageIndex = -1

        // Add an empty assistant message that we'll update as we stream
        setChatMessages(prev => {
          const newMessages: Array<{role: 'user' | 'assistant', content: string, emails?: any[]}> = [...prev, { role: 'assistant' as const, content: '', emails: [] }]
          assistantMessageIndex = newMessages.length - 1
          return newMessages
        })

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)

              if (data === '[DONE]') {
                console.log('✅ [Chat] Streaming complete');
                break
              }

              try {
                const parsed = JSON.parse(data)

                if (parsed.emails) {
                  // First message contains the emails
                  emails = parsed.emails
                  console.log('📧 [Chat] Received', emails.length, 'emails');
                } else if (parsed.content) {
                  // Stream content chunks
                  streamedContent += parsed.content

                  // Update the message in real-time
                  setChatMessages(prev => {
                    const updated = [...prev]
                    if (assistantMessageIndex >= 0 && assistantMessageIndex < updated.length) {
                      updated[assistantMessageIndex] = {
                        role: 'assistant',
                        content: streamedContent,
                        emails: emails
                      }
                    }
                    return updated
                  })
                } else if (parsed.error) {
                  console.error('❌ [Chat] Stream error:', parsed.error)
                  throw new Error(parsed.error)
                }
              } catch (e) {
                console.warn('⚠️ [Chat] Failed to parse stream data:', data)
              }
            }
          }
        }

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
                                        onClick={(e) => {
                                          e.stopPropagation();
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

                {/* Security Actions */}
                <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                  <div className="flex items-center gap-x-2 font-semibold mb-3">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                    Security
                  </div>
                  <div className="space-y-2">
                    <button
                      onClick={handleLock}
                      disabled={isLocked}
                      className="w-full flex items-center justify-center gap-x-2 bg-violet-600/20 hover:bg-violet-600/30 text-violet-400 text-sm py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                      {isLocked ? 'Locked' : 'Lock MailFind'}
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
              <div className="space-y-3">
                {analyticsLoading ? (
                  <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl text-center">
                    <div className="flex items-center justify-center gap-2 text-slate-400">
                      <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                    <div className="text-sm text-slate-400 mt-2">Loading analytics...</div>
                  </div>
                ) : analyticsData ? (
                  <>
                    {/* Key Metrics Cards */}
                    <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-x-2 font-semibold">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                          Email Analytics
                        </div>
                        <button
                          onClick={loadAnalytics}
                          className="text-xs text-slate-400 hover:text-violet-400 transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                        </button>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-gradient-to-br from-violet-600/20 to-violet-600/5 rounded-lg p-3 border border-violet-600/30">
                          <div className="text-2xl font-bold text-violet-400">{analyticsData.totalEmails.toLocaleString()}</div>
                          <div className="text-xs text-slate-400 mt-1">Total Emails</div>
                        </div>

                        <div className="bg-gradient-to-br from-blue-600/20 to-blue-600/5 rounded-lg p-3 border border-blue-600/30">
                          <div className="text-2xl font-bold text-blue-400">{analyticsData.weeklyEmails.toLocaleString()}</div>
                          <div className="text-xs text-slate-400 mt-1">This Week</div>
                        </div>

                        <div className="bg-gradient-to-br from-green-600/20 to-green-600/5 rounded-lg p-3 border border-green-600/30">
                          <div className="text-2xl font-bold text-green-400">{analyticsData.importantEmails.toLocaleString()}</div>
                          <div className="text-xs text-slate-400 mt-1">Important</div>
                        </div>

                        <div className="bg-gradient-to-br from-amber-600/20 to-amber-600/5 rounded-lg p-3 border border-amber-600/30">
                          <div className="text-2xl font-bold text-amber-400">{analyticsData.emailsWithAttachments.toLocaleString()}</div>
                          <div className="text-xs text-slate-400 mt-1">Automated</div>
                        </div>
                      </div>
                    </div>

                    {/* Inbox Health Score */}
                    <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                      <div className="flex items-center gap-x-2 font-semibold mb-3">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Inbox Health Score
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="relative w-32 h-32">
                          <svg className="w-full h-full transform -rotate-90">
                            <circle
                              cx="64"
                              cy="64"
                              r="56"
                              stroke="currentColor"
                              strokeWidth="8"
                              fill="none"
                              className="text-slate-700"
                            />
                            <circle
                              cx="64"
                              cy="64"
                              r="56"
                              stroke="currentColor"
                              strokeWidth="8"
                              fill="none"
                              strokeDasharray={`${2 * Math.PI * 56}`}
                              strokeDashoffset={`${2 * Math.PI * 56 * (1 - analyticsData.inboxHealthScore / 100)}`}
                              className={`transition-all ${
                                analyticsData.inboxHealthScore >= 70 ? 'text-green-500' :
                                analyticsData.inboxHealthScore >= 40 ? 'text-yellow-500' :
                                'text-red-500'
                              }`}
                              strokeLinecap="round"
                            />
                          </svg>
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-center">
                              <div className={`text-3xl font-bold ${
                                analyticsData.inboxHealthScore >= 70 ? 'text-green-400' :
                                analyticsData.inboxHealthScore >= 40 ? 'text-yellow-400' :
                                'text-red-400'
                              }`}>
                                {analyticsData.inboxHealthScore}
                              </div>
                              <div className="text-[10px] text-slate-400">/ 100</div>
                            </div>
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="text-sm text-slate-300 mb-2">
                            {analyticsData.inboxHealthScore >= 70 && "🎉 Excellent! Your inbox is well-managed."}
                            {analyticsData.inboxHealthScore >= 40 && analyticsData.inboxHealthScore < 70 && "⚠️ Good, but room for improvement."}
                            {analyticsData.inboxHealthScore < 40 && "📉 Your inbox needs attention."}
                          </div>
                          <div className="text-xs text-slate-400 space-y-1">
                            <div>• {analyticsData.noiseRatio.personalPercentage}% personal emails</div>
                            <div>• {((analyticsData.importantEmails / analyticsData.totalEmails) * 100).toFixed(1)}% important</div>
                            {analyticsData.unsubscribeCandidates.length > 0 && (
                              <div>• {analyticsData.unsubscribeCandidates.length} cleanup opportunities</div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Noise Ratio */}
                    <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                      <div className="flex items-center gap-x-2 font-semibold mb-3">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                        </svg>
                        Personal vs Automated
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="flex-1">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-violet-400">Personal</span>
                            <span className="text-slate-400 font-mono">{analyticsData.noiseRatio.personal} ({analyticsData.noiseRatio.personalPercentage}%)</span>
                          </div>
                          <div className="w-full bg-slate-700 rounded-full h-2 mb-3">
                            <div
                              className="bg-violet-500 h-2 rounded-full transition-all"
                              style={{ width: `${analyticsData.noiseRatio.personalPercentage}%` }}
                            ></div>
                          </div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-amber-400">Automated</span>
                            <span className="text-slate-400 font-mono">{analyticsData.noiseRatio.automated} ({analyticsData.noiseRatio.automatedPercentage}%)</span>
                          </div>
                          <div className="w-full bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-amber-500 h-2 rounded-full transition-all"
                              style={{ width: `${analyticsData.noiseRatio.automatedPercentage}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Unsubscribe Candidates */}
                    {analyticsData.unsubscribeCandidates.length > 0 && (
                      <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                        <div className="flex items-center gap-x-2 font-semibold mb-3">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                          Cleanup Opportunities
                        </div>
                        <div className="text-xs text-slate-400 mb-2">
                          These senders send frequent automated emails with unsubscribe links:
                        </div>
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                          {analyticsData.unsubscribeCandidates.slice(0, 5).map((candidate, index) => (
                            <div key={index} className="flex items-center justify-between bg-slate-700/30 rounded-lg p-2">
                              <div className="flex-1 min-w-0">
                                <div className="text-xs text-slate-300 truncate">{candidate.sender}</div>
                                <div className="text-[10px] text-slate-500">{candidate.count} emails</div>
                              </div>
                              <button className="text-[10px] bg-red-600/20 hover:bg-red-600/30 text-red-400 px-2 py-1 rounded transition-colors">
                                Unsubscribe
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Smart Sender Insights */}
                    {(analyticsData.vipSenders.length > 0 || analyticsData.timeWasters.length > 0) && (
                      <div className="grid grid-cols-2 gap-3">
                        {/* VIP Senders */}
                        {analyticsData.vipSenders.length > 0 && (
                          <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                            <div className="flex items-center gap-x-2 font-semibold mb-2 text-sm">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                              </svg>
                              <span className="text-xs">VIP Senders</span>
                            </div>
                            <div className="space-y-1">
                              {analyticsData.vipSenders.slice(0, 3).map((vip, index) => (
                                <div key={index} className="text-[10px] text-slate-400 truncate">
                                  • {vip.sender} ({vip.count})
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Time Wasters */}
                        {analyticsData.timeWasters.length > 0 && (
                          <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                            <div className="flex items-center gap-x-2 font-semibold mb-2 text-sm">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              <span className="text-xs">High Volume</span>
                            </div>
                            <div className="space-y-1">
                              {analyticsData.timeWasters.slice(0, 3).map((waster, index) => (
                                <div key={index} className="text-[10px] text-slate-400 truncate">
                                  • {waster.sender} ({waster.count})
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Top Senders */}
                    {analyticsData.topSenders.length > 0 && (
                      <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                        <div className="flex items-center gap-x-2 font-semibold mb-3">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                          </svg>
                          Top Senders
                        </div>
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                          {analyticsData.topSenders.map((sender, index) => {
                            const maxCount = analyticsData.topSenders[0].count
                            const percentage = (sender.count / maxCount) * 100
                            return (
                              <div key={index} className="flex items-center gap-2">
                                <div className="flex-1 min-w-0">
                                  <div className="text-xs text-slate-300 truncate">{sender.sender}</div>
                                  <div className="w-full bg-slate-700 rounded-full h-1.5 mt-1">
                                    <div
                                      className="bg-violet-500 h-1.5 rounded-full transition-all"
                                      style={{ width: `${percentage}%` }}
                                    ></div>
                                  </div>
                                </div>
                                <div className="text-xs text-slate-400 font-mono">{sender.count}</div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* Daily Volume Chart */}
                    {analyticsData.dailyVolume.length > 0 && (
                      <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                        <div className="flex items-center gap-x-2 font-semibold mb-3">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                          </svg>
                          Last 7 Days
                        </div>
                        <div className="flex items-end justify-between gap-1 h-32">
                          {analyticsData.dailyVolume.map((day, index) => {
                            const maxCount = Math.max(...analyticsData.dailyVolume.map(d => d.count))
                            const height = maxCount > 0 ? (day.count / maxCount) * 100 : 0
                            const date = new Date(day.date)
                            const dayName = date.toLocaleDateString('en-US', { weekday: 'short' })
                            return (
                              <div key={index} className="flex-1 flex flex-col items-center gap-1">
                                <div className="w-full flex items-end justify-center h-24">
                                  <div
                                    className="w-full bg-gradient-to-t from-violet-600 to-violet-400 rounded-t transition-all hover:from-violet-500 hover:to-violet-300"
                                    style={{ height: `${height}%`, minHeight: day.count > 0 ? '4px' : '0' }}
                                    title={`${day.count} emails on ${day.date}`}
                                  ></div>
                                </div>
                                <div className="text-[10px] text-slate-400">{dayName}</div>
                                <div className="text-[9px] text-slate-500 font-mono">{day.count}</div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* Email Patterns & Trends */}
                    <div className="grid grid-cols-2 gap-3">
                      {/* Peak Hour */}
                      <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                        <div className="flex items-center gap-x-2 font-semibold mb-2 text-sm">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                          </svg>
                          <span className="text-xs">Peak Hour</span>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-400">{analyticsData.peakHour.label}</div>
                          <div className="text-[10px] text-slate-400 mt-1">{analyticsData.peakHour.count} emails</div>
                        </div>
                      </div>

                      {/* Busiest Day */}
                      <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                        <div className="flex items-center gap-x-2 font-semibold mb-2 text-sm">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                          <span className="text-xs">Busiest Day</span>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-400">{analyticsData.busiestDay.day}</div>
                          <div className="text-[10px] text-slate-400 mt-1">{analyticsData.busiestDay.count} emails</div>
                        </div>
                      </div>
                    </div>

                    {/* Week-over-Week Trend */}
                    <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                      <div className="flex items-center gap-x-2 font-semibold mb-3">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                        </svg>
                        Week-over-Week Trend
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="text-xs text-slate-400 mb-1">Last Week</div>
                          <div className="text-lg font-bold text-slate-300">{analyticsData.weekOverWeek.lastWeek}</div>
                        </div>
                        <div className="flex items-center gap-2">
                          {analyticsData.weekOverWeek.trend === 'up' && (
                            <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                            </svg>
                          )}
                          {analyticsData.weekOverWeek.trend === 'down' && (
                            <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                            </svg>
                          )}
                          {analyticsData.weekOverWeek.trend === 'stable' && (
                            <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
                            </svg>
                          )}
                          <div className={`text-sm font-bold ${
                            analyticsData.weekOverWeek.trend === 'up' ? 'text-red-400' :
                            analyticsData.weekOverWeek.trend === 'down' ? 'text-green-400' :
                            'text-slate-400'
                          }`}>
                            {analyticsData.weekOverWeek.percentageChange > 0 ? '+' : ''}{analyticsData.weekOverWeek.percentageChange}%
                          </div>
                        </div>
                        <div className="flex-1 text-right">
                          <div className="text-xs text-slate-400 mb-1">This Week</div>
                          <div className="text-lg font-bold text-slate-300">{analyticsData.weekOverWeek.thisWeek}</div>
                        </div>
                      </div>
                      <div className="text-[10px] text-slate-500 mt-2 text-center">
                        {analyticsData.weekOverWeek.trend === 'up' && `${analyticsData.weekOverWeek.change} more emails than last week`}
                        {analyticsData.weekOverWeek.trend === 'down' && `${Math.abs(analyticsData.weekOverWeek.change)} fewer emails than last week`}
                        {analyticsData.weekOverWeek.trend === 'stable' && 'Same volume as last week'}
                      </div>
                    </div>

                    {/* Category Breakdown */}
                    {analyticsData.categoryBreakdown.length > 0 && (
                      <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl">
                        <div className="flex items-center gap-x-2 font-semibold mb-3">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                          </svg>
                          Categories
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          {analyticsData.categoryBreakdown.slice(0, 6).map((cat, index) => (
                            <div key={index} className="bg-slate-700/30 rounded-lg p-2">
                              <div className="text-xs text-slate-300 truncate capitalize">{cat.category}</div>
                              <div className="text-lg font-bold text-violet-400 mt-1">{cat.count}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl text-center">
                    <div className="text-sm text-slate-400">No analytics data available. Sync your inbox to see analytics.</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Password Setup Modal */}
      {showPasswordSetup && (
        <PasswordSetup
          onSetup={handlePasswordSetup}
          isMigration={isMigration}
        />
      )}

      {/* Unlock Modal */}
      {showUnlockModal && (
        <UnlockModal
          onUnlock={handleUnlock}
          passwordHint={passwordHint}
        />
      )}

      <Toaster />
    </div>
  )
}

export default App
