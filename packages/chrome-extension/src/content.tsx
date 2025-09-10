// Content script that runs on Gmail pages
console.log('MailFind content script loaded');

// Gmail DOM selectors - updated with real Gmail DOM elements
// List view toolbar often uses gh="tm"; thread view often uses gh="mtb".
// We also include generic ARIA-based selectors to be resilient across layouts/locales.
const GMAIL_LIST_TOOLBAR_SELECTOR = 'div[gh="tm"]';
const GMAIL_THREAD_TOOLBAR_SELECTOR = 'div[gh="mtb"]';
const GMAIL_ARIA_TOOLBAR_SELECTOR = '[aria-label="Toolbar"], [role="toolbar"]';
// const GMAIL_EMAIL_VIEW_SELECTOR = '.aeH'; // Previously used; not needed now

const TOOLBAR_CANDIDATES = [
  GMAIL_THREAD_TOOLBAR_SELECTOR,
  GMAIL_LIST_TOOLBAR_SELECTOR,
  GMAIL_ARIA_TOOLBAR_SELECTOR,
].join(', ');

// Lightweight toast utilities for non-blocking notifications
function showToast(message: string) {
  let toast = document.getElementById('mailfind-toast') as HTMLDivElement | null;
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'mailfind-toast';
    toast.style.cssText = `
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 2147483647;
      background: rgba(32,33,36,0.98);
      color: #fff;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 12px;
      box-shadow: 0 6px 16px rgba(0,0,0,0.25);
      max-width: 360px;
      line-height: 1.4;
      pointer-events: none;
    `;
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  return toast;
}

function updateToast(toast: HTMLElement, message: string) {
  if (toast) toast.textContent = message;
}

function hideToast(toast: HTMLElement, delayMs = 1200) {
  if (!toast) return;
  setTimeout(() => { toast.remove(); }, delayMs);
}

// Extract visible email content directly from Gmail's DOM
function extractVisibleEmailContent(): string {
  console.log('🔍 [Gmail] Extracting visible email content...');
  
  // Gmail conversation selectors
  const conversationSelectors = [
    '[role="main"] .ii.gt', // Gmail conversation messages
    '.nH.if .ii.gt', // Alternative conversation container
    '.adn.ads .ii.gt', // Thread container messages
    '[role="main"] .adn', // Main conversation area
    '.nH.if .adn', // Alternative conversation area
    '[data-message-id]', // Elements with message IDs
  ];
  
  let extractedContent = '';
  
  for (const selector of conversationSelectors) {
    const elements = document.querySelectorAll(selector);
    console.log(`🔍 [Gmail] Found ${elements.length} elements for selector: ${selector}`);
    
    for (const element of elements) {
      const rect = element.getBoundingClientRect();
      const isVisible = rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top < window.innerHeight;
      
      if (isVisible) {
        const text = element.textContent?.trim();
        if (text && text.length > 50) {
          extractedContent += text + '\n\n';
          console.log(`📄 [Gmail] Added content from visible element: ${text.substring(0, 100)}...`);
        }
      }
    }
    
    // If we found content, break early
    if (extractedContent.length > 200) {
      console.log(`✅ [Gmail] Extracted sufficient content (${extractedContent.length} chars) from selector: ${selector}`);
      break;
    }
  }
  
  // Fallback: Get any visible text from the main area
  if (extractedContent.length < 100) {
    console.log('🔍 [Gmail] Trying fallback content extraction...');
    const mainArea = document.querySelector('[role="main"]') || document.querySelector('.nH.if');
    
    if (mainArea) {
      const text = mainArea.textContent?.trim();
      if (text && text.length > 100) {
        // Clean up the text - remove Gmail UI elements
        const cleanText = text
          .replace(/Archive|Delete|Mark as spam|Move to|Labels|More/g, '')
          .replace(/\s+/g, ' ')
          .trim();
        
        if (cleanText.length > 100) {
          extractedContent = cleanText;
          console.log(`📄 [Gmail] Using fallback content: ${cleanText.substring(0, 100)}...`);
        }
      }
    }
  }
  
  console.log(`📊 [Gmail] Total extracted content length: ${extractedContent.length} characters`);
  return extractedContent.trim();
}

// Inline summary card renderer
function renderInlineSummary(summaryText: string): HTMLDivElement {
  const existing = document.getElementById('mailfind-summary-card');
  if (existing) existing.remove();

  const card = document.createElement('div');
  card.id = 'mailfind-summary-card';
  card.style.cssText = `
    position: fixed;
    top: 72px;
    right: 16px;
    z-index: 2147483647;
    width: 420px;
    max-height: 70vh;
    overflow: auto;
    background: #fff;
    color: #202124;
    border-radius: 12px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.2);
    border: 1px solid rgba(0,0,0,0.08);
    padding: 16px;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
  `;

  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;margin-bottom:8px;gap:6px;';
  const title = document.createElement('div');
  title.textContent = 'MailFind Summary';
  title.style.cssText = 'font-weight:600;font-size:14px;flex:1;';

  // Minimize toggle button
  const minimizeBtn = document.createElement('button');
  minimizeBtn.textContent = '–';
  minimizeBtn.setAttribute('aria-label', 'Minimize summary');
  minimizeBtn.title = 'Minimize';
  minimizeBtn.style.cssText = 'border:none;background:transparent;color:#5f6368;cursor:pointer;font-size:16px;line-height:16px;padding:0 4px;';

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.setAttribute('aria-label', 'Close summary');
  closeBtn.title = 'Close';
  closeBtn.style.cssText = 'border:none;background:transparent;color:#5f6368;cursor:pointer;font-size:18px;line-height:18px;padding:0 4px;';
  closeBtn.addEventListener('click', () => card.remove());

  header.appendChild(title);
  header.appendChild(minimizeBtn);
  header.appendChild(closeBtn);
  const body = document.createElement('div');
  body.textContent = summaryText;
  card.appendChild(header);
  card.appendChild(body);
  document.body.appendChild(card);

  // Minimize/expand behavior
  let collapsed = false;
  const applyCollapsed = () => {
    if (collapsed) {
      body.style.display = 'none';
      card.style.width = '260px';
      card.style.maxHeight = 'unset';
      title.textContent = 'MailFind Summary (minimized)';
      minimizeBtn.textContent = '+';
      minimizeBtn.title = 'Expand';
      minimizeBtn.setAttribute('aria-label', 'Expand summary');
    } else {
      body.style.display = '';
      card.style.width = '420px';
      card.style.maxHeight = '70vh';
      title.textContent = 'MailFind Summary';
      minimizeBtn.textContent = '–';
      minimizeBtn.title = 'Minimize';
      minimizeBtn.setAttribute('aria-label', 'Minimize summary');
    }
  };
  minimizeBtn.addEventListener('click', () => { collapsed = !collapsed; applyCollapsed(); });
  applyCollapsed();
  return body as HTMLDivElement;
}

function isVisible(element: Element | null): boolean {
  if (!element) return false;
  const rect = (element as HTMLElement).getBoundingClientRect();
  const style = window.getComputedStyle(element as HTMLElement);
  return (
    rect.width > 0 &&
    rect.height > 0 &&
    style.visibility !== 'hidden' &&
    style.display !== 'none'
  );
}

function findToolbar(): Element | null {
  const candidates = Array.from(document.querySelectorAll(TOOLBAR_CANDIDATES));
  
  // Prefer visible candidates
  const visible = candidates.filter(isVisible);

  // Prefer thread toolbar gh="mtb" when present
  const threadFirst = visible.sort((a, b) => {
    const aIsThread = (a as HTMLElement).getAttribute('gh') === 'mtb' ? 1 : 0;
    const bIsThread = (b as HTMLElement).getAttribute('gh') === 'mtb' ? 1 : 0;
    return bIsThread - aIsThread;
  });

  const chosen = threadFirst[0] || null;
  return chosen;
}

// Global flag to prevent multiple rapid injections
let isInjecting = false;
let lastInjectedToolbar: Element | null = null;

// Function to inject the summarize button into Gmail
function injectSummarizeButton() {
  if (isInjecting) return; // Prevent concurrent injections
  
  const existing = document.getElementById('mailfind-summarize-btn');
  const toolbar = findToolbar();

  if (!toolbar) {
    return;
  }

  // If button exists in the current toolbar and toolbar hasn't changed, do nothing
  if (existing && existing.parentElement === toolbar && lastInjectedToolbar === toolbar) {
    return;
  }

  // Remove button if it's in wrong toolbar or toolbar changed
  if (existing && (existing.parentElement !== toolbar || lastInjectedToolbar !== toolbar)) {
    existing.remove();
  }

  // Only inject if no button exists and we have a toolbar
  if (toolbar && !document.getElementById('mailfind-summarize-btn')) {
    isInjecting = true;
    // Find the "More" button (three dots) to position our button after it
    const moreButton = (toolbar as Element).querySelector('[data-tooltip="More"], [aria-label="More"]');

    const button = document.createElement('button');
    button.id = 'mailfind-summarize-btn';
    button.className = 'mailfind-btn';
    button.innerHTML = '📧 Summarize';
    button.style.cssText = `
      background: #1a73e8;
      color: white;
      border: none;
      border-radius: 12px;
      height: 24px;
      line-height: 24px;
      padding: 0 10px;
      margin-left: 6px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      z-index: 1000;
      position: relative;
      display: inline-flex;
      align-items: center;
      box-sizing: border-box;
      vertical-align: middle;
      white-space: nowrap;
    `;

    // Hover effect
    button.addEventListener('mouseenter', () => {
      button.style.background = '#1557b0';
    });
    button.addEventListener('mouseleave', () => {
      button.style.background = '#1a73e8';
    });

    button.addEventListener('click', handleSummarizeClick);

    if (moreButton && moreButton.parentNode) {
      moreButton.parentNode.insertBefore(button, moreButton.nextSibling);
    } else {
      (toolbar as Element).appendChild(button);
    }

    // Update tracking variables
    lastInjectedToolbar = toolbar;
    isInjecting = false;
    
    console.log('✅ [Gmail] Summarize button injected');
  }
}

// Handle summarize button click
async function handleSummarizeClick() {
  console.log('📧 [Gmail] Summarize button clicked in Gmail interface');

  try {
    console.log('🔍 [Gmail] Trying direct content extraction approach...');

    // New approach: Extract visible email content directly from the page
    let emailContent = extractVisibleEmailContent();
    
    if (emailContent && emailContent.length > 100) {
      console.log('✅ [Gmail] Extracted email content directly from page:', emailContent.substring(0, 100) + '...');
      
      const toast = showToast('Summarizing visible email content...');
      
      // Send content directly to backend
      const response = await fetch('http://localhost:8000/summarize-content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: emailContent }),
        credentials: 'include'
      });
      
      if (response.ok) {
        // Stream text chunks and update UI live
        const bodyEl = renderInlineSummary('');
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let aggregated = '';
        if (!reader) {
          bodyEl.textContent = 'No stream available.';
          updateToast(toast, 'Summary ready.');
          hideToast(toast, 800);
          return;
        }
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (value) {
              const chunk = decoder.decode(value, { stream: true });
              aggregated += chunk;
              bodyEl.textContent = aggregated;
            }
          }
          updateToast(toast, 'Summary ready.');
          hideToast(toast, 800);
          return; // Success - exit early
        } catch (e) {
          console.error('❌ [Gmail] Stream read error', e);
          bodyEl.textContent = aggregated || 'Failed to read stream.';
          updateToast(toast, 'Stream ended with error.');
          hideToast(toast, 1200);
          return;
        }
      } else {
        console.warn('⚠️ [Gmail] Content summarization failed, falling back to thread ID approach');
      }
    } else {
      console.log('⚠️ [Gmail] Could not extract sufficient email content, falling back to thread ID approach');
    }

    // Fallback to thread ID approach if content extraction fails
    console.log('🔍 [Gmail] Falling back to thread ID detection...');
    let threadId: string | null = null;
    
    // Method 1: Look for visible thread elements in the current conversation
    const threadElements = document.querySelectorAll('[data-thread-id]');
    console.log('🔍 [Gmail] Found', threadElements.length, 'thread elements in DOM');
    
    // Strategy 1: Look for elements that are actually visible and in viewport
    let visibleElements = [];
    for (const element of threadElements) {
      const rect = element.getBoundingClientRect();
      const isVisible = rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top < window.innerHeight;
      
      if (isVisible) {
        let id = element.getAttribute('data-thread-id');
        if (id) {
          // Clean up the thread ID format
          if (id.startsWith('#thread-f:')) {
            id = id.substring(10); // Remove "#thread-f:" prefix
          }
          if (id && id.length > 10) {
            visibleElements.push({ element, id, rect });
          }
        }
      }
    }
    
    console.log('🔍 [Gmail] Found', visibleElements.length, 'visible thread elements');
    
    // Debug: Log some of the thread IDs we found
    if (visibleElements.length > 0) {
      console.log('🔍 [Gmail] Visible thread IDs:', visibleElements.slice(0, 3).map(v => v.id));
    }
    
    // Strategy 2: If we have visible elements, pick the most likely candidate
    if (visibleElements.length > 0) {
      // Prefer elements in main conversation areas
      for (const item of visibleElements) {
        const isInMainArea = item.element.closest('[role="main"]') !== null || 
                            item.element.closest('.nH.if') !== null ||
                            item.element.closest('.adn.ads') !== null;
        
        if (isInMainArea) {
          threadId = item.id;
          console.log('✅ [Gmail] Found thread ID from main conversation area:', threadId);
          break;
        }
      }
      
      // If no main area elements, take the first visible one
      if (!threadId) {
        threadId = visibleElements[0].id;
        console.log('✅ [Gmail] Using first visible thread ID:', threadId);
      }
    }
    
    // Strategy 3: If still no thread ID, be more aggressive - check all elements
    if (!threadId && threadElements.length > 0) {
      console.log('🔍 [Gmail] No visible elements found, checking all thread elements...');
      
      // Get unique thread IDs and their counts
      const threadIdCounts = new Map();
      for (const element of threadElements) {
        let id = element.getAttribute('data-thread-id');
        if (id) {
          if (id.startsWith('#thread-f:')) {
            id = id.substring(10);
          }
          if (id && id.length > 10) {
            threadIdCounts.set(id, (threadIdCounts.get(id) || 0) + 1);
          }
        }
      }
      
      // Strategy: Instead of just using the most frequent, try to find a newer/different thread
      // Sort by frequency and try to avoid the old stale thread (1842828099281674475)
      const sortedByFreq = Array.from(threadIdCounts.entries()).sort((a, b) => b[1] - a[1]);
      const staleThreadId = '1842828099281674475'; // Known stale thread
      
      // Try to find a thread that's not the stale one
      let selectedThread = null;
      for (const [id, count] of sortedByFreq) {
        if (id !== staleThreadId) {
          selectedThread = { id, count };
          console.log(`✅ [Gmail] Found alternative thread ID: ${id} (appears ${count} times, avoiding stale thread)`);
          break;
        }
      }
      
      // If no alternative found, use the most frequent (even if stale)
      if (!selectedThread && sortedByFreq.length > 0) {
        selectedThread = { id: sortedByFreq[0][0], count: sortedByFreq[0][1] };
        console.log(`⚠️ [Gmail] Using most frequent thread ID (might be stale): ${selectedThread.id} (appears ${selectedThread.count} times)`);
      }
      
      if (selectedThread) {
        threadId = selectedThread.id;
      } else {
        console.log('⚠️ [Gmail] No valid thread IDs found in frequency analysis');
      }
      
      // Debug: Show all thread IDs found with their frequencies
      const sortedThreadIds = Array.from(threadIdCounts.entries()).sort((a, b) => b[1] - a[1]);
      console.log('🔍 [Gmail] Thread ID frequency map (sorted):', sortedThreadIds.slice(0, 5));
    } else {
      console.log('⚠️ [Gmail] No thread elements found at all');
    }
    
    // Method 2: If no visible thread ID found, get the URL fragment and send it
    // The backend will handle the conversion
    if (!threadId) {
      console.log('🔍 [Gmail] No visible thread ID found, using URL fragment...');
      const hash = window.location.hash;
      console.log('🔍 [Gmail] Current URL hash:', hash);
      
      const fragmentMatch = hash.match(/#inbox\/([a-zA-Z0-9]+)/);
      if (fragmentMatch) {
        const urlFragment = fragmentMatch[1];
        console.log('✅ [Gmail] Using URL fragment as identifier:', urlFragment);
        threadId = urlFragment; // We'll send this to backend as a "messageId" for processing
      }
    }

    if (!threadId) {
      console.warn('⚠️ [Gmail] Could not find thread identifier. Make sure you\'re viewing a conversation.');
      const t = showToast('MailFind: Open a specific email conversation to summarize.');
      hideToast(t, 2500);
      return;
    }

    console.log('📤 [Gmail] Sending identifier to backend:', threadId);
    const toast = showToast(`Request sent for ${threadId.substring(0, 8)}...`);

    // Send as messageId - backend will determine if it's a real message ID or URL fragment
    const response = await fetch('http://localhost:8000/summarize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messageId: threadId }),
      credentials: 'include'
    });

    if (response.ok) {
      // Stream text chunks and update UI live
      const bodyEl = renderInlineSummary('');
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let aggregated = '';
      if (!reader) {
        bodyEl.textContent = 'No stream available.';
        updateToast(toast, 'Summary ready.');
        hideToast(toast, 800);
        return;
      }
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            aggregated += chunk;
            bodyEl.textContent = aggregated;
          }
        }
        updateToast(toast, 'Summary ready.');
        hideToast(toast, 800);
      } catch (e) {
        console.error('❌ [Gmail] Stream read error', e);
        bodyEl.textContent = aggregated || 'Failed to read stream.';
        updateToast(toast, 'Stream ended with error.');
        hideToast(toast, 1200);
      }
    } else {
      console.error(`❌ [Gmail] Backend error: ${response.status} ${response.statusText}`);
      if (response.status === 401) {
        updateToast(toast, 'Login required. Opening login page…');
        try {
          window.open('http://localhost:8000/login', '_blank', 'noopener');
        } catch (_) {}
      } else if (response.status === 404) {
        updateToast(toast, 'Thread not found. Open a conversation and try again.');
      } else if (response.status === 400) {
        updateToast(toast, 'Invalid request. Refresh Gmail and try again.');
      } else {
        updateToast(toast, 'Server error. Check backend logs.');
      }
      throw new Error(`Backend error: ${response.status}`);
    }
  } catch (error) {
    console.error('💥 [Gmail] Summarization failed:', error);
    const t = document.getElementById('mailfind-toast');
    if (t) updateToast(t as HTMLElement, 'MailFind: Failed.');
    alert('MailFind: Failed to send summary request. Please try again.');
  }
}

// Watch for Gmail navigation and inject button
function observeGmailChanges() {
  let debounceId: number | null = null;
  let lastUrl = window.location.href;
  
  const trigger = () => {
    if (debounceId) clearTimeout(debounceId);
    debounceId = window.setTimeout(() => {
      injectSummarizeButton();
    }, 800); // Increased debounce time
  };

  const observer = new MutationObserver((mutations) => {
    // Only trigger if URL changed or if there are significant DOM changes
    const currentUrl = window.location.href;
    if (currentUrl !== lastUrl) {
      lastUrl = currentUrl;
      trigger();
      return;
    }
    
    // Only trigger for significant changes (avoid infinite loops)
    const hasSignificantChange = mutations.some(mutation => 
      mutation.type === 'childList' && 
      mutation.addedNodes.length > 0 &&
      Array.from(mutation.addedNodes).some(node => 
        node.nodeType === Node.ELEMENT_NODE && 
        (node as Element).matches('[gh="mtb"], [gh="tm"], [role="toolbar"]')
      )
    );
    
    if (hasSignificantChange) {
      trigger();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  window.addEventListener('popstate', trigger);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(injectSummarizeButton, 1000);
    observeGmailChanges();
  });
} else {
  setTimeout(injectSummarizeButton, 1000);
  observeGmailChanges();
}

