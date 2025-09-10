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
    console.log('🔍 [Gmail] Searching for thread ID...');

    // Look for elements with data-thread-id attribute
    const threadIdElement = document.querySelector('[data-thread-id]');
    let threadId: string | null = null;

    if (threadIdElement) {
      threadId = threadIdElement.getAttribute('data-thread-id');
      console.log('✅ [Gmail] Found thread ID from data-thread-id:', threadId);
    } else {
      // Fallbacks: URL param th=, then list view legacy id, then page content
      const url = window.location.href;
      const threadMatch = url.match(/[#&?]th=([a-f0-9]+)/i);
      if (threadMatch) {
        threadId = threadMatch[1];
        console.log('✅ [Gmail] Found thread ID from URL:', threadId);
      } else {
        const row = document.querySelector('[data-legacy-thread-id]');
        const legacyId = row && (row as HTMLElement).getAttribute('data-legacy-thread-id');
        if (legacyId) {
          threadId = legacyId;
          console.log('✅ [Gmail] Found legacy thread ID from list row:', threadId);
        } else {
          // Look for thread ID in page content
          const pageContent = document.body.innerHTML;
          const contentMatch = pageContent.match(/thread_id["\s]*[:=]["\s]*([a-f0-9]+)/i);
          if (contentMatch) {
            threadId = contentMatch[1];
            console.log('✅ [Gmail] Found thread ID from page content:', threadId);
          }
        }
      }
    }

    if (!threadId) {
      console.warn('⚠️ [Gmail] Could not find thread ID. Open a conversation and retry.');
      const t = showToast('MailFind: Open a conversation to summarize.');
      hideToast(t, 1800);
      return;
    }

    console.log('📤 [Gmail] Sending thread ID to backend:', threadId);
    const toast = showToast('Request sent. Generating summary…');

    // Call backend API
    const response = await fetch('http://localhost:8000/summarize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ threadId }),
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
