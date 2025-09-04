// Content script that runs on Gmail pages
console.log('MailFind content script loaded');

// Gmail DOM selectors - updated with real Gmail DOM elements
// List view toolbar often uses gh="tm"; thread view often uses gh="mtb".
// We also include generic ARIA-based selectors to be resilient across layouts/locales.
const GMAIL_LIST_TOOLBAR_SELECTOR = 'div[gh="tm"]';
const GMAIL_THREAD_TOOLBAR_SELECTOR = 'div[gh="mtb"]';
const GMAIL_ARIA_TOOLBAR_SELECTOR = '[aria-label="Toolbar"], [role="toolbar"]';
const GMAIL_EMAIL_VIEW_SELECTOR = '.aeH'; // Generic container present in Gmail views

const TOOLBAR_CANDIDATES = [
  GMAIL_THREAD_TOOLBAR_SELECTOR,
  GMAIL_LIST_TOOLBAR_SELECTOR,
  GMAIL_ARIA_TOOLBAR_SELECTOR,
].join(', ');

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
  console.log(`🔍 [Gmail] Toolbar candidates found: ${candidates.length}`);

  // Prefer visible candidates
  const visible = candidates.filter(isVisible);

  // Prefer thread toolbar gh="mtb" when present
  const threadFirst = visible.sort((a, b) => {
    const aIsThread = (a as HTMLElement).getAttribute('gh') === 'mtb' ? 1 : 0;
    const bIsThread = (b as HTMLElement).getAttribute('gh') === 'mtb' ? 1 : 0;
    return bIsThread - aIsThread;
  });

  const chosen = threadFirst[0] || null;
  console.log('🔍 [Gmail] Chosen toolbar:', chosen);
  return chosen;
}

// Function to inject the summarize button into Gmail
function injectSummarizeButton() {
  const existing = document.getElementById('mailfind-summarize-btn');
  const toolbar = findToolbar();

  if (!toolbar) {
    console.log('⚠️ [Gmail] No toolbar found; will retry on next mutation');
    return;
  }

  if (existing && existing.parentElement !== toolbar) {
    existing.remove();
  }

  if (toolbar && !document.getElementById('mailfind-summarize-btn')) {
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
      // Fallback: look for thread ID in URL or other common locations
      const url = window.location.href;
      const threadMatch = url.match(/th=([a-f0-9]+)/);
      if (threadMatch) {
        threadId = threadMatch[1];
        console.log('✅ [Gmail] Found thread ID from URL:', threadId);
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

    if (!threadId) {
      console.warn('⚠️ [Gmail] Could not find thread ID, using fallback');
      threadId = 'fallback-thread-id';
    }

    console.log('📤 [Gmail] Sending thread ID to backend:', threadId);

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
      const result = await response.json();
      console.log('✅ [Gmail] Backend response:', result);
      alert(`MailFind: ${result.message || 'Summary request sent successfully!'}`);
    } else {
      console.error(`❌ [Gmail] Backend error: ${response.status} ${response.statusText}`);
      throw new Error(`Backend error: ${response.status}`);
    }
  } catch (error) {
    console.error('💥 [Gmail] Summarization failed:', error);
    alert('MailFind: Failed to send summary request. Please try again.');
  }
}

// Watch for Gmail navigation and inject button
function observeGmailChanges() {
  let debounceId: number | null = null;
  const trigger = () => {
    if (debounceId) clearTimeout(debounceId);
    debounceId = window.setTimeout(() => {
      injectSummarizeButton();
    }, 400);
  };

  const observer = new MutationObserver(() => {
    trigger();
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
