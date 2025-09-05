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
function renderInlineSummary(summaryText: string) {
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
  header.style.cssText = 'display:flex;align-items:center;margin-bottom:8px;';
  const title = document.createElement('div');
  title.textContent = 'MailFind Summary';
  title.style.cssText = 'font-weight:600;font-size:14px;flex:1;';
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.setAttribute('aria-label', 'Close summary');
  closeBtn.title = 'Close';
  closeBtn.style.cssText = 'border:none;background:transparent;color:#5f6368;cursor:pointer;font-size:18px;line-height:18px;';
  closeBtn.addEventListener('click', () => card.remove());

  header.appendChild(title);
  header.appendChild(closeBtn);
  const body = document.createElement('div');
  body.textContent = summaryText;
  card.appendChild(header);
  card.appendChild(body);
  document.body.appendChild(card);
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
      const result = await response.json();
      console.log('✅ [Gmail] Backend response:', result);
      updateToast(toast, 'Summary ready.');
      hideToast(toast, 800);
      const summary = (result && (result.summary || result.message)) || 'Summary generated.';
      renderInlineSummary(summary);
    } else {
      console.error(`❌ [Gmail] Backend error: ${response.status} ${response.statusText}`);
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
