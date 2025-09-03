// Content script that runs on Gmail pages
console.log('MailFind content script loaded');

// Gmail DOM selectors - updated with real Gmail DOM elements
// GMAIL_TOOLBAR_SELECTOR: Real Gmail email toolbar selector found through DOM inspection
// GMAIL_EMAIL_VIEW_SELECTOR: Still needs to be updated with real Gmail email view selector
const GMAIL_TOOLBAR_SELECTOR = 'div[gh="tm"]';
const GMAIL_EMAIL_VIEW_SELECTOR = '.aXjCH';

// Function to inject the summarize button into Gmail
function injectSummarizeButton() {
  console.log('🔍 [Gmail] Attempting to inject summarize button...');
  
  // Look for a good place to inject the button (Gmail's toolbar area)
  const toolbar = document.querySelector(GMAIL_TOOLBAR_SELECTOR) || 
                 document.querySelector('[role="toolbar"]') ||
                 document.querySelector('[data-tooltip="More"]')?.parentElement;
  
  console.log('📱 [Gmail] Toolbar element found:', !!toolbar);
  
  if (toolbar && !document.getElementById('mailfind-summarize-btn')) {
    const button = document.createElement('button');
    button.id = 'mailfind-summarize-btn';
    button.className = 'mailfind-btn';
    button.innerHTML = '📧 Summarize';
    button.style.cssText = `
      background: #1a73e8;
      color: white;
      border: none;
      border-radius: 4px;
      padding: 8px 16px;
      margin-left: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
    `;
    
    button.addEventListener('click', handleSummarizeClick);
    toolbar.appendChild(button);
    console.log('✅ [Gmail] Summarize button successfully injected into Gmail toolbar');
  } else {
    console.log('⚠️ [Gmail] Button injection skipped - button already exists or toolbar not found');
  }
}

// Handle summarize button click
function handleSummarizeClick() {
  console.log('📧 [Gmail] Summarize button clicked in Gmail interface');
  
  // TODO: Extract current email thread ID from Gmail DOM
  // TODO: Call backend API with real thread ID
  // TODO: Show summary modal with real data
  
  // For now, just show a simple alert
  alert('MailFind: Summarization feature coming soon!');
}

// Watch for Gmail navigation and inject button
function observeGmailChanges() {
  const observer = new MutationObserver(() => {
    // Check if we're in an email view before injecting
    if (document.querySelector(GMAIL_EMAIL_VIEW_SELECTOR)) {
      injectSummarizeButton();
    }
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
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
