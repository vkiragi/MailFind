// Content script that runs on Gmail pages
console.log('MailFind content script loaded');

// Gmail DOM selectors - updated with real Gmail DOM elements
// GMAIL_TOOLBAR_SELECTOR: Real Gmail email toolbar selector found through DOM inspection
// GMAIL_EMAIL_VIEW_SELECTOR: Still needs to be updated with real Gmail email view selector
const GMAIL_TOOLBAR_SELECTOR = 'div[gh="tm"]';
const GMAIL_EMAIL_VIEW_SELECTOR = '.aeH'; // The toolbar container

// Function to inject the summarize button into Gmail
function injectSummarizeButton() {
  console.log('🔍 [Gmail] Attempting to inject summarize button...');
  
  // Debug: Log all potential toolbar elements
  console.log('🔍 [Gmail] Searching for toolbar elements...');
  
  // Use the known Gmail toolbar selector
  const toolbar = document.querySelector(GMAIL_TOOLBAR_SELECTOR);
  console.log(`🔍 [Gmail] Looking for toolbar with selector: ${GMAIL_TOOLBAR_SELECTOR}`);
  console.log(`📱 [Gmail] Toolbar element found:`, !!toolbar);
  
  if (toolbar) {
    console.log('✅ [Gmail] Found toolbar element:', toolbar);
    console.log('🔍 [Gmail] Toolbar classes:', toolbar.className);
    console.log('🔍 [Gmail] Toolbar attributes:', Array.from(toolbar.attributes).map(attr => `${attr.name}="${attr.value}"`));
  }
  
  // Look for elements with gh attribute
  const ghElements = document.querySelectorAll('[gh]');
  console.log('🔍 [Gmail] Elements with gh attribute:', Array.from(ghElements).map(el => ({ 
    gh: el.getAttribute('gh'), 
    tagName: el.tagName,
    className: el.className 
  })));
  
  // Look for elements near action buttons
  const actionButtons = document.querySelectorAll('[data-tooltip]');
  console.log('🔍 [Gmail] Action buttons found:', Array.from(actionButtons).map(el => ({
    tooltip: el.getAttribute('data-tooltip'),
    parent: el.parentElement?.tagName,
    parentClass: el.parentElement?.className
  })));
  
  if (toolbar && !document.getElementById('mailfind-summarize-btn')) {
    // Find the "More" button (three dots) to position our button after it
    const moreButton = toolbar.querySelector('[data-tooltip="More"]');
    
    if (moreButton) {
      console.log('✅ [Gmail] Found More button, positioning summarize button after it');
      
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
      
      // Add hover effect
      button.addEventListener('mouseenter', () => {
        button.style.background = '#1557b0';
      });
      
      button.addEventListener('mouseleave', () => {
        button.style.background = '#1a73e8';
      });
      
      button.addEventListener('click', handleSummarizeClick);
      
      // Insert after the More button
      if (moreButton.parentNode) {
        moreButton.parentNode.insertBefore(button, moreButton.nextSibling);
      } else {
        // Fallback: append to toolbar
        toolbar.appendChild(button);
      }
      
      console.log('✅ [Gmail] Summarize button successfully injected after More button');
    } else {
      console.log('⚠️ [Gmail] More button not found, using fallback injection');
      
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
      
      button.addEventListener('click', handleSummarizeClick);
      toolbar.appendChild(button);
      console.log('✅ [Gmail] Summarize button injected using fallback method');
    }
  } else {
    console.log('⚠️ [Gmail] Button injection skipped - button already exists or toolbar not found');
  }
}

// Handle summarize button click
async function handleSummarizeClick() {
  console.log('📧 [Gmail] Summarize button clicked in Gmail interface');
  
  try {
    // Search for thread ID in Gmail DOM
    console.log('🔍 [Gmail] Searching for thread ID...');
    
    // Look for elements with data-thread-id attribute
    const threadIdElement = document.querySelector('[data-thread-id]');
    let threadId = null;
    
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
