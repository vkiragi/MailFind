// Content script that runs on Gmail pages
console.log('MailFind content script loaded');

// Function to inject the summarize button into Gmail
function injectSummarizeButton() {
  // Look for a good place to inject the button (Gmail's toolbar area)
  const toolbar = document.querySelector('[role="toolbar"]') || 
                 document.querySelector('.aXjCH') ||
                 document.querySelector('[data-tooltip="More"]')?.parentElement;
  
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
  }
}

// Handle summarize button click
function handleSummarizeClick() {
  console.log('Summarize button clicked');
  
  // TODO: Extract current email thread ID
  // TODO: Call backend API
  // TODO: Show summary modal
  
  // For now, just show a simple alert
  alert('MailFind: Summarization feature coming soon!');
}

// Watch for Gmail navigation and inject button
function observeGmailChanges() {
  const observer = new MutationObserver(() => {
    injectSummarizeButton();
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
