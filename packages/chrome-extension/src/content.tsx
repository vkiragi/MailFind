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

    // Check for dark mode
    const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

    toast.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 2147483647;
      background: ${isDarkMode ?
        'linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.95))' :
        'linear-gradient(135deg, rgba(30, 58, 138, 0.95), rgba(79, 70, 229, 0.95))'};
      backdrop-filter: blur(12px);
      color: #fff;
      padding: 16px 20px;
      border-radius: 24px;
      font-size: 13px;
      font-weight: 500;
      box-shadow: 0 10px 25px rgba(0,0,0,0.15), 0 4px 10px rgba(0,0,0,0.1);
      border: 1px solid rgba(255,255,255,0.2);
      max-width: 380px;
      line-height: 1.5;
      pointer-events: none;
      transform: translateY(-10px);
      opacity: 0;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    `;
    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
      toast!.style.transform = 'translateY(0)';
      toast!.style.opacity = '1';
    });
  }
  toast.textContent = message;
  return toast;
}

function updateToast(toast: HTMLElement, message: string) {
  if (toast) toast.textContent = message;
}

function hideToast(toast: HTMLElement, delayMs = 1200) {
  if (!toast) return;
  setTimeout(() => {
    if (toast) {
      toast.style.transform = 'translateY(-10px)';
      toast.style.opacity = '0';
      setTimeout(() => {
        if (toast && toast.parentNode) {
          toast.remove();
        }
      }, 300);
    }
  }, delayMs);
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

  // Check for dark mode
  const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

  card.style.cssText = `
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 2147483647;
    width: 440px;
    max-height: 75vh;
    overflow: hidden;
    background: ${isDarkMode ?
      'linear-gradient(135deg, #1f2937 0%, #111827 100%)' :
      'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)'};
    color: ${isDarkMode ? '#f9fafb' : '#1f2937'};
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12), 0 8px 16px rgba(0,0,0,0.08);
    border: 1px solid ${isDarkMode ? 'rgba(75, 85, 99, 0.3)' : 'rgba(148, 163, 184, 0.2)'};
    backdrop-filter: blur(12px);
    transform: translateY(-20px) scale(0.95);
    opacity: 0;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  `;

  // Animate in
  document.body.appendChild(card);
  requestAnimationFrame(() => {
    card.style.transform = 'translateY(0) scale(1)';
    card.style.opacity = '1';
  });

  const headerIsDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

  const header = document.createElement('div');
  header.style.cssText = `
    display: flex;
    align-items: center;
    padding: 20px 20px 16px 20px;
    background: ${headerIsDarkMode ?
      'linear-gradient(135deg, #475569 0%, #334155 100%)' :
      'linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)'};
    border-radius: 24px 24px 0 0;
    margin: 0;
    gap: 8px;
    cursor: move;
    user-select: none;
  `;

  const iconContainer = document.createElement('div');
  iconContainer.style.cssText = `
    width: 32px;
    height: 32px;
    background: rgba(255,255,255,0.2);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
  `;

  // Add mail icon SVG instead of emoji
  iconContainer.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.5">
      <path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 7.89a2 2 0 002.83 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  `;

  const titleContainer = document.createElement('div');
  titleContainer.style.cssText = 'flex: 1;';

  const title = document.createElement('div');
  title.textContent = 'MailFind Summary';
  title.style.cssText = 'font-weight: 600; font-size: 15px; color: white; margin-bottom: 2px;';

  const subtitle = document.createElement('div');
  subtitle.textContent = 'AI-powered email insights';
  subtitle.style.cssText = 'font-size: 11px; color: rgba(255,255,255,0.8);';

  titleContainer.appendChild(title);
  titleContainer.appendChild(subtitle);

  // Minimize toggle button
  const minimizeBtn = document.createElement('button');
  minimizeBtn.innerHTML = '−';
  minimizeBtn.setAttribute('aria-label', 'Minimize summary');
  minimizeBtn.title = 'Minimize';
  minimizeBtn.style.cssText = `
    border: none;
    background: rgba(255,255,255,0.2);
    color: white;
    cursor: pointer;
    font-size: 16px;
    width: 28px;
    height: 28px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.2s;
  `;
  minimizeBtn.addEventListener('mouseenter', () => {
    minimizeBtn.style.background = 'rgba(255,255,255,0.3)';
  });
  minimizeBtn.addEventListener('mouseleave', () => {
    minimizeBtn.style.background = 'rgba(255,255,255,0.2)';
  });

  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '×';
  closeBtn.setAttribute('aria-label', 'Close summary');
  closeBtn.title = 'Close';
  closeBtn.style.cssText = `
    border: none;
    background: rgba(255,255,255,0.2);
    color: white;
    cursor: pointer;
    font-size: 18px;
    width: 28px;
    height: 28px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.2s;
  `;
  closeBtn.addEventListener('mouseenter', () => {
    closeBtn.style.background = 'rgba(255,255,255,0.3)';
  });
  closeBtn.addEventListener('mouseleave', () => {
    closeBtn.style.background = 'rgba(255,255,255,0.2)';
  });
  closeBtn.addEventListener('click', () => {
    card.style.transform = 'translateY(-20px) scale(0.95)';
    card.style.opacity = '0';
    setTimeout(() => card.remove(), 300);
  });

  header.appendChild(iconContainer);
  header.appendChild(titleContainer);
  header.appendChild(minimizeBtn);
  header.appendChild(closeBtn);

  // Add drag functionality - with enhanced debugging
  let isDragging = false;
  let dragOffset = { x: 0, y: 0 };

  // Add visual indicator for debugging
  console.log('🖱️ [Drag] Setting up drag functionality for header');
  
  // Make entire header clickable area more obvious
  header.style.minHeight = '60px';
  
  header.addEventListener('mousedown', (e) => {
    console.log('🖱️ [Drag] Mouse down on header', e.target);
    
    // Don't start drag if clicking on buttons
    if ((e.target as Element).closest('button')) {
      console.log('🖱️ [Drag] Clicked on button, ignoring');
      return;
    }
    
    console.log('🖱️ [Drag] Starting drag operation');
    isDragging = true;
    const rect = card.getBoundingClientRect();
    dragOffset.x = e.clientX - rect.left;
    dragOffset.y = e.clientY - rect.top;
    
    // Add visual feedback
    card.style.transition = 'none';
    card.style.transform = 'scale(1.02)';
    card.style.boxShadow = '0 25px 50px rgba(0,0,0,0.25), 0 12px 24px rgba(0,0,0,0.15)';
    header.style.cursor = 'grabbing';
    document.body.style.cursor = 'grabbing';
    
    e.preventDefault();
    e.stopPropagation();
  });

  // Use a more specific event handler to avoid conflicts
  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    
    console.log('🖱️ [Drag] Moving card to', e.clientX, e.clientY);
    const x = e.clientX - dragOffset.x;
    const y = e.clientY - dragOffset.y;
    
    // Keep card within viewport bounds
    const maxX = window.innerWidth - card.offsetWidth;
    const maxY = window.innerHeight - card.offsetHeight;
    
    const clampedX = Math.max(0, Math.min(x, maxX));
    const clampedY = Math.max(0, Math.min(y, maxY));
    
    card.style.left = `${clampedX}px`;
    card.style.top = `${clampedY}px`;
    card.style.right = 'auto';
    
    e.preventDefault();
  };

  const handleMouseUp = () => {
    if (!isDragging) return;
    
    console.log('🖱️ [Drag] Ending drag operation');
    isDragging = false;
    
    // Restore visual state
    card.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
    card.style.transform = 'scale(1)';
    card.style.boxShadow = '0 20px 40px rgba(0,0,0,0.12), 0 8px 16px rgba(0,0,0,0.08)';
    header.style.cursor = 'move';
    document.body.style.cursor = '';
  };

  document.addEventListener('mousemove', handleMouseMove);
  document.addEventListener('mouseup', handleMouseUp);
  
  // Clean up event listeners when card is removed
  const originalRemove = card.remove;
  card.remove = function() {
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    originalRemove.call(this);
  };

  const body = document.createElement('div');
  body.style.cssText = `
    padding: 20px;
    font-size: 14px;
    line-height: 1.6;
    color: ${headerIsDarkMode ? '#f3f4f6' : '#374151'};
    white-space: pre-wrap;
    max-height: calc(75vh - 80px);
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: ${headerIsDarkMode ? '#4b5563 #374151' : '#d1d5db #f3f4f6'};
  `;

  // Custom scrollbar styles for webkit browsers
  const style = document.createElement('style');
  style.textContent = `
    #mailfind-summary-card .summary-body::-webkit-scrollbar {
      width: 6px;
    }
    #mailfind-summary-card .summary-body::-webkit-scrollbar-track {
      background: #f3f4f6;
      border-radius: 3px;
    }
    #mailfind-summary-card .summary-body::-webkit-scrollbar-thumb {
      background: #d1d5db;
      border-radius: 3px;
    }
    #mailfind-summary-card .summary-body::-webkit-scrollbar-thumb:hover {
      background: #9ca3af;
    }
  `;
  document.head.appendChild(style);

  body.className = 'summary-body';
  body.textContent = summaryText;

  card.appendChild(header);
  card.appendChild(body);

  // Minimize/expand behavior
  let collapsed = false;
  const applyCollapsed = () => {
    if (collapsed) {
      body.style.display = 'none';
      card.style.width = '280px';
      card.style.maxHeight = 'auto';
      title.textContent = 'MailFind Summary';
      subtitle.textContent = 'Minimized - click to expand';
      minimizeBtn.innerHTML = '+';
      minimizeBtn.title = 'Expand';
      minimizeBtn.setAttribute('aria-label', 'Expand summary');
    } else {
      body.style.display = '';
      card.style.width = '440px';
      card.style.maxHeight = '75vh';
      title.textContent = 'MailFind Summary';
      subtitle.textContent = 'AI-powered email insights';
      minimizeBtn.innerHTML = '−';
      minimizeBtn.title = 'Minimize';
      minimizeBtn.setAttribute('aria-label', 'Minimize summary');
    }
  };
  minimizeBtn.addEventListener('click', () => {
    collapsed = !collapsed;
    applyCollapsed();
  });
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
  // Find the filter toolbar with "Is unread", "Advanced search" links
  // This is typically a div.Dj or div containing the "Advanced search" link
  const advancedSearchLink = document.querySelector('a[href*="advanced-search"]');

  if (advancedSearchLink) {
    const filterToolbar = advancedSearchLink.closest('.Dj') || advancedSearchLink.parentElement;
    if (filterToolbar && isVisible(filterToolbar)) {
      console.log('✅ [Gmail] Found filter toolbar with Advanced search link');
      return filterToolbar;
    }
  }

  // Alternative: Look for the toolbar by class
  const filterToolbars = document.querySelectorAll('.Dj');
  for (const toolbar of filterToolbars) {
    if (isVisible(toolbar) && toolbar.textContent?.includes('Advanced search')) {
      console.log('✅ [Gmail] Found filter toolbar by text content');
      return toolbar;
    }
  }

  // Fallback to email action toolbars
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

    // Check if we're in the filter toolbar (with "Advanced search" link)
    const isFilterToolbar = toolbar.classList.contains('Dj') || toolbar.textContent?.includes('Advanced search');

    // Find where to insert the button
    let insertAfter: Element | null = null;

    if (isFilterToolbar) {
      // In filter toolbar: find the "Advanced search" link
      insertAfter = toolbar.querySelector('a[href*="advanced-search"]');
      console.log('📍 [Gmail] Filter toolbar - insert after Advanced search link:', !!insertAfter);
    } else {
      // In email action toolbar: find "More" button
      insertAfter = toolbar.querySelector('[data-tooltip="More"], [aria-label="More"]');
      console.log('📍 [Gmail] Action toolbar - insert after More button:', !!insertAfter);
    }

    const button = document.createElement('button');
    button.id = 'mailfind-summarize-btn';
    button.className = 'mailfind-btn';

    // Check for dark mode in Gmail
    const gmailIsDarkMode = document.documentElement.getAttribute('data-color-scheme') === 'dark' ||
                           document.querySelector('[data-color-scheme="dark"]') !== null;

    button.innerHTML = `
      <svg style="margin-right: 6px; width: 14px; height: 14px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 7.89a2 2 0 002.83 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
      <span>Summarize</span>
    `;

    // Style differently based on toolbar type
    if (isFilterToolbar) {
      // Match filter toolbar link style
      button.style.cssText = `
        background: ${gmailIsDarkMode ?
          'linear-gradient(135deg, #475569 0%, #334155 100%)' :
          'linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)'};
        color: white;
        border: none;
        border-radius: 16px;
        height: 28px;
        padding: 0 12px;
        margin-left: 12px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        vertical-align: middle;
        white-space: nowrap;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      `;
    } else {
      // Original action toolbar style
      button.style.cssText = `
        background: ${gmailIsDarkMode ?
          'linear-gradient(135deg, #475569 0%, #334155 100%)' :
          'linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)'};
        color: white;
        border: none;
        border-radius: 20px;
        height: 36px;
        padding: 0 16px;
        margin: 0 4px;
        cursor: pointer;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        vertical-align: middle;
        white-space: nowrap;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        flex-shrink: 0;
      `;
    }

    // Hover and interaction effects
    const originalBg = gmailIsDarkMode ?
      'linear-gradient(135deg, #475569 0%, #334155 100%)' :
      'linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)';
    const hoverBg = gmailIsDarkMode ?
      'linear-gradient(135deg, #374151 0%, #1f2937 100%)' :
      'linear-gradient(135deg, #2563eb 0%, #5b21b6 100%)';

    button.addEventListener('mouseenter', () => {
      button.style.background = hoverBg;
      button.style.transform = 'scale(1.05)';
      button.style.boxShadow = '0 4px 8px rgba(59, 130, 246, 0.4)';
    });
    button.addEventListener('mouseleave', () => {
      button.style.background = originalBg;
      button.style.transform = 'scale(1)';
      button.style.boxShadow = '0 2px 4px rgba(59, 130, 246, 0.3)';
    });
    button.addEventListener('mousedown', () => {
      button.style.transform = 'scale(0.98)';
    });
    button.addEventListener('mouseup', () => {
      button.style.transform = 'scale(1.05)';
    });

    button.addEventListener('click', handleSummarizeClick);

    if (insertAfter && insertAfter.parentNode) {
      insertAfter.parentNode.insertBefore(button, insertAfter.nextSibling);
      console.log('✅ [Gmail] Button inserted after target element');
    } else {
      (toolbar as Element).appendChild(button);
      console.log('⚠️ [Gmail] Button appended to toolbar (no target found)');
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
    const emailContent = extractVisibleEmailContent();
    
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
    const visibleElements = [];
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
        } catch {
          // Ignore popup errors
        }
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

