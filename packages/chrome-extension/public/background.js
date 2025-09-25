// Background script for MailFind extension
// Handles side panel opening when extension icon is clicked

chrome.action.onClicked.addListener((tab) => {
  // Open the side panel for the current tab
  chrome.sidePanel.open({ tabId: tab.id });
});

// Handle keyboard shortcut
chrome.commands.onCommand.addListener((command) => {
  if (command === '_execute_action') {
    // Get the current active tab
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.sidePanel.open({ tabId: tabs[0].id });
      }
    });
  }
});
