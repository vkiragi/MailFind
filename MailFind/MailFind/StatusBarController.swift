import AppKit

final class StatusBarController: NSObject {
    private let statusItem: NSStatusItem
    private let popover: NSPopover

    init(popover: NSPopover) {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        self.popover = popover
        super.init()

        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "magnifyingglass.circle", accessibilityDescription: "Search")
            button.action = #selector(togglePopover(_:))
            button.target = self
        }
    }

    @objc func togglePopover(_ sender: Any?) {
        if popover.isShown {
            popover.performClose(sender)
        } else if let button = statusItem.button {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}
