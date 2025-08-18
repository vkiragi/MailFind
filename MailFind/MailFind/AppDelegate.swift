import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusBarController: StatusBarController!
    private let popover = NSPopover()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Build the popover with your SwiftUI view
        popover.behavior = .transient
        popover.contentSize = NSSize(width: 420, height: 520)
        popover.contentViewController = NSHostingController(rootView: SearchView())

        // Create the menu bar item
        statusBarController = StatusBarController(popover: popover)

        // Register global hotkey: ⌥⌘K
        HotKeyManager.shared.registerToggle { [weak self] in
            self?.statusBarController.togglePopover(nil)
        }

        // Optional: bring to foreground when first launching
        NSApp.activate(ignoringOtherApps: true)
    }
}
