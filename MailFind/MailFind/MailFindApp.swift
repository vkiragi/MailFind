import SwiftUI

@main
struct MailFindApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // No main window for now; menubar only. Keep Settings empty.
        Settings { EmptyView() }
    }
}
