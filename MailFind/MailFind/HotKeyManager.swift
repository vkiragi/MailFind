//
//  HotKeyManager.swift
//  MailFind
//
//  Created by Varun Kiragi on 8/17/25.
//


import Foundation
import HotKey

final class HotKeyManager {
    static let shared = HotKeyManager()
    private var hotKey: HotKey?

    func registerToggle(_ handler: @escaping () -> Void) {
        // ⌥⌘K
        hotKey = HotKey(key: .k, modifiers: [.command, .option])
        hotKey?.keyDownHandler = handler
    }
}
