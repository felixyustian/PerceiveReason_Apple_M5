// PerceiveReasonApp.swift
//
// On-device "Perceive -> Reason" reference app for the Apple ML stack.
//
// ┌──────────────────────────────────────────────────────────────────────┐
// │ BUILD NOTE — read before opening in Xcode                              │
// │                                                                        │
// │ This target uses the Foundation Models framework updates announced at  │
// │ WWDC 2026 (image input + the swappable `LanguageModel` provider).      │
// │ It is written to compile with Xcode 27 against iOS 27 / macOS 27.      │
// │ It is intentionally NOT built in this repo's Linux CI — Swift +        │
// │ Core ML + Foundation Models require macOS / Apple silicon. Compile and │
// │ run it on your Mac. The Python pipeline in ../pipeline is what CI      │
// │ verifies.                                                              │
// └──────────────────────────────────────────────────────────────────────┘

import SwiftUI

@main
struct PerceiveReasonApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
