// OnDeviceReasoner.swift
//
// The REASON stage running entirely on device via Apple's Foundation Models
// framework — no network, no API key, free. This is the default provider.
//
// Requires iOS 27 / macOS 27 (Foundation Models) and a device whose chip
// supports the on-device model. `SystemLanguageModel.isAvailable` guards that.

import Foundation
import FoundationModels   // Apple, iOS 26+ (expanded at WWDC 2026)

/// Reasoning backed by Apple's on-device `SystemLanguageModel`.
struct OnDeviceReasoner: ReasoningProvider {
    let name = "apple-on-device"

    func reason(prompt: String, system: String) async throws -> ReasoningResult {
        // Availability reflects device capability + user settings.
        let model = SystemLanguageModel.default
        guard case .available = model.availability else {
            throw ReasoningError.modelUnavailable
        }

        // `instructions` is Foundation Models' system-prompt equivalent.
        let session = LanguageModelSession(model: model, instructions: system)
        let response = try await session.respond(to: prompt)
        return ReasoningResult(text: response.content, provider: name)
    }
}

enum ReasoningError: Error {
    case modelUnavailable
    case missingAPIKey
}
