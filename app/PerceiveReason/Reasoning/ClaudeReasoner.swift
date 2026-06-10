// ClaudeReasoner.swift
//
// The REASON stage routed to Anthropic Claude — the "swap providers in one
// line" pattern Apple highlighted at WWDC 2026. Use it when a query is harder
// than the on-device model should handle, or when you want a quality ceiling
// above the local model.
//
// This implementation calls the Anthropic Messages API directly with
// URLSession so the repo has no third-party Swift dependency. In production you
// would instead use Apple's `LanguageModel` provider plumbing (e.g. via the
// official Anthropic Swift package) so the same `LanguageModelSession` API
// works for both providers and keys stay in the Keychain.
//
// SECURITY: never ship an API key in the app binary. Read it from the Keychain
// or a server-side proxy. The env/Info.plist lookup below is for local dev only.

import Foundation

struct ClaudeReasoner: ReasoningProvider {
    let name = "claude"
    let model: String

    init(model: String = "claude-sonnet-4-6") {
        self.model = model
    }

    private var apiKey: String? {
        ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]
            ?? Bundle.main.object(forInfoDictionaryKey: "ANTHROPIC_API_KEY") as? String
    }

    func reason(prompt: String, system: String) async throws -> ReasoningResult {
        guard let apiKey, !apiKey.isEmpty else { throw ReasoningError.missingAPIKey }

        var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.setValue("application/json", forHTTPHeaderField: "content-type")

        let payload: [String: Any] = [
            "model": model,
            "max_tokens": 1024,
            "system": system,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, _) = try await URLSession.shared.data(for: request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let blocks = (json?["content"] as? [[String: Any]]) ?? []
        let text = blocks
            .compactMap { $0["type"] as? String == "text" ? $0["text"] as? String : nil }
            .joined()
        return ReasoningResult(text: text, provider: name)
    }
}
