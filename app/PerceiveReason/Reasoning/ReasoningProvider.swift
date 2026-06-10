// ReasoningProvider.swift
//
// The Swift mirror of the Python `LanguageModel` protocol. Same idea as Apple's
// WWDC 2026 model-abstraction layer: the orchestration code (the view model)
// depends only on this protocol, so swapping Apple's on-device model for Claude
// is a one-line change at the call site.

import Foundation

/// A normalised reasoning result, regardless of which provider produced it.
struct ReasoningResult: Sendable {
    let text: String
    let provider: String
}

/// Any reasoning backend (on-device Foundation Models, Claude, etc.).
protocol ReasoningProvider: Sendable {
    var name: String { get }
    func reason(prompt: String, system: String) async throws -> ReasoningResult
}

/// Build the user prompt from detections. Kept identical to
/// `pipeline.reasoning.prompts.build_prompt` so on-device and cloud behave alike.
enum PromptBuilder {
    static let systemPrompt = """
    You are an on-device reasoning step in a perceive->reason pipeline. You \
    receive structured machine-vision output and must return a concise, \
    well-calibrated assessment. Never invent detections that are not present. \
    State your confidence and flag when the visual evidence is insufficient.
    """

    static func build(detections: [Detection], task: String) -> String {
        let observations: String
        if detections.isEmpty {
            observations = "(no detections returned by the vision model)"
        } else {
            observations = detections
                .map { String(format: "- %@ (%.2f)", $0.label, $0.confidence) }
                .joined(separator: "\n")
        }
        return """
        Task: \(task)

        Machine-vision detections:
        \(observations)

        Using only the detections above, give a brief assessment, state your \
        confidence, and recommend the next action. If the evidence is weak or \
        ambiguous, say so explicitly rather than guessing.
        """
    }
}
