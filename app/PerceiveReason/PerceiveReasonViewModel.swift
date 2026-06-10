// PerceiveReasonViewModel.swift
//
// Orchestration: image -> VisionPerceptor (Core ML) -> ReasoningProvider -> text.
// The view model depends only on the `ReasoningProvider` protocol, so the
// on-device/Claude swap is a single assignment.

import SwiftUI
import CoreImage

@MainActor
final class PerceiveReasonViewModel: ObservableObject {

    enum ProviderChoice: String, CaseIterable, Identifiable {
        case onDevice = "On-device (Apple)"
        case claude = "Claude (cloud)"
        var id: String { rawValue }
    }

    @Published var task: String = "Assess PPE compliance for this worker."
    @Published var providerChoice: ProviderChoice = .onDevice
    @Published var detections: [Detection] = []
    @Published var assessment: String = ""
    @Published var isRunning = false
    @Published var errorMessage: String?

    private let perceptor: VisionPerceptor?

    init() {
        // Loading can fail in previews / before a model is bundled; surface it
        // rather than crashing.
        self.perceptor = try? VisionPerceptor()
    }

    private func makeProvider() -> ReasoningProvider {
        switch providerChoice {
        case .onDevice: return OnDeviceReasoner()
        case .claude:   return ClaudeReasoner()
        }
    }

    func run(on image: CGImage) async {
        guard let perceptor else {
            errorMessage = "Vision model not bundled. Run the Python pipeline to produce it."
            return
        }
        isRunning = true
        errorMessage = nil
        defer { isRunning = false }

        do {
            // PERCEIVE
            let dets = try await perceptor.perceive(image)
            detections = dets

            // REASON
            let provider = makeProvider()
            let prompt = PromptBuilder.build(detections: dets, task: task)
            let result = try await provider.reason(
                prompt: prompt, system: PromptBuilder.systemPrompt)
            assessment = "[\(result.provider)] \(result.text)"
        } catch {
            errorMessage = "\(error)"
        }
    }
}
