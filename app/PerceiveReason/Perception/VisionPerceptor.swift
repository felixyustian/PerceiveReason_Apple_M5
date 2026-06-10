// VisionPerceptor.swift
//
// The PERCEIVE stage on device: run the Core ML model converted by the Python
// pipeline (../../pipeline) through the Vision framework and return structured
// detections. This mirrors `pipeline.reasoning.prompts.Detection` so the two
// halves of the project speak the same shape.

import CoreML
import Vision
import CoreImage

/// One structured observation from the vision model. Mirrors the Python
/// `Detection` dataclass so the reasoning prompt is identical across platforms.
struct Detection: Identifiable, Sendable {
    let id = UUID()
    let label: String
    let confidence: Float
}

enum PerceptionError: Error {
    case modelLoadFailed
    case inferenceFailed
}

/// Loads a compiled Core ML model and produces `Detection`s for an image.
final class VisionPerceptor: @unchecked Sendable {

    private let model: VNCoreMLModel

    /// - Parameter modelName: Name of the `.mlmodelc`/`.mlpackage` bundled in
    ///   the app. Produce it with:
    ///   `python -m pipeline convert mobilenet_v3_small Model.mlpackage --pretrained`
    init(modelName: String = "MobileNetV3") throws {
        // .all lets Core ML schedule onto CPU, GPU, and the Neural Engine
        // (preferring ANE when the op set fits), instead of being pinned to CPU.
        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard
            let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: modelName, withExtension: "mlpackage"),
            let mlModel = try? MLModel(contentsOf: url, configuration: config),
            let visionModel = try? VNCoreMLModel(for: mlModel)
        else {
            throw PerceptionError.modelLoadFailed
        }
        self.model = visionModel
    }

    /// Run the model and return the top-`k` detections, sorted by confidence.
    func perceive(_ image: CGImage, topK: Int = 5) async throws -> [Detection] {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if error != nil {
                    continuation.resume(throwing: PerceptionError.inferenceFailed)
                    return
                }
                let results = (request.results as? [VNClassificationObservation]) ?? []
                let detections = results
                    .sorted { $0.confidence > $1.confidence }
                    .prefix(topK)
                    .map { Detection(label: $0.identifier, confidence: $0.confidence) }
                continuation.resume(returning: Array(detections))
            }
            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: PerceptionError.inferenceFailed)
            }
        }
    }
}
