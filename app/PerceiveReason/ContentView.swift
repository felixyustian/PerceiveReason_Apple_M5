// ContentView.swift
//
// Minimal UI to demonstrate the perceive->reason flow: pick an image, choose a
// reasoning provider, run, and read the assessment plus the raw detections.
// Image picking is intentionally left thin so the file stays focused on wiring.

import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var vm = PerceiveReasonViewModel()
    @State private var image: CGImage?
    @State private var pickerItem: PhotosPickerItem?
    @State private var showFileImporter = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Task") {
                    TextField("What should the model assess?", text: $vm.task,
                              axis: .vertical)
                    Picker("Reasoning", selection: $vm.providerChoice) {
                        ForEach(PerceiveReasonViewModel.ProviderChoice.allCases) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                }

                Section("Run") {
                    let pickerLabel = image == nil ? "Choose image" : "Change image"
                    PhotosPicker(
                        selection: $pickerItem,
                        matching: .images,
                        photoLibrary: .shared()
                    ) {
                        Text(pickerLabel)
                    }

                    Button("Browse files…") { showFileImporter = true }

                    Button {
                        guard let image else { return }
                        Task { await vm.run(on: image) }
                    } label: {
                        if vm.isRunning { ProgressView() }
                        else { Text("Perceive → Reason") }
                    }
                    .disabled(image == nil || vm.isRunning)
                }
                .onChange(of: pickerItem) { _, newItem in
                    guard let newItem else { return }
                    Task {
                        if let data = try? await newItem.loadTransferable(type: Data.self),
                           let source = CGImageSourceCreateWithData(data as CFData, nil),
                           let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) {
                            image = cgImage
                        }
                    }
                }
                .fileImporter(
                    isPresented: $showFileImporter,
                    allowedContentTypes: [.image, .movie],
                    allowsMultipleSelection: false
                ) { result in
                    guard case let .success(urls) = result, let url = urls.first
                    else { return }
                    Task {
                        if let cgImage = await loadCGImage(from: url) {
                            image = cgImage
                        } else {
                            vm.errorMessage = "Could not load image from file."
                        }
                    }
                }

                if let error = vm.errorMessage {
                    Section("Error") { Text(error).foregroundStyle(.red) }
                }

                if !vm.detections.isEmpty {
                    Section("Detections") {
                        ForEach(vm.detections) { d in
                            HStack {
                                Text(d.label)
                                Spacer()
                                Text(String(format: "%.2f", d.confidence))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                if !vm.assessment.isEmpty {
                    Section("Assessment") { Text(vm.assessment) }
                }
            }
            .navigationTitle("PerceiveReason")
        }
    }

    // Loads a CGImage from a user-picked file URL. Security-scoped access is
    // required for files outside the app sandbox (Files app, iCloud Drive).
    // Videos are reduced to a single representative frame so the existing
    // CGImage-based perception flow works unchanged.
    private func loadCGImage(from url: URL) async -> CGImage? {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let type = try? url.resourceValues(forKeys: [.contentTypeKey]).contentType
        if let type, type.conforms(to: .movie) {
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            return try? await generator.image(at: .zero).image
        }

        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return nil }
        return cgImage
    }
}
