# `app/` — the on-device Swift app

A SwiftUI reference app that runs the full **Perceive → Reason** flow on device.
**Compile this on a Mac** — Swift + Core ML + Foundation Models are
macOS/Apple-silicon only and are not part of this repo's Linux CI.

## Requirements

- **Xcode 27** (iOS 27 / macOS 27 SDKs) — for the WWDC 2026 Foundation Models APIs.
- An Apple-silicon device whose chip supports the on-device model, for the
  `OnDeviceReasoner` path. (`SystemLanguageModel.availability` guards this.)

## Build

```bash
brew install xcodegen
xcodegen generate          # reads project.yml, no binary .xcodeproj committed
open PerceiveReason.xcodeproj
```

## Bundle a model

Produce the Core ML model with the Python pipeline and add it to the target:

```bash
python -m pipeline convert mobilenet_v3_small \
    app/PerceiveReason/MobileNetV3.mlpackage --pretrained
```

Then add it under `resources:` in `project.yml` (a commented example is there)
and re-run `xcodegen generate`.

## File map

| File | Role |
|---|---|
| `PerceiveReasonApp.swift` | App entry point + the build note. |
| `ContentView.swift` | UI: task field, provider picker, run button, results. |
| `PerceiveReasonViewModel.swift` | Orchestration; depends only on `ReasoningProvider`. |
| `Perception/VisionPerceptor.swift` | Core ML + Vision → `[Detection]`. |
| `Reasoning/ReasoningProvider.swift` | The provider protocol + shared prompt builder. |
| `Reasoning/OnDeviceReasoner.swift` | Apple Foundation Models (default, on-device). |
| `Reasoning/ClaudeReasoner.swift` | Claude via the Messages API (the WWDC 2026 swap). |

## Provider swap

`PerceiveReasonViewModel.makeProvider()` returns either `OnDeviceReasoner()` or
`ClaudeReasoner()` based on the UI picker — the entire "swap models in one line"
idea, made literal. The view model never changes.

> **Security:** `ClaudeReasoner` reads `ANTHROPIC_API_KEY` from the environment /
> Info.plist for local dev only. Ship keys via Keychain or a server-side proxy —
> never embed them in a distributed binary.

## Note on the beta APIs

The Foundation Models calls (`SystemLanguageModel`, `LanguageModelSession`,
`session.respond(to:)`) target the iOS 27 / macOS 27 SDK and may shift during
the beta. If a symbol moved, check
[Apple's Foundation Models docs](https://developer.apple.com/documentation/FoundationModels)
and adjust — the architecture (protocol + two providers) does not change.
