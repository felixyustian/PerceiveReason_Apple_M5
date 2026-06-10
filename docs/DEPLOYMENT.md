# Deployment guide (on device)

Taking the pipeline output and running it inside the app, on a Mac.

## Prerequisites

- macOS with **Xcode 27** (iOS 27 / macOS 27 SDKs).
- Apple-silicon device or simulator that supports the on-device model (for the
  `OnDeviceReasoner` path). The app degrades gracefully via
  `SystemLanguageModel.availability` if it is unavailable.
- `xcodegen` (`brew install xcodegen`).

## Steps

1. **Produce and compress the model:**

   ```bash
   python -m pipeline convert mobilenet_v3_small \
       app/PerceiveReason/MobileNetV3.mlpackage --pretrained
   python -m pipeline optimize \
       app/PerceiveReason/MobileNetV3.mlpackage \
       app/PerceiveReason/MobileNetV3.mlpackage --dtype int8
   ```

2. **Bundle it:** add the `.mlpackage` to `resources:` in `app/project.yml`
   (commented example included), then:

   ```bash
   cd app && xcodegen generate && open PerceiveReason.xcodeproj
   ```

   Xcode compiles `.mlpackage` → `.mlmodelc` automatically;
   `VisionPerceptor` loads either extension.

3. **Wire up image input.** `ContentView` leaves a hook where you add a
   `PhotosPicker` or camera capture to set the `CGImage`. Usage-description
   Info.plist keys are already declared in `project.yml`.

4. **Choose the reasoning provider** at runtime via the picker:
   - **On-device (Apple):** free, offline, private. No key.
   - **Claude (cloud):** set `ANTHROPIC_API_KEY` (env or Info.plist) **for local
     dev only**. For distribution, route through a server-side proxy or store
     the key in the Keychain — never embed it in the shipped binary.

## Performance checks worth doing on device

CI cannot run these; do them in Xcode/Instruments on a real device:

- **Core ML latency** per inference, and which compute unit Core ML scheduled
  onto (CPU/GPU/Neural Engine). Compare `compute_units` settings.
- **Accuracy after quantisation** vs. the fp16 baseline on a held-out set.
- **App size** contribution of the bundled model (this is where the int8 step
  pays off).
- **On-device vs. Claude** reasoning latency and quality on representative
  detections, to decide your routing policy.
