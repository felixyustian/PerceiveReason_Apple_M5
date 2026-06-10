# Architecture

PerceiveReason is two stages behind one provider abstraction.

```
            PERCEIVE                              REASON
  ┌───────────────────────────┐      ┌────────────────────────────────────┐
  │ image (CGImage / pixels)  │      │ LanguageModel (protocol)            │
  │   │                       │      │   ├─ Apple on-device (Foundation    │
  │   ▼                       │      │   │   Models) — default, free,      │
  │ Core ML model (.mlpackage)│      │   │   offline, private              │
  │   run via Vision          │ dets │   └─ Claude (Messages API) —        │
  │   on the Neural Engine     │ ───▶ │       harder judgement / ceiling   │
  │   │                       │      │           │                        │
  │   ▼                       │      │           ▼                        │
  │ [Detection(label, conf)]  │      │ calibrated assessment + next step  │
  └───────────────────────────┘      └────────────────────────────────────┘
        Python: coreml_convert              Python: reasoning
        Swift:  VisionPerceptor             Swift:  OnDeviceReasoner / ClaudeReasoner
```

## Design principles

1. **One shared shape across languages.** The Python `Detection` dataclass and
   the Swift `Detection` struct are identical (`label`, `confidence`), and both
   sides build the same prompt. A reviewer can read either half and understand
   the other.

2. **The orchestrator never names a vendor.** `reason_over_detections` (Python)
   and `PerceiveReasonViewModel` (Swift) depend only on the `LanguageModel` /
   `ReasoningProvider` interface. Swapping providers is one line; adding a new
   one is a new file. This is the WWDC 2026 abstraction, applied.

3. **Perception is compressed before it ships.** Conversion is always followed
   by an explicit optimisation step, because download size and memory pressure
   are first-class constraints on device — the same discipline as a
   TensorRT/edge export pipeline.

4. **Honest verification boundary.** Anything requiring the Core ML runtime or a
   Swift toolchain is documented but not claimed as CI-tested. CI asserts the
   things that can be asserted off-device (structure, I/O, compression).

## Why "perceive then reason" instead of one multimodal call

You *could* hand a raw image to a multimodal LLM and skip Core ML. Two reasons
not to, for an on-device product:

- **Cost and latency.** A small Core ML model on the Neural Engine is far
  cheaper and faster than a multimodal LLM call for the perception step.
- **Calibration and auditability.** Structured detections with confidences are
  inspectable and testable; a black-box multimodal answer is not. The reasoning
  step is then explicitly constrained to *only* use those detections, which is
  what `prompts.py` enforces in the prompt.

The architecture still allows a multimodal path — pass the image to the
reasoning provider too — but the default keeps perception and reasoning separable.
