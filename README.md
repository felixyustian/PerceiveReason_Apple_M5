# PerceiveReason

**A provider-agnostic, on-device "Perceive → Reason" pipeline for the Apple ML stack.**
Core ML handles perception on the Neural Engine; a swappable `LanguageModel`
(Apple's on-device Foundation Models *or* Claude) handles reasoning — following
the model-abstraction pattern Apple shipped at **WWDC 2026 (June 8, 2026)**.

> Rename freely — `PerceiveReason` is a working title, not a brand.

---

## Why this exists

Most "AI on Apple" demos are either a single Core ML classifier (perception with
no reasoning) or a chat wrapper (reasoning with no perception). This repo wires
the two stages together the way a real on-device feature does:

```
            ┌──────────────────────┐        ┌──────────────────────────────┐
  image ──▶ │  PERCEIVE            │ ──────▶│  REASON                      │──▶ assessment
            │  Core ML + Vision    │  dets  │  LanguageModel (swappable)   │
            │  (Neural Engine)     │        │  Apple on-device  | Claude   │
            └──────────────────────┘        └──────────────────────────────┘
```

The reasoning stage never names a vendor. You write the session logic once and
choose the provider at the edge — on-device for free/offline/private inference,
Claude for the hardest judgement calls — which is exactly the
[WWDC 2026 `LanguageModel` protocol](docs/WWDC2026.md) made concrete.

---

## What is verified vs. what you compile on a Mac

This repo is honest about its boundary, because the Apple ML runtime is
macOS/Apple-silicon only:

| Component | Language | Runs in CI (Linux)? | Notes |
|---|---|---|---|
| `pipeline/coreml_convert` — convert + compress | Python | ✅ Yes | `coremltools` converts and quantises off-device. |
| `pipeline/reasoning` — provider abstraction | Python | ✅ Yes | Offline `MockProvider`; Claude provider needs a key. |
| Model **prediction** on the converted package | — | ❌ No | Core ML's runtime ships with macOS only. |
| `app/` — SwiftUI + Vision + Foundation Models | Swift | ❌ No | Compile in **Xcode 27**, iOS/macOS 27. |

CI verifies *structure and compression*, not on-device behaviour. That catches
the vast majority of conversion regressions before they reach a device. The
Swift app is documented and idiomatic but deliberately not CI-built.

---

## Quickstart (Python pipeline — runs anywhere)

```bash
pip install -r requirements.txt

# Convert MobileNetV3 → Core ML, then int8-quantise, then assess detections.
python -m pipeline demo
```

Real output from this pipeline (random-weight MobileNetV3, measured):

```
== Perceive: convert MobileNetV3 -> Core ML ==
{'model_type': 'mlProgram', 'inputs': ['image'], 'outputs': ['logits'], 'size_mb': 5.167}
== Optimize: int8 weight quantisation ==
compression ratio: 1.93x          # 5.17 MB → 2.68 MB
== Reason: assess detections ==
[mock] Primary observation: hard_hat (confidence 0.94, high). Evidence is strong enough to act on.
```

Individual steps:

```bash
python -m pipeline convert mobilenet_v3_small Model.mlpackage --pretrained
python -m pipeline optimize Model.mlpackage Model_int8.mlpackage --dtype int8
python -m pipeline inspect Model_int8.mlpackage
echo '[{"label":"hard_hat","confidence":0.94}]' > dets.json
python -m pipeline reason dets.json --task "Assess PPE compliance." --provider mock
# Route reasoning to Claude instead (needs a key):
export ANTHROPIC_API_KEY=sk-...
python -m pipeline reason dets.json --task "Assess PPE compliance." --provider claude
```

## Building the app (macOS)

```bash
brew install xcodegen
cd app && xcodegen generate && open PerceiveReason.xcodeproj
```

See [`app/README.md`](app/README.md) and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Repository map

```
perceive-reason/
├── pipeline/                 # Python — Perceive (Core ML) + Reason (LanguageModel)
│   ├── coreml_convert/       #   convert.py · optimize.py · inspect_model.py
│   ├── reasoning/            #   language_model.py · perceive_reason.py · prompts.py
│   │   └── providers/        #   claude_provider.py · mock_provider.py
│   ├── examples/run_demo.py  #   end-to-end offline demo
│   └── cli.py                #   `python -m pipeline ...`
├── app/                      # Swift — on-device reference app (compile on Mac)
│   └── PerceiveReason/       #   Perception/ · Reasoning/ · views + view model
├── tests/                    # pytest — conversion, compression, reasoning
├── docs/                     # ARCHITECTURE · WWDC2026 · CONVERSION · DEPLOYMENT
└── .github/workflows/ci.yml  # runs the pipeline + tests on Linux
```

Each directory has its own README; each source file documents its role at the top.

---

## Tests

```bash
pip install pytest && pytest -q     # 8 tests, no network or API key required
```

## License

MIT — see [LICENSE](LICENSE).

*Implementation Copyright:** © 2026 [Felix Yustian Setiono](https://linkedin.com/in/felixsetiono). The entire system architecture, API source code, and experimental analysis documents within this repository are the original intellectual property of the author.
