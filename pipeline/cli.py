"""Command-line entry point: ``python -m pipeline ...``

Subcommands:
    convert   Convert a torchvision model to Core ML.
    optimize  Compress a converted .mlpackage's weights.
    inspect   Print a structural summary of an .mlpackage (JSON).
    reason    Run the Reason stage over detections from a JSON file.
    demo      End-to-end convert -> optimize -> inspect -> reason (offline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_convert(args: argparse.Namespace) -> int:
    from .coreml_convert import convert, inspect_model
    from .config import ConvertConfig

    cfg = ConvertConfig(image_size=args.size, deployment_target=args.target,
                        compute_units=args.compute_units)
    convert.convert_torchvision(args.arch, args.out, pretrained=args.pretrained,
                                config=cfg)
    print(json.dumps(inspect_model.summarize(args.out).as_dict(), indent=2))
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    import coremltools as ct
    from .coreml_convert import optimize, inspect_model
    from .config import OptimizeConfig

    mlmodel = ct.models.MLModel(args.model)
    cfg = OptimizeConfig(dtype=args.dtype, nbits_palette=args.palette)
    optimize.compress(mlmodel, args.out, config=cfg)
    ratio = inspect_model.compression_ratio(args.model, args.out)
    print(f"compression ratio (orig/new): {ratio}x")
    print(json.dumps(inspect_model.summarize(args.out).as_dict(), indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .coreml_convert import inspect_model
    print(json.dumps(inspect_model.summarize(args.model).as_dict(), indent=2))
    return 0


def _build_model(provider: str, model_id: str):
    from .reasoning.providers import ClaudeProvider, MockProvider
    return MockProvider() if provider == "mock" else ClaudeProvider(model=model_id)


def _cmd_reason(args: argparse.Namespace) -> int:
    from .reasoning.perceive_reason import reason_over_detections
    from .reasoning.prompts import Detection

    data = json.loads(Path(args.detections).read_text())
    dets = [Detection(label=d["label"], confidence=d["confidence"],
                      box=tuple(d["box"]) if d.get("box") else None) for d in data]
    resp = reason_over_detections(dets, args.task,
                                  _build_model(args.provider, args.model))
    print(f"[{resp.provider}] {resp.text}")
    if resp.usage:
        print(f"usage: {resp.usage}", file=sys.stderr)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .examples.run_demo import main as demo_main
    return demo_main(provider=args.provider)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline",
                                description="PerceiveReason pipeline CLI")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="Convert a torchvision model to Core ML")
    c.add_argument("arch", help="e.g. mobilenet_v3_small, resnet18")
    c.add_argument("out", help="output .mlpackage path")
    c.add_argument("--size", type=int, default=224)
    c.add_argument("--target", default="iOS18")
    c.add_argument("--compute-units", default="ALL")
    c.add_argument("--pretrained", action="store_true")
    c.set_defaults(func=_cmd_convert)

    o = sub.add_parser("optimize", help="Compress a .mlpackage")
    o.add_argument("model", help="input .mlpackage")
    o.add_argument("out", help="output .mlpackage")
    o.add_argument("--dtype", default="int8", choices=["int8", "int4"])
    o.add_argument("--palette", type=int, default=None,
                   help="use k-means palettisation at this bit width instead")
    o.set_defaults(func=_cmd_optimize)

    i = sub.add_parser("inspect", help="Summarise a .mlpackage")
    i.add_argument("model")
    i.set_defaults(func=_cmd_inspect)

    r = sub.add_parser("reason", help="Run the Reason stage over detections JSON")
    r.add_argument("detections", help="JSON list of {label,confidence,box?}")
    r.add_argument("--task", default="Assess the scene.")
    r.add_argument("--provider", default="mock", choices=["mock", "claude"])
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.set_defaults(func=_cmd_reason)

    d = sub.add_parser("demo", help="Run the full offline demo")
    d.add_argument("--provider", default="mock", choices=["mock", "claude"])
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
