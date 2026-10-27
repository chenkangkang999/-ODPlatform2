"""FishScope title wrapper; the shared platform code and YAML stay unchanged."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps/platform/src"))


def main():
    parser = argparse.ArgumentParser(description="FishScope Gradio image demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--weight", default=None)
    args = parser.parse_args()
    from od_platform.visualization.gradio_app import build_app, build_setup

    overrides = {"title": "渔见 FishScope · 淡水鱼图像检测演示",
                 "server_name": "127.0.0.1", "server_port": args.port,
                 "share": False, "inbrowser": False}
    if args.weight:
        overrides["weight"] = args.weight
    # Use the engine's public configuration merge so the title source is recorded.
    setup = build_setup(overrides)
    app = build_app(setup=setup)
    app.launch(server_name=setup.demo_config.server_name,
               server_port=setup.demo_config.server_port,
               share=False, inbrowser=False, show_error=True)


if __name__ == "__main__":
    main()
