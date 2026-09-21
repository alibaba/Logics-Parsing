"""Download Logics-Parsing-V3 from Hugging Face or ModelScope."""

from argparse import ArgumentParser
from pathlib import Path


MODEL_IDS = {
    "huggingface": "Logics-MLLM/Logics-Parsing-V3",
    "modelscope": "Alibaba-DT/Logics-Parsing-V3",
}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "weights" / "Logics-Parsing-V3"


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", "--type", "-t", dest="source",
        choices=("hf", "huggingface", "modelscope"), default="huggingface",
        help="Download provider (default: huggingface). --type/-t remain supported.",
    )
    parser.add_argument(
        "--local-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help="Destination (default: weights/Logics-Parsing-V3 next to this script).",
    )
    parser.add_argument(
        "--revision", help="Optional provider-specific branch, tag, or commit.",
    )
    args = parser.parse_args(argv)
    source = "huggingface" if args.source == "hf" else args.source
    package = "huggingface_hub" if source == "huggingface" else "modelscope"
    try:
        if source == "huggingface":
            from huggingface_hub import snapshot_download
        else:
            from modelscope import snapshot_download
    except ImportError as exc:
        parser.exit(
            1, f"Cannot import {package}: {exc}\n"
            f"Install it with: python -m pip install -U {package}\n",
        )

    model_dir = args.local_dir.expanduser().resolve()
    kwargs = {"local_dir": str(model_dir)}
    if args.revision is not None:
        kwargs["revision"] = args.revision
    model_id = MODEL_IDS[source]
    print(f"Downloading {model_id} from {source} to {model_dir}")
    if source == "huggingface":
        snapshot_download(repo_id=model_id, **kwargs)
    else:
        snapshot_download(model_id=model_id, **kwargs)
    print(f"Model downloaded to {model_dir}")


if __name__ == "__main__":
    main()
