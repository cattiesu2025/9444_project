"""Download the exact ImageNet initialization linked by the PMT authors."""
import hashlib
import json
from pathlib import Path
import urllib.request

URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"


def main():
    target = Path(__file__).resolve().parents[1] / "checkpoints" / URL.rsplit("/", 1)[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".download")
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith("80ecf9dd"):
        raise RuntimeError(f"Pretrained hash does not match the published filename: {digest}")
    record = {"url": URL, "sha256": digest, "bytes": target.stat().st_size}
    target.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
