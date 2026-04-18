"""
Standalone model server that loads a patch encoder and exposes it via HTTP.

Usage:
    python load_model.py --gpu 0 --port 3001 --patch_encoder conch_v15
    python load_model.py --gpu 0 --port 3001 --patch_encoder plip --patch_encoder_ckpt_path /path/to/ckpt
"""

import argparse
import base64
import io
import traceback

import numpy as np
import torch
import torchvision.transforms.functional as F_vis
from flask import Flask, jsonify, request

from trident.patch_encoder_models.load import encoder_factory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a Trident patch encoder over HTTP."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
    parser.add_argument(
        "--port", type=int, default=3001, help="Port to bind the server to."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to."
    )
    parser.add_argument(
        "--patch_encoder",
        type=str,
        default="conch_v15",
        help="Name of the patch encoder.",
    )
    parser.add_argument(
        "--patch_encoder_ckpt_path",
        type=str,
        default=None,
        help="Optional local path to a patch encoder checkpoint.",
    )
    return parser.parse_args()


app = Flask(__name__)

_encoder = None
_device = None
_eval_transforms = None


def _apply_transforms(images_np: np.ndarray) -> torch.Tensor:
    transformed = []
    for i in range(images_np.shape[0]):
        img_tensor = torch.from_numpy(images_np[i])
        pil_img = F_vis.to_pil_image(img_tensor)
        transformed.append(_eval_transforms(pil_img))
    return torch.stack(transformed)


def _encode_batch(images_np: np.ndarray) -> np.ndarray:
    images_tensor = _apply_transforms(images_np).to(_device)
    precision = getattr(_encoder, "precision", torch.float32)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=_device.type if _device.type == "cuda" else "cpu",
            dtype=precision,
            enabled=(precision != torch.float32),
        ),
    ):
        features = _encoder(images_tensor)
    return features.cpu().numpy()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/model_info", methods=["GET"])
def model_info():
    info = {
        "model_name": getattr(_encoder, "enc_name", "unknown"),
        "precision": str(getattr(_encoder, "precision", torch.float32)),
    }
    return jsonify(info)


@app.route("/encode", methods=["POST"])
def encode():
    try:
        payload = request.get_json(force=True)
        images_b64 = payload["images"]
        images_list = []
        for b64_str in images_b64:
            raw = base64.b64decode(b64_str)
            buf = io.BytesIO(raw)
            arr = np.load(buf, allow_pickle=False)
            images_list.append(arr)
        images_np = np.stack(images_list, axis=0).astype(np.float32)

        features = _encode_batch(images_np)

        buf = io.BytesIO()
        np.save(buf, features)
        features_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return jsonify({"features": features_b64, "shape": list(features.shape)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/encode_raw", methods=["POST"])
def encode_raw():
    try:
        raw = request.get_data()
        buf = io.BytesIO(raw)
        images_np = np.load(buf, allow_pickle=False).astype(np.float32)

        features = _encode_batch(images_np)

        out_buf = io.BytesIO()
        np.save(out_buf, features)
        return out_buf.getvalue(), 200, {"Content-Type": "application/octet-stream"}
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def main():
    global _encoder, _device, _eval_transforms

    args = parse_args()

    _device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading patch encoder: {args.patch_encoder}")
    _encoder = encoder_factory(
        args.patch_encoder, weights_path=args.patch_encoder_ckpt_path
    )
    _eval_transforms = _encoder.eval_transforms
    _encoder.to(_device)
    _encoder.eval()
    print(f"Model loaded on {_device}")

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
