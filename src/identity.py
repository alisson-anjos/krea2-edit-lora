from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.dataset import EditExample
from src.latent_cache import cache_key


ARCFACE_REPO = "py-feat/arcface_r50"
ARCFACE_FILE = "arcface_r50.safetensors"
IDENTITY_CACHE_SCHEMA = 1


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)


class IResNetBlock(nn.Module):
    """Fused-BN InsightFace residual block used by the converted ArcFace weights."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv1 = _conv3x3(in_channels, out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=True)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.conv2(self.prelu(self.conv1(self.bn1(x)))) + identity


class ArcFaceR50(nn.Module):
    """Differentiable ArcFace IResNet-50 accepting RGB face crops in [0, 1].

    The fused architecture and checkpoint conversion are from py-feat (MIT).
    The downloaded pretrained weights retain InsightFace's non-commercial
    research license; see the project README before enabling this feature.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _conv3x3(3, 64)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._stage(64, 64, 3)
        self.layer2 = self._stage(64, 128, 4)
        self.layer3 = self._stage(128, 256, 14)
        self.layer4 = self._stage(256, 512, 3)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(0.0, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-5, affine=True)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        return nn.Sequential(
            IResNetBlock(in_channels, out_channels, stride=2),
            *(IResNetBlock(out_channels, out_channels) for _ in range(1, blocks)),
        )

    def forward(self, face_crops: torch.Tensor) -> torch.Tensor:
        if face_crops.shape[-2:] != (112, 112):
            face_crops = F.interpolate(face_crops, (112, 112), mode="bilinear", align_corners=False)
        x = face_crops * 2.0 - 1.0
        x = self.prelu(self.conv1(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.features(self.fc(self.dropout(self.bn2(x)).flatten(1)))
        return x


def load_arcface(device: torch.device | str, weights: str | Path | None = None) -> ArcFaceR50:
    path = Path(weights) if weights else Path(hf_hub_download(ARCFACE_REPO, ARCFACE_FILE))
    model = ArcFaceR50()
    missing, unexpected = model.load_state_dict(load_file(str(path)), strict=False)
    missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"ArcFace checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def aligned_face_grid(
    inverse_affine: torch.Tensor,
    source_height: int,
    source_width: int,
    output_size: int = 112,
) -> torch.Tensor:
    """Build a grid_sample grid from a crop-to-source pixel affine matrix."""
    batch = inverse_affine.shape[0]
    coords = torch.linspace(0, output_size - 1, output_size, device=inverse_affine.device, dtype=inverse_affine.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    homogeneous = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(1, output_size * output_size, 3)
    source = homogeneous.expand(batch, -1, -1) @ inverse_affine.transpose(1, 2)
    source_x = source[..., 0] * (2.0 / max(1, source_width - 1)) - 1.0
    source_y = source[..., 1] * (2.0 / max(1, source_height - 1)) - 1.0
    return torch.stack((source_x, source_y), dim=-1).reshape(batch, output_size, output_size, 2)


def extract_aligned_faces(images: torch.Tensor, inverse_affine: torch.Tensor) -> torch.Tensor:
    grid = aligned_face_grid(inverse_affine, images.shape[-2], images.shape[-1])
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)


def crop_identity_latent(
    latent: torch.Tensor,
    inverse_affine: torch.Tensor,
    spatial_scale: int = 8,
    padding_pixels: int = 128,
    minimum_latent_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a padded face ROI before VAE decode and adjust its alignment matrix."""
    if latent.shape[0] != 1 or inverse_affine.shape[0] != 1:
        raise ValueError("crop_identity_latent handles one sample at a time")
    corners = torch.tensor(
        [[0.0, 0.0, 1.0], [111.0, 0.0, 1.0], [0.0, 111.0, 1.0], [111.0, 111.0, 1.0]],
        device=inverse_affine.device,
        dtype=inverse_affine.dtype,
    )
    source = corners @ inverse_affine[0].transpose(0, 1)
    x0 = int(torch.floor((source[:, 0].min() - padding_pixels) / spatial_scale).item())
    x1 = int(torch.ceil((source[:, 0].max() + padding_pixels) / spatial_scale).item())
    y0 = int(torch.floor((source[:, 1].min() - padding_pixels) / spatial_scale).item())
    y1 = int(torch.ceil((source[:, 1].max() + padding_pixels) / spatial_scale).item())
    height, width = latent.shape[-2:]

    def expand(start: int, end: int, limit: int) -> tuple[int, int]:
        start, end = max(0, start), min(limit, end)
        missing = max(0, min(minimum_latent_size, limit) - (end - start))
        start -= missing // 2
        end += missing - missing // 2
        if start < 0:
            end, start = min(limit, end - start), 0
        if end > limit:
            start, end = max(0, start - (end - limit)), limit
        return start, end

    x0, x1 = expand(x0, x1, width)
    y0, y1 = expand(y0, y1, height)
    adjusted = inverse_affine.clone()
    adjusted[:, 0, 2] -= x0 * spatial_scale
    adjusted[:, 1, 2] -= y0 * spatial_scale
    return latent[:, :, y0:y1, x0:x1], adjusted


def arcface_identity_loss(
    model: ArcFaceR50,
    predicted_pixels: torch.Tensor,
    inverse_affine: torch.Tensor,
    reference_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    faces = extract_aligned_faces(predicted_pixels.float().clamp(0, 1), inverse_affine.float())
    predicted = F.normalize(model(faces), dim=-1)
    references = F.normalize(reference_embeddings.float(), dim=-1)
    similarity = F.cosine_similarity(predicted, references, dim=-1)
    return (1.0 - similarity).mean(), similarity.mean()


def identity_cache_file(cache_dir: str | Path, item: EditExample) -> Path:
    return Path(cache_dir) / f"{cache_key(item)}.safetensors"


def identity_cache_signature(manifest: str | Path, width: int, height: int, detector_model: str) -> dict:
    manifest_path = Path(manifest).resolve()
    return {
        "schema": IDENTITY_CACHE_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "width": int(width),
        "height": int(height),
        "detector_model": detector_model,
        "embedding_model": f"{ARCFACE_REPO}/{ARCFACE_FILE}",
    }


def verify_identity_cache(cache_dir: str | Path, manifest: str | Path, width: int, height: int) -> None:
    metadata_file = Path(cache_dir) / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected = identity_cache_signature(manifest, width, height, metadata["detector_model"])
    if metadata != expected:
        raise ValueError("Identity cache does not match the manifest or current resolution. Rebuild it.")
