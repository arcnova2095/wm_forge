import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from imwatermark import WatermarkDecoder, WatermarkEncoder
from trustmark import TrustMark

# --- CONFIGURATION & HARDCODED PATHS ---
WATERMARKED_DIR = Path("watermarked_sources")
CLEAN_DIR = Path("clean_targets")
TEMP_OUT_DIR = Path("submission_temp_forge")
ZIP_OUT = Path("submission.zip")

CATEGORIES: Tuple[Tuple[str, int, int], ...] = (
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
)
EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}


@dataclass(frozen=True)
class FingerprintConfig:
    residual_strength: float
    residual_sigmas: Tuple[float, ...]
    frequency_strength: float = 0.75
    freq_low: float = 0.04
    freq_high: float = 0.20
    aggregation: str = "median"
    channel_mode: str = "rgb"
    perturbation_clip: float = 14.0

# Configs for fingerprint classes
FINGERPRINT_CONFIGS = {
    "WM_3": FingerprintConfig(
        residual_strength=4.0,
        residual_sigmas=(10.0,),
        frequency_strength=0.75,
        freq_low=0.04,
        freq_high=0.20,
    ),
    "WM_4": FingerprintConfig(
        residual_strength=3.5,
        residual_sigmas=(0.5, 1.0),
        frequency_strength=0.50,
        freq_low=0.08,
        freq_high=0.30,
    ),
    "WM_5": FingerprintConfig(
        residual_strength=4.0,
        residual_sigmas=(1.0,),
        frequency_strength=0.50,
        freq_low=0.06,
        freq_high=0.25,
    ),
    "WM_6": FingerprintConfig(
        residual_strength=4.0,
        residual_sigmas=(6.0,),
        frequency_strength=0.0,
    ),
    "WM_7": FingerprintConfig(
        residual_strength=4.0,
        residual_sigmas=(8.0,),
        frequency_strength=0.75,
        freq_low=0.04,
        freq_high=0.20,
    ),
    "WM_8": FingerprintConfig(
        residual_strength=4.0,
        residual_sigmas=(10.0,),
        frequency_strength=0.0,
    ),
}

# ── Utility Helpers ───────────────────────────────────────────────────────────
def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])

def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)

def save_rgb(array: np.ndarray, path: Path) -> None:
    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)

def gaussian_blur_rgb(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)

def robust_channel_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])

def robust_masked_scale(array: np.ndarray, mask: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    selected = mask[:, :, 0] > 0.15
    if selected.sum() < 16:
        return robust_channel_scale(array, eps)
    vals = array[selected, :]
    med = np.median(vals, axis=0, keepdims=True)
    centered = vals - med
    mad = 1.4826 * np.median(np.abs(centered), axis=0, keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    scale = np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])
    return scale.reshape(1, 1, 3)

def normalize_full(fp: np.ndarray, strength: float, clip_value: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fp, dtype=np.float32)
    centered = fp - np.mean(fp, axis=(0, 1), keepdims=True)
    out = centered / robust_channel_scale(centered) * float(strength)
    return np.clip(out, -clip_value, clip_value).astype(np.float32)

def normalize_masked(fp: np.ndarray, mask: np.ndarray, strength: float, clip_value: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fp, dtype=np.float32)
    selected = mask[:, :, 0] > 0.15
    centered = fp.copy().astype(np.float32)
    if selected.sum() >= 16:
        centered -= np.median(centered[selected, :], axis=0).reshape(1, 1, 3)
    else:
        centered -= np.median(centered, axis=(0, 1), keepdims=True)
    out = centered / robust_masked_scale(centered, mask) * float(strength)
    return np.clip(out * mask, -clip_value, clip_value).astype(np.float32)

def smooth_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return (np.asarray(blurred, dtype=np.float32) / 255.0)[:, :, None]

def border_mask(height: int, width: int, ratio: float) -> np.ndarray:
    bw = max(4, int(round(min(height, width) * ratio)))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:bw, :] = 1
    mask[-bw:, :] = 1
    mask[:, :bw] = 1
    mask[:, -bw:] = 1
    return smooth_mask(mask, max(1.0, bw / 3.0)).astype(np.float32)

def make_edge_mask(h: int, w: int, width: int, mode: str, softness: int) -> np.ndarray:
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    left = x < width
    right = x >= (w - width)
    top = y < width
    bottom = y >= (h - width)
    if mode == "corners":
        corner_h = max(width * 3, 32)
        corner_w = max(width * 3, 32)
        bl = (x < corner_w) & (y >= h - corner_h)
        br = (x >= w - corner_w) & (y >= h - corner_h)
        tl = (x < corner_w) & (y < corner_h)
        tr = (x >= w - corner_w) & (y < corner_h)
        mask2d = bl | br | tl | tr
    else:
        mask2d = left | right | top | bottom
    mask = mask2d.astype(np.float32)
    if softness > 0:
        img = Image.fromarray(np.uint8(mask * 255), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(radius=float(softness)))
        mask = np.asarray(img, dtype=np.float32) / 255.0
    return mask[:, :, None].astype(np.float32)

def radial_bandpass_mask(height: int, width: int, low: float, high: float) -> np.ndarray:
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    return ((radius >= low) & (radius <= high)).astype(np.float32)[:, :, None]

# ── Signal Processing Estimators ──────────────────────────────────────────────
def highpass_residual(image: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    residuals = [image - gaussian_blur_rgb(image, sigma) for sigma in sigmas]
    return np.mean(np.stack(residuals, axis=0), axis=0)

def aggregate_stack(stack: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        return np.median(stack, axis=0)
    if method == "mean":
        return np.mean(stack, axis=0)
    if method == "trimmed_mean":
        if stack.shape[0] < 5:
            return np.mean(stack, axis=0)
        sorted_stack = np.sort(stack, axis=0)
        trim = max(1, int(round(0.1 * stack.shape[0])))
        return np.mean(sorted_stack[trim:-trim], axis=0)
    raise ValueError(f"Unknown aggregation: {method}")

def estimate_residual_fingerprint(
    source_paths: Sequence[Path], sigmas: Sequence[float], strength: float, aggregation: str
) -> np.ndarray:
    residuals = []
    for path in source_paths:
        image = read_rgb(path)
        res = highpass_residual(image, sigmas)
        res = res - np.median(res, axis=(0, 1), keepdims=True)
        residuals.append(res)
    fp = aggregate_stack(np.stack(residuals, axis=0), aggregation)
    return normalize_full(fp, strength=strength, clip_value=max(2.0, 3.0 * strength))

def bandpass_residual(image: np.ndarray, low: float, high: float) -> np.ndarray:
    height, width, _ = image.shape
    centered = image - np.mean(image, axis=(0, 1), keepdims=True)
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    filtered = np.fft.ifft2(
        spectrum * radial_bandpass_mask(height, width, low=low, high=high), axes=(0, 1)
    ).real.astype(np.float32)
    return filtered

def estimate_frequency_fingerprint(
    source_paths: Sequence[Path], low: float, high: float, strength: float, aggregation: str
) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(read_rgb(source_paths[0]), dtype=np.float32)

    residuals = []
    for path in source_paths:
        image = read_rgb(path)
        res = bandpass_residual(image, low=low, high=high)
        res = res - np.mean(res, axis=(0, 1), keepdims=True)
        res = res / robust_channel_scale(res)
        residuals.append(res)
    fp = aggregate_stack(np.stack(residuals, axis=0), aggregation)
    return normalize_full(fp, strength=strength, clip_value=max(2.0, 3.0 * strength))

# ── Processors per Category ───────────────────────────────────────────────────
def trustmark_raw_decode(tm: TrustMark, img: Image.Image) -> np.ndarray:
    resized = img.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    stego = transforms.ToTensor()(resized).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    with torch.no_grad():
        secret_binaryarray = (tm.decoder.decoder(stego) > 0).cpu().numpy()
    return secret_binaryarray[0].astype(np.uint8)

def process_wm7_trustmark(source_paths: Sequence[Path], targets: range, clean_dir: Path, out_dir: Path) -> None:
    print("WM_7: Decoding majority bits and encoding via trustmark (variant Q) ...")
    tm = TrustMark(verbose=False, model_type="Q")
    
    bits = []
    for p in source_paths:
        img = Image.open(p).convert("RGB")
        bits.append(trustmark_raw_decode(tm, img))
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    print(f"  recovered majority message (agreement: {agreement:.4f})")
    
    secret_str = "".join(str(b) for b in majority.tolist())
    tm.use_ECC = False
    for i in targets:
        img = Image.open(clean_dir / f"{i}.png").convert("RGB")
        forged = tm.encode(img, secret_str, MODE="binary")
        forged.save(out_dir / f"{i}.png")

def process_wm8_trustmark(source_paths: Sequence[Path], targets: range, clean_dir: Path, out_dir: Path) -> None:
    print("WM_8: Decoding majority bits and encoding via trustmark (variant P) ...")
    tm = TrustMark(verbose=False, model_type="P")
    
    bits = []
    for p in source_paths:
        img = Image.open(p).convert("RGB")
        bits.append(trustmark_raw_decode(tm, img))
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    print(f"  recovered majority message (agreement: {agreement:.4f})")
    
    secret_str = "".join(str(b) for b in majority.tolist())
    tm.use_ECC = False
    for i in targets:
        img = Image.open(clean_dir / f"{i}.png").convert("RGB")
        forged = tm.encode(img, secret_str, MODE="binary")
        forged.save(out_dir / f"{i}.png")

def process_wm1_dwtdct(source_paths: Sequence[Path], targets: range, clean_dir: Path, out_dir: Path) -> None:
    print("WM_1: Decoding majority bits and encoding via dwtDct (32 bits) ...")
    decoder = WatermarkDecoder("bits", 32)
    bits = []
    for p in source_paths:
        bgr = cv2.imread(str(p))
        bits.append(np.asarray(decoder.decode(bgr, "dwtDct"), dtype=np.uint8))
    majority = (np.stack(bits, axis=0).mean(axis=0) >= 0.5).astype(np.uint8)
    print(f"  recovered majority message: {majority.tolist()}")
    
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", majority.tolist())
    for i in targets:
        bgr = cv2.imread(str(clean_dir / f"{i}.png"))
        forged = encoder.encode(bgr, "dwtDct")
        cv2.imwrite(str(out_dir / f"{i}.png"), forged)

def process_wm2_rivagan(source_paths: Sequence[Path], targets: range, clean_dir: Path, out_dir: Path) -> None:
    print("WM_2: Decoding majority bits and encoding via rivaGan (32 bits) ...")
    decoder = WatermarkDecoder("bits", 32)
    encoder = WatermarkEncoder()
    decoder.loadModel()
    encoder.loadModel()
    
    bits = []
    for p in source_paths:
        bgr = cv2.imread(str(p))
        bits.append(np.asarray(decoder.decode(bgr, "rivaGan"), dtype=np.uint8))
    majority = (np.stack(bits, axis=0).mean(axis=0) >= 0.5).astype(np.uint8)
    print(f"  recovered majority message: {majority.tolist()}")
    
    encoder.set_watermark("bits", majority.tolist())
    for i in targets:
        bgr = cv2.imread(str(clean_dir / f"{i}.png"))
        forged = encoder.encode(bgr, "rivaGan")
        cv2.imwrite(str(out_dir / f"{i}.png"), forged)

def process_wm_generic(
    wm_name: str, source_paths: Sequence[Path], targets: range, clean_dir: Path, out_dir: Path
) -> None:
    cfg = FINGERPRINT_CONFIGS[wm_name]
    print(f"{wm_name}: Running residual + frequency fingerprint ...")
    residual_fp = estimate_residual_fingerprint(
        source_paths=source_paths,
        sigmas=cfg.residual_sigmas,
        strength=cfg.residual_strength,
        aggregation=cfg.aggregation,
    )
    frequency_fp = estimate_frequency_fingerprint(
        source_paths=source_paths,
        low=cfg.freq_low,
        high=cfg.freq_high,
        strength=cfg.frequency_strength,
        aggregation=cfg.aggregation,
    )
    
    perturbation = residual_fp + frequency_fp
    
    if cfg.channel_mode == "luma":
        y = (
            0.299 * perturbation[:, :, 0:1]
            + 0.587 * perturbation[:, :, 1:2]
            + 0.114 * perturbation[:, :, 2:3]
        )
        perturbation = np.repeat(y, 3, axis=2).astype(np.float32)
        
    perturbation = np.clip(perturbation, -cfg.perturbation_clip, cfg.perturbation_clip).astype(np.float32)
    
    for i in targets:
        target = read_rgb(clean_dir / f"{i}.png")
        save_rgb(target + perturbation, out_dir / f"{i}.png")

# ── Verification & ZIP Package ────────────────────────────────────────────────
def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output files mismatch. Missing={missing[:5]}, extra={extra[:5]}")
    
    if zip_out.exists():
        zip_out.unlink()
    
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 201):
            p = out_dir / f"{i}.png"
            zf.write(p, arcname=p.name)
            
    with zipfile.ZipFile(zip_out, "r") as zf:
        if set(zf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP validation failed: zip is not a flat 200-image directory")

# ── Execution Entrypoint ──────────────────────────────────────────────────────
def build_submission() -> None:
    if not CLEAN_DIR.exists() or not WATERMARKED_DIR.exists():
        raise FileNotFoundError(f"Missing dataset directories! Ensure '{CLEAN_DIR}' and '{WATERMARKED_DIR}' exist.")

    if TEMP_OUT_DIR.exists():
        shutil.rmtree(TEMP_OUT_DIR)
    TEMP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root      : {CLEAN_DIR.parent}")
    print(f"Output directory  : {TEMP_OUT_DIR}")
    print(f"ZIP output        : {ZIP_OUT}\n")

    for wm_name, start, stop in CATEGORIES:
        targets = range(start, stop + 1)
        wm_src_dir = WATERMARKED_DIR / wm_name
        source_paths = sorted(wm_src_dir.glob("*.png"), key=numeric_key)
        
        if len(source_paths) != len(targets):
            raise RuntimeError(
                f"Expected {len(targets)} source images for {wm_name}, found {len(source_paths)}"
            )

        if wm_name == "WM_1":
            process_wm1_dwtdct(source_paths, targets, CLEAN_DIR, TEMP_OUT_DIR)
        elif wm_name == "WM_2":
            process_wm2_rivagan(source_paths, targets, CLEAN_DIR, TEMP_OUT_DIR)
        elif wm_name == "WM_7":
            process_wm7_trustmark(source_paths, targets, CLEAN_DIR, TEMP_OUT_DIR)
        elif wm_name == "WM_8":
            process_wm8_trustmark(source_paths, targets, CLEAN_DIR, TEMP_OUT_DIR)
        else:
            process_wm_generic(wm_name, source_paths, targets, CLEAN_DIR, TEMP_OUT_DIR)
            
        print(f"Finished {wm_name}.\n")

    validate_and_zip(TEMP_OUT_DIR, ZIP_OUT)
    print(f"Success! Generated {ZIP_OUT} containing 200 forged target images.")


def main() -> int:
    try:
        build_submission()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())