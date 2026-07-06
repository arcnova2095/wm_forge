import os
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from imwatermark import WatermarkDecoder, WatermarkEncoder
import torch
from torchvision import transforms
from trustmark import TrustMark
from blind_watermark import WaterMark



# FILE PATHS
WATERMARKED_DIR = Path("watermarked_sources")
CLEAN_DIR = Path("clean_targets")
TEMP_OUT_DIR = Path("submission_temp")

WM_RANGES = {
    "WM_1": range(1, 26), "WM_2": range(26, 51), "WM_3": range(51, 76), "WM_4": range(76, 101),
    "WM_5": range(101, 126), "WM_6": range(126, 151), "WM_7": range(151, 176), "WM_8": range(176, 201),
}


#  HELPER FUNCTIONS
def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])

def mean_psnr(clean_dir: Path, out_dir: Path, targets: range) -> float:
    values = []
    for i in targets:
        clean_path = clean_dir / f"{i}.png"
        forged_path = out_dir / f"{i}.png"
        if not clean_path.exists() or not forged_path.exists(): continue
            
        clean = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.float32)
        forged = np.asarray(Image.open(forged_path).convert("RGB"), dtype=np.float32)
        mse = np.mean((clean - forged) ** 2)
        values.append(float("inf") if mse <= 1e-12 else 20.0 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(values)) if values else 0.0


# INVISIBLE-WATERMARK METHODS
def test_imwatermark(source_paths, method, n_bits=32):
    decoder = WatermarkDecoder("bits", n_bits)
    if method == "rivaGan":
        decoder.loadModel()
    
    bits = []
    for p in source_paths:
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        decoded_bits = np.asarray(decoder.decode(bgr, method), dtype=np.uint8)
        bits.append(decoded_bits)
        
    if not bits: return None, 0.0
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    
    return majority, agreement

def forge_imwatermark(targets, message_bits, method, out_dir):
    encoder = WatermarkEncoder()
    if method == "rivaGan":
        encoder.loadModel()
    
    encoder.set_watermark("bits", message_bits.tolist())
    for i in targets:
        bgr = cv2.imread(str(CLEAN_DIR / f"{i}.png"))
        if bgr is None: continue
        forged = encoder.encode(bgr, method)
        cv2.imwrite(str(out_dir / f"{i}.png"), forged)


# TRUSTMARK METHODS
def trustmark_raw_decode(tm, img):
    resized = img.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    stego = transforms.ToTensor()(resized).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    with torch.no_grad():
        secret_binaryarray = (tm.decoder.decoder(stego) > 0).cpu().numpy()
    return secret_binaryarray[0].astype(np.uint8)

def test_trustmark(source_paths, model_type="Q"):
    if TrustMark is None: return None, 0.0
    try:
        tm = TrustMark(verbose=False, model_type=model_type)
    except Exception as e:
        print(f"    [Error loading TrustMark {model_type}]: {e}")
        return None, 0.0

    bits = []
    for p in source_paths:
        img = Image.open(p).convert("RGB")
        bits.append(trustmark_raw_decode(tm, img))
        
    if not bits: return None, 0.0
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    
    return majority, agreement

def forge_trustmark(targets, message_bits, model_type, out_dir):
    tm = TrustMark(verbose=False, model_type=model_type)
    secret_str = "".join(str(b) for b in message_bits.tolist())
    tm.use_ECC = False
    
    for i in targets:
        img = Image.open(CLEAN_DIR / f"{i}.png").convert("RGB")
        forged = tm.encode(img, secret_str, MODE="binary")
        forged.save(out_dir / f"{i}.png")


#BLIND_WATERMARK METHODS
def test_blind_wm(source_paths, n_bits=32):
    if WaterMark is None: return None, 0.0
    bits = []
    for p in source_paths:
        try:
            bwm = WaterMark(password_wm=1, password_img=1, mode='common')
            # Extract raw float values
            raw = bwm.extract(filename=str(p), wm_shape=(1, n_bits), mode='bit')
            # Convert to binary
            binary = (raw >= 0.5).astype(np.uint8).flatten()
            bits.append(binary)
        except Exception as e:
            continue
            
    if not bits: return None, 0.0
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    
    return majority, agreement

def forge_blind_wm(targets, message_bits, out_dir):
    for i in targets:
        source_img = str(CLEAN_DIR / f"{i}.png")
        out_img = str(out_dir / f"{i}.png")
        
        bwm = WaterMark(password_img=1, password_wm=1)
        bwm.read_img(source_img)
  
        wm_data = [bool(b) for b in message_bits]
        bwm.read_wm(wm_data, mode='bit')
        bwm.embed(out_img)


# MAIN AUTO-CRACK & FORGE LOOP
def main():
    
    TEST_BATTERY = [
        ("imwatermark: dwtDct",    test_imwatermark, forge_imwatermark, "dwtDct"),
        ("imwatermark: dwtDctSvd", test_imwatermark, forge_imwatermark, "dwtDctSvd"),
        ("imwatermark: rivaGan",   test_imwatermark, forge_imwatermark, "rivaGan"),
        ("TrustMark: Q",           test_trustmark,   forge_trustmark,   "Q"),
        ("TrustMark: P",           test_trustmark,   forge_trustmark,   "P"),
        ("TrustMark: C",           test_trustmark,   forge_trustmark,   "C"),
        ("blind_watermark: 32b",   test_blind_wm,    forge_blind_wm,    32),
        ("blind_watermark: 64b",   test_blind_wm,    forge_blind_wm,    64),
        ("blind_watermark: 128b",  test_blind_wm,    forge_blind_wm,    128),
    ]

    for wm_name, targets in WM_RANGES.items():
        source_dir = WATERMARKED_DIR / wm_name
        source_paths = sorted(list(source_dir.glob("*.png")), key=numeric_key)
        
            
        print(f"Analyzing {wm_name}...")
        cracked = False
        
        # Run through the battery of tests
        for label, test_func, forge_func, param in TEST_BATTERY:
            msg, agreement = test_func(source_paths, param)
            
            if msg is not None and agreement > 0.85:
                print(f"{wm_name} uses '{label}'! (Bit Agreement: {agreement*100:})")
                
                
                # Execute the specific forgery function
                if "blind_watermark" in label:
                    forge_func(targets, msg, TEMP_OUT_DIR)
                else:
                    forge_func(targets, msg, param, TEMP_OUT_DIR)
                
                psnr = mean_psnr(CLEAN_DIR, TEMP_OUT_DIR, targets)
                print(f"Mean PSNR vs clean targets: {psnr:.2f} dB")
                
                cracked = True
                break
                
        if not cracked:
            print(f"Could not crack {wm_name} ")
        print("-" * 50)



if __name__ == "__main__":
    main()
