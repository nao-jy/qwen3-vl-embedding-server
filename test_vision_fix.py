#!/usr/bin/env python3
"""
Quick smoke test for multimodal (image + text) embedding.
Edit TEST_FRAMES below to point at your own images before running.

Usage:
    python test_vision_fix.py

Requires the server to be running on localhost:8000.
"""
import requests
import base64

BASE_URL = "http://localhost:8000"

# --- Edit these paths to point at real images on your system ---
TEST_FRAMES = [
    "/path/to/your/image_001.png",
    "/path/to/your/image_002.png",
]
# ---------------------------------------------------------------


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


print("Testing multimodal embedding...")

for frame_path in TEST_FRAMES:
    print(f"\nTesting: {frame_path.split('/')[-1]}")

    try:
        b64 = encode_image(frame_path)
        payload = {
            "input": {
                "text": "What is happening in this image?",
                "image": {"type": "base64", "data": b64}
            }
        }

        resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            emb_len = len(data['data'][0]['embedding'])
            print(f"   SUCCESS! Embedding dim: {emb_len}")
        else:
            print(f"   FAILED: {resp.status_code} - {resp.text[:200]}")

    except FileNotFoundError:
        print(f"   SKIPPED: file not found — update TEST_FRAMES at the top of this script")
    except Exception as e:
        print(f"   ERROR: {e}")

print("\nDone.")
