"""
Test suite for Qwen3-VL-Embedding Server multimodal support
Tests text-only, base64 images, URL images, and local path images
"""

import os
import sys
import base64
import json
import time
import io
import requests
from PIL import Image, ImageDraw

# Configuration
BASE_URL = "http://localhost:8000"
EMBEDDINGS_ENDPOINT = f"{BASE_URL}/v1/embeddings"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
TEST_IMAGE_PATH = "/tmp/test_multimodal_image.png"
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"


def create_test_image(path: str):
    """Generate a simple test image with text"""
    print(f"Creating test image at {path}...")
    
    # Create a simple colored image with some shapes
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
    draw.ellipse([200, 50, 350, 200], fill='green', outline='black', width=2)
    draw.polygon([(100, 200), (200, 280), (300, 200)], fill='yellow', outline='black', width=2)
    
    # Save the image
    img.save(path)
    print(f"Test image saved: {path}")
    return path


def encode_image_to_base64(path: str) -> str:
    """Encode image file to base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Device: {data.get('device')}")
        print(f"Model: {data.get('model')}")
        print(f"Multimodal Supported: {data.get('multimodal_supported')}")
        
        assert data['status'] == 'ok', "Health check failed"
        assert data['multimodal_supported'] == True, "Multimodal not enabled"
        print("✅ Health check PASSED")
        return True
    except Exception as e:
        print(f"❌ Health check FAILED: {e}")
        return False


def test_text_only_single():
    """Test single text embedding (backward compatibility)"""
    print("\n" + "="*60)
    print("TEST 2: Text-Only Single Embedding")
    print("="*60)
    
    payload = {
        "input": "A beautiful sunset over the ocean",
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response object type: {data.get('object')}")
        print(f"Model: {data.get('model')}")
        print(f"Number of embeddings: {len(data.get('data', []))}")
        
        embedding = data['data'][0]['embedding']
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        assert len(embedding) > 0, "Empty embedding"
        print("✅ Text-only single PASSED")
        return True
    except Exception as e:
        print(f"❌ Text-only single FAILED: {e}")
        return False


def test_text_only_batch():
    """Test batch text embeddings (backward compatibility)"""
    print("\n" + "="*60)
    print("TEST 3: Text-Only Batch Embeddings")
    print("="*60)
    
    payload = {
        "input": [
            "A cat sitting on a couch",
            "A dog playing in the park",
            "A bird flying in the sky"
        ],
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"Number of embeddings: {len(data.get('data', []))}")
        
        for i, item in enumerate(data['data']):
            print(f"  Embedding {i}: dimension={len(item['embedding'])}, index={item['index']}")
            assert len(item['embedding']) > 0, f"Empty embedding at index {i}"
        
        print("✅ Text-only batch PASSED")
        return True
    except Exception as e:
        print(f"❌ Text-only batch FAILED: {e}")
        return False


def test_multimodal_base64():
    """Test multimodal embedding with base64 image"""
    print("\n" + "="*60)
    print("TEST 4: Multimodal - Base64 Image")
    print("="*60)
    
    # Ensure test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        create_test_image(TEST_IMAGE_PATH)
    
    base64_image = encode_image_to_base64(TEST_IMAGE_PATH)
    print(f"Base64 image length: {len(base64_image)} chars")
    
    payload = {
        "input": {
            "text": "What shapes and colors are in this image?",
            "image": {
                "type": "base64",
                "data": base64_image
            }
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response status: {response.status_code}")
        print(f"Number of embeddings: {len(data.get('data', []))}")
        
        embedding = data['data'][0]['embedding']
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        assert len(embedding) > 0, "Empty embedding"
        print("✅ Multimodal base64 PASSED")
        return True
    except Exception as e:
        print(f"❌ Multimodal base64 FAILED: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False


def test_multimodal_url():
    """Test multimodal embedding with URL image"""
    print("\n" + "="*60)
    print("TEST 5: Multimodal - URL Image")
    print("="*60)
    
    print(f"Testing URL: {TEST_IMAGE_URL}")
    
    payload = {
        "input": {
            "text": "Describe the transparency demonstration in this image",
            "image": {
                "type": "url",
                "data": TEST_IMAGE_URL
            }
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response status: {response.status_code}")
        print(f"Number of embeddings: {len(data.get('data', []))}")
        
        embedding = data['data'][0]['embedding']
        print(f"Embedding dimension: {len(embedding)}")
        
        assert len(embedding) > 0, "Empty embedding"
        print("✅ Multimodal URL PASSED")
        return True
    except Exception as e:
        print(f"❌ Multimodal URL FAILED: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False


def test_multimodal_local_path():
    """Test multimodal embedding with local path image"""
    print("\n" + "="*60)
    print("TEST 6: Multimodal - Local Path Image")
    print("="*60)
    
    # Ensure test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        create_test_image(TEST_IMAGE_PATH)
    
    print(f"Testing path: {TEST_IMAGE_PATH}")
    
    payload = {
        "input": {
            "text": "What geometric shapes do you see?",
            "image": {
                "type": "path",
                "data": TEST_IMAGE_PATH
            }
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response status: {response.status_code}")
        print(f"Number of embeddings: {len(data.get('data', []))}")
        
        embedding = data['data'][0]['embedding']
        print(f"Embedding dimension: {len(embedding)}")
        
        assert len(embedding) > 0, "Empty embedding"
        print("✅ Multimodal local path PASSED")
        return True
    except Exception as e:
        print(f"❌ Multimodal local path FAILED: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False


def test_multimodal_batch():
    """Test batch multimodal embeddings with mixed text-only and text+image"""
    print("\n" + "="*60)
    print("TEST 7: Multimodal - Batch Mixed")
    print("="*60)
    
    # Ensure test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        create_test_image(TEST_IMAGE_PATH)
    
    base64_image = encode_image_to_base64(TEST_IMAGE_PATH)
    
    payload = {
        "input": [
            {"text": "Text-only query without image"},
            {
                "text": "Query with base64 image",
                "image": {"type": "base64", "data": base64_image}
            },
            {"text": "Another text-only query"}
        ],
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response status: {response.status_code}")
        print(f"Number of embeddings: {len(data.get('data', []))}")
        
        for i, item in enumerate(data['data']):
            print(f"  Embedding {i}: dimension={len(item['embedding'])}, index={item['index']}")
            assert len(item['embedding']) > 0, f"Empty embedding at index {i}"
        
        print("✅ Multimodal batch PASSED")
        return True
    except Exception as e:
        print(f"❌ Multimodal batch FAILED: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False


def test_error_invalid_base64():
    """Test error handling for invalid base64"""
    print("\n" + "="*60)
    print("TEST 8: Error Handling - Invalid Base64")
    print("="*60)
    
    payload = {
        "input": {
            "text": "This should fail",
            "image": {
                "type": "base64",
                "data": "not_valid_base64!!!"
            }
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        data = response.json()
        print(f"Error detail: {data.get('detail')}")
        print("✅ Error handling (base64) PASSED")
        return True
    except AssertionError as e:
        print(f"❌ Error handling (base64) FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ Error handling (base64) FAILED: {e}")
        return False


def test_error_invalid_path():
    """Test error handling for invalid file path"""
    print("\n" + "="*60)
    print("TEST 9: Error Handling - Invalid Path")
    print("="*60)
    
    payload = {
        "input": {
            "text": "This should fail",
            "image": {
                "type": "path",
                "data": "/nonexistent/path/to/image.png"
            }
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        data = response.json()
        print(f"Error detail: {data.get('detail')}")
        print("✅ Error handling (path) PASSED")
        return True
    except AssertionError as e:
        print(f"❌ Error handling (path) FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ Error handling (path) FAILED: {e}")
        return False


def test_error_invalid_image_type():
    """Test error handling for invalid image type"""
    print("\n" + "="*60)
    print("TEST 10: Error Handling - Invalid Image Type")
    print("="*60)
    
    payload = {
        "input": {
            "text": "This should fail",
            "image": {
                "type": "invalid_type",
                "data": "some_data"
            }
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        data = response.json()
        print(f"Error detail: {data.get('detail')}")
        print("✅ Error handling (type) PASSED")
        return True
    except AssertionError as e:
        print(f"❌ Error handling (type) FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ Error handling (type) FAILED: {e}")
        return False


def test_instruction():
    """Test that instructions change the embedding"""
    print("\n" + "="*60)
    print("TEST 12: Instruction Parameter")
    print("="*60)

    def get_embedding(text, instruction=None):
        payload = {
            "input": text,
            "model": "qwen-vl-embedding"
        }
        if instruction:
            payload["instruction"] = instruction
        response = requests.post(EMBEDDINGS_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    text = "A cat sitting on a couch"

    try:
        print("Getting embedding without instruction...")
        emb_no_inst = get_embedding(text)

        print("Getting embedding with instruction: \"Represent this for retrieval\"")
        emb_inst_1 = get_embedding(text, "Represent this for retrieval")

        print("Getting embedding with instruction: \"Describe this image\"")
        emb_inst_2 = get_embedding(text, "Describe this image")

        import numpy as np
        vec_no_inst = np.array(emb_no_inst)
        vec_1 = np.array(emb_inst_1)
        vec_2 = np.array(emb_inst_2)

        sim_no_vs_1 = np.dot(vec_no_inst, vec_1) / (np.linalg.norm(vec_no_inst) * np.linalg.norm(vec_1))
        sim_1_vs_2 = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

        print(f"Similarity (No Inst vs Inst 1): {sim_no_vs_1:.4f}")
        print(f"Similarity (Inst 1 vs Inst 2): {sim_1_vs_2:.4f}")

        assert sim_no_vs_1 < 0.999, f"Instruction did not change embedding (sim={sim_no_vs_1:.4f})"
        assert sim_1_vs_2 < 0.999, f"Different instructions produced same embedding (sim={sim_1_vs_2:.4f})"

        print("✅ Instruction parameter PASSED")
        return True
    except Exception as e:
        print(f"❌ Instruction parameter FAILED: {e}")
        return False




def test_embedding_similarity():
    """Test that similar texts/images produce similar embeddings"""
    print("\n" + "="*60)
    print("TEST 11: Embedding Similarity")
    print("="*60)
    
    # Ensure test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        create_test_image(TEST_IMAGE_PATH)
    
    base64_image = encode_image_to_base64(TEST_IMAGE_PATH)
    
    # Two similar multimodal queries
    payload1 = {
        "input": {
            "text": "Describe the shapes in this image",
            "image": {"type": "base64", "data": base64_image}
        },
        "model": "qwen-vl-embedding"
    }
    
    payload2 = {
        "input": {
            "text": "What geometric shapes are visible?",
            "image": {"type": "base64", "data": base64_image}
        },
        "model": "qwen-vl-embedding"
    }
    
    try:
        response1 = requests.post(EMBEDDINGS_ENDPOINT, json=payload1, timeout=60)
        response2 = requests.post(EMBEDDINGS_ENDPOINT, json=payload2, timeout=60)
        
        response1.raise_for_status()
        response2.raise_for_status()
        
        emb1 = response1.json()['data'][0]['embedding']
        emb2 = response2.json()['data'][0]['embedding']
        
        # Calculate cosine similarity
        import numpy as np
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        print(f"Similarity between similar queries: {similarity:.4f}")
        assert similarity > 0.5, f"Expected high similarity, got {similarity}"
        
        print("✅ Embedding similarity PASSED")
        return True
    except Exception as e:
        print(f"❌ Embedding similarity FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "🧪" * 30)
    print("Qwen3-VL-Embedding Multimodal Test Suite")
    print("🧪" * 30)
    
    # Check if server is running
    print("\nChecking server connectivity...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        print(f"✅ Server is running on {BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Server not running on {BASE_URL}")
        print("Please start the server first: python server.py")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_text_only_single,
        test_text_only_batch,
        test_multimodal_base64,
        test_multimodal_url,
        test_multimodal_local_path,
        test_multimodal_batch,
        test_error_invalid_base64,
        test_error_invalid_path,
        test_error_invalid_image_type,
        test_embedding_similarity,
        test_instruction,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ {test_func.__name__} CRASHED: {e}")
            results.append((test_func.__name__, False))
        time.sleep(0.5)  # Brief pause between tests
    
    # Report results
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("="*60)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*60)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)