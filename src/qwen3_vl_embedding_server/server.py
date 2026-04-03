"""
FastAPI server for Qwen3-VL-Embedding-2B
OpenAI-compatible /v1/embeddings endpoint with multimodal support (text + images)
"""

import time
import base64
import io
import os
from typing import Union, List, Optional
from urllib.parse import urlparse
import requests

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

app = FastAPI(title="Qwen3-VL-Embedding Server")

# Global model and processor
model = None
processor = None
device = None


class ImageInput(BaseModel):
    """Image input specification with type and data"""
    type: str = Field(..., description="Image source type: 'base64', 'url', or 'path'")
    data: str = Field(..., description="Image data: base64 string, URL, or file path")


class MultimodalInput(BaseModel):
    """Single multimodal input entry containing text and optional image"""
    text: str = Field(..., description="Text content to embed")
    image: Optional[ImageInput] = Field(None, description="Optional image input")


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request with multimodal support"""
    input: Union[str, List[str], MultimodalInput, List[MultimodalInput]]
    model: str = "qwen-vl-embedding"
    encoding_format: str = "float"
    instruction: Optional[str] = Field(
        default=None,
        description="Task instruction for instruction-based embeddings (e.g., 'Represent this for retrieval')"
    )


class EmbeddingObject(BaseModel):
    """Single embedding result"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: dict


# ============================================================================
# Image Processing Functions
# ============================================================================

def load_image_from_base64(base64_str: str) -> Image.Image:
    """Load PIL Image from base64-encoded string"""
    try:
        # Remove data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")


def load_image_from_url(url: str) -> Image.Image:
    """Download and load image from URL"""
    try:
        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            raise HTTPException(status_code=400, detail="URL must use http or https scheme")
        
        # Set reasonable timeout and headers
        headers = {'User-Agent': 'Qwen3-VL-Embedding-Server/1.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            # Try to load anyway, some servers don't set content-type correctly
            pass
        
        image = Image.open(io.BytesIO(response.content))
        return image.convert('RGB')
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail=f"Image download timeout: {url}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data from URL: {str(e)}")


def load_image_from_path(path: str) -> Image.Image:
    """Load image from local filesystem path"""
    try:
        # Resolve path and check if it exists
        resolved_path = os.path.expanduser(os.path.expandvars(path))
        if not os.path.exists(resolved_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
        
        if not os.path.isfile(resolved_path):
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
        
        image = Image.open(resolved_path)
        return image.convert('RGB')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from path: {str(e)}")


def load_image(image_input: ImageInput) -> Image.Image:
    """Dispatch to appropriate image loader based on type"""
    if image_input.type == 'base64':
        return load_image_from_base64(image_input.data)
    elif image_input.type == 'url':
        return load_image_from_url(image_input.data)
    elif image_input.type == 'path':
        return load_image_from_path(image_input.data)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image type: '{image_input.type}'. Must be 'base64', 'url', or 'path'"
        )


# ============================================================================
# Multimodal Processing
# ============================================================================

def is_multimodal_input(input_data) -> bool:
    """Check if input is multimodal (contains image fields)"""
    if isinstance(input_data, dict):
        return 'text' in input_data
    elif hasattr(input_data, 'text'):
        return True
    return False


def normalize_inputs(input_data) -> List[MultimodalInput]:
    """Normalize various input formats to list of MultimodalInput objects"""
    # Handle string input (backward compatibility - text only)
    if isinstance(input_data, str):
        return [MultimodalInput(text=input_data)]
    
    # Handle list inputs
    if isinstance(input_data, list):
        if len(input_data) == 0:
            raise HTTPException(status_code=400, detail="Input list cannot be empty")
        if isinstance(input_data[0], str):
            return [MultimodalInput(text=text) for text in input_data]
        if isinstance(input_data[0], dict):
            return [MultimodalInput(**item) for item in input_data]
        if isinstance(input_data[0], MultimodalInput):
            return input_data

    # Handle single multimodal input dict
    if isinstance(input_data, dict):
        return [MultimodalInput(**input_data)]

    # Already a MultimodalInput
    if isinstance(input_data, MultimodalInput):
        return [input_data]
    
    raise HTTPException(status_code=400, detail=f"Invalid input format")


def process_multimodal_inputs(inputs: List[MultimodalInput], instruction: Optional[str] = None):
    """
    Process multimodal inputs into format expected by Qwen3-VL processor.
    
    Args:
        inputs: List of multimodal inputs
        instruction: Optional task instruction to prepend to each text
    
    Returns: (texts, images) tuple where images is None if no images present
    """
    texts = []
    images = []
    has_images = False
    
    for item in inputs:
        # Prepend instruction if provided (instruction-based embeddings)
        text = item.text
        if instruction:
            text = f"{instruction}: {text}"
        
        # Add vision tokens to text when image is present (Qwen3-VL requirement)
        if item.image:
            # Qwen3-VL requires vision anchor tokens for image-text alignment
            texts.append(f"<|vision_start|><|image_pad|><|vision_end|>{text}")
        else:
            texts.append(text)
        
        if item.image:
            try:
                pil_image = load_image(item.image)
                images.append(pil_image)
                has_images = True
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
        else:
            # Placeholder for text-only entries in batch
            images.append(None)
    
    # If no images, return None for images (text-only mode)
    if not has_images:
        return texts, None
    
    return texts, images


# ============================================================================
# Model Loading
# ============================================================================

@app.on_event("startup")
async def load_model():
    global model, processor, device
    print("=" * 60)
    print("Loading Qwen3-VL-Embedding-2B...")
    print("=" * 60)
    start = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    
    try:
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.eval()
        
        elapsed = time.time() - start
        print(f"Model loaded in {elapsed:.2f}s")
        print(f"Multimodal support: ENABLED")
        print("=" * 60)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for text and/or multimodal inputs.
    
    Supports:
    - Single text: {"input": "text here"}
    - List of texts: {"input": ["text1", "text2"]}
    - Multimodal (text + image):
      {"input": {"text": "describe", "image": {"type": "base64", "data": "..."}}}
    - List of multimodal inputs with mixed text-only and text+image
    """
    start_time = time.time()

    if request.encoding_format != "float":
        raise HTTPException(status_code=400, detail="Only encoding_format='float' is supported")

    try:
        # Normalize inputs to list of MultimodalInput objects
        normalized_inputs = normalize_inputs(request.input)
        
        # Process inputs (load images if present, apply instruction if provided)
        texts, images = process_multimodal_inputs(normalized_inputs, request.instruction)
        
        # Count tokens (approximate)
        total_tokens = sum(len(text.split()) for text in texts)
        
        with torch.no_grad():
            if images is not None:
                # Multimodal processing
                # Filter out None images for processor
                valid_images = [img for img in images if img is not None]
                
                # Process with Qwen3-VL processor
                # For mixed batches, we need to handle text-only items specially
                if len(valid_images) == len(texts):
                    # All entries have images
                    inputs = processor(
                        text=texts,
                        images=valid_images,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                else:
                    # Mixed batch - process separately for now
                    # This is a simplification; true batch processing would be more complex
                    embeddings_list = []
                    for text, img in zip(texts, images):
                        if img:
                            proc_inputs = processor(
                                text=[text],
                                images=[img],
                                return_tensors="pt",
                                padding=True
                            ).to(device)
                        else:
                            proc_inputs = processor(
                                text=[text],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                            ).to(device)
                        
                        outputs = model(**proc_inputs)
                        # Use LAST TOKEN pooling for correct embedding extraction
                        emb = outputs.last_hidden_state[:, -1, :].cpu().float().numpy()
                        embeddings_list.append(emb[0])
                    
                    # Format response
                    data = []
                    for i, emb in enumerate(embeddings_list):
                        data.append(EmbeddingObject(
                            embedding=emb.tolist(),
                            index=i
                        ))
                    
                    elapsed = time.time() - start_time
                    
                    return EmbeddingResponse(
                        data=data,
                        model=request.model,
                        usage={
                            "prompt_tokens": total_tokens,
                            "total_tokens": total_tokens
                        }
                    )
            else:
                # Text-only processing (backward compatible)
                inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
            
            # Get embeddings using LAST TOKEN pooling (not mean pooling)
            # Use attention mask to find the real last token per item (handles padding in batches)
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            sequence_lengths = attention_mask.sum(dim=1) - 1
            embeddings = outputs.last_hidden_state[
                torch.arange(outputs.last_hidden_state.size(0), device=device),
                sequence_lengths
            ].cpu().float().numpy()
        
        # Format response
        data = []
        for i, emb in enumerate(embeddings):
            data.append(EmbeddingObject(
                embedding=emb.tolist(),
                index=i
            ))
        
        elapsed = time.time() - start_time
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint with model status"""
    return {
        "status": "ok", 
        "device": str(device), 
        "model": "Qwen3-VL-Embedding-2B",
        "multimodal_supported": True
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Qwen3-VL-Embedding Server",
        "version": "1.0.0",
        "model": "Qwen3-VL-Embedding-2B",
        "endpoints": {
            "POST /v1/embeddings": "Create embeddings (text + images)",
            "GET /health": "Health check"
        },
        "multimodal": True
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Qwen3-VL-Embedding Server...")
    print("Features: Multimodal (text + images), OpenAI-compatible API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
