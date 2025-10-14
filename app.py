from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import hashlib
import logging
import numpy as np
import base64
from functools import lru_cache
from diskcache import Cache
import asyncio
from typing import Optional, Union, List
import os

# Import settings
from settings import (
    MODEL_ID, DEVICE, DTYPE, MAX_TEXTS, RETURN_NUMPY,
    CACHE_DIR, CACHE_SIZE_LIMIT, CACHE_EVICTION_POLICY,
    MAX_CONCURRENT_REQUESTS, LOG_LEVEL, LOG_CACHE_STATS
)


# LOGGING SETUP

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding-service")


# FASTAPI APP

app = FastAPI(
    title="Qwen Embedding API",
    description="Local embedding service powered by Qwen SentenceTransformer.",
    version="1.0.0"
)


# CONCURRENCY CONTROL

# Semaphore to limit concurrent embedding requests
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# GLOBAL VARIABLES

model = None
device_name = None
dtype_name = None
cache_stats = {"hits": 0, "misses": 0}
cache = None


# SCHEMAS

class EmbedRequest(BaseModel):
    texts: list[str]
    return_numpy: Optional[bool] = None

class EmbedResponse(BaseModel):
    embeddings: Union[List[List[float]], str]  # List of floats or base64 encoded numpy
    num_texts: int
    dim: int
    format: str  # "list" or "numpy"


# STARTUP & SHUTDOWN

@app.on_event("startup")
async def startup_event():
    global model, device_name, dtype_name, cache
    
    # Determine device
    if DEVICE == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = DEVICE
    
    # Determine dtype
    if DTYPE == "auto":
        dtype_name = torch.float16 if device_name == "cuda" else torch.float32
    elif DTYPE == "float16":
        dtype_name = torch.float16
    else:
        dtype_name = torch.float32
    
    logger.info(f"🔹 Loading model {MODEL_ID} on {device_name} ({dtype_name})...")
    model = SentenceTransformer(MODEL_ID, device=device_name)
    
    # Initialize cache with size limit and eviction policy
    cache = Cache(
        CACHE_DIR,
        size_limit=CACHE_SIZE_LIMIT,
        eviction_policy=CACHE_EVICTION_POLICY
    )
    logger.info(f"Cache initialized at {CACHE_DIR} with {CACHE_SIZE_LIMIT} bytes limit")

@app.on_event("shutdown")
async def shutdown_event():
    if cache:
        cache.close()
    logger.info("Application shutting down")


# UTILITIES

@lru_cache(maxsize=1024)
def _hash_text(text: str) -> str:
    """Secure hash for text caching using SHA-256"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

async def _cached_encode(texts, return_numpy=None):
    """Encode texts with caching, with optional numpy binary return format"""
    use_numpy = RETURN_NUMPY if return_numpy is None else return_numpy
    results = []
    uncached_texts = []
    uncached_indices = []
    
    # Track cache stats
    hits = 0
    misses = 0

    # Check cache
    for i, text in enumerate(texts):
        h = _hash_text(text)
        if h in cache:
            results.append(cache[h])
            hits += 1
        else:
            results.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)
            misses += 1

    # Update global stats
    if LOG_CACHE_STATS:
        cache_stats["hits"] += hits
        cache_stats["misses"] += misses
        logger.info(f"Cache stats - Hits: {hits}, Misses: {misses}, "
                   f"Total: {cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']} "
                   f"({cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) * 100:.1f}%)")

    # Encode missing ones
    if uncached_texts:
        # Use semaphore to limit concurrent model inference
        async with request_semaphore:
            # Run in a thread to not block the event loop
            new_embs = await asyncio.to_thread(
                model.encode, uncached_texts, convert_to_numpy=True
            )
            
            for idx, emb in zip(uncached_indices, new_embs):
                h = _hash_text(texts[idx])
                # Always store as list in cache for compatibility
                cache[h] = emb.tolist()
                results[idx] = emb.tolist()

    # Convert to numpy binary if requested
    if use_numpy and results:
        # Convert to numpy array
        np_array = np.array(results, dtype=np.float32)
        # Encode as base64 string
        binary_data = base64.b64encode(np_array.tobytes()).decode('ascii')
        return binary_data, "numpy", np_array.shape[1]
    else:
        return results, "list", len(results[0]) if results else 0


# ROUTES

@app.get("/")
def root():
    return {
        "message": "Qwen Embedding API is running",
        "model": MODEL_ID,
        "device": device_name,
        "max_texts": MAX_TEXTS,
        "cache_stats": cache_stats if LOG_CACHE_STATS else "disabled"
    }

@app.post("/embed")
async def embed_texts(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")
    if len(req.texts) > MAX_TEXTS:
        raise HTTPException(status_code=400, detail=f"Too many texts. Max allowed: {MAX_TEXTS}.")

    embeddings, format_type, dim = await _cached_encode(req.texts, req.return_numpy)
    
    return {
        "embeddings": embeddings,
        "num_texts": len(req.texts),
        "dim": dim,
        "format": format_type
    }
