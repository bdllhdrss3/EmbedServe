# API Embedding Service (Default is Qwen)

A high-performance, production-ready embedding service powered by Qwen embedding models. This service provides a simple REST API for generating text embeddings with built-in caching, concurrency control, and configurable settings.

## Motivation

Text embeddings are a fundamental component of modern NLP systems, enabling semantic search, recommendation systems, clustering, and more. However, running embedding models efficiently at scale presents several challenges:

- **Cost**: Using cloud-based embedding APIs can become expensive with high volume
- **Privacy**: Sensitive data should not be sent to third-party services
- **Latency**: Network calls to external APIs add latency
- **Customization**: Limited control over model parameters and behavior

This project addresses these challenges by providing a self-hosted embedding service that:

1. Runs locally or in your own infrastructure
2. Implements intelligent caching to reduce redundant computations
3. Provides concurrency control to maximize throughput
4. Offers flexible configuration options
5. Supports binary response format for large embedding batches

## Features

- 🚀 **High Performance**: Optimized for throughput with concurrency control
- 💾 **Intelligent Caching**: Disk-based caching with configurable size limits
- 📊 **Monitoring**: Cache hit/miss statistics for performance tuning
- 🔧 **Configurable**: Environment variables for all settings
- 🐳 **Containerized**: Ready-to-use Docker support
- 🔄 **Format Options**: Return embeddings as JSON lists or binary numpy arrays
- 🔌 **Model Flexibility**: Easy to swap embedding models

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (CPU or CUDA)

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd embed-model
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   # On Windows
   .\env\Scripts\activate
   # On Linux/Mac
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

5. Run the service:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

### Option 2: Docker

1. Build the Docker image:
   ```bash
   docker build -t qwen-embedding-service .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -v ./cache:/app/cache qwen-embedding-service
   ```

   With custom environment variables:
   ```bash
   docker run -p 8000:8000 -v ./cache:/app/cache \
     -e MODEL_ID=Qwen/Qwen3-Embedding-0.6B \
     -e MAX_CONCURRENT_REQUESTS=20 \
     qwen-embedding-service
   ```

## Usage

### API Endpoints

#### GET /

Returns basic information about the service.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Qwen Embedding API is running",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "device": "cuda",
  "max_texts": 32,
  "cache_stats": {"hits": 45, "misses": 12}
}
```

#### POST /embed

Generates embeddings for a list of texts.

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "How are you?"], "return_numpy": false}'
```

Response:
```json
{
  "embeddings": [[0.123, 0.456, ...], [0.789, 0.012, ...]],
  "num_texts": 2,
  "dim": 1536,
  "format": "list"
}
```

For large responses, use `return_numpy: true` to get a more efficient binary format:

```json
{
  "embeddings": "base64_encoded_binary_data...",
  "num_texts": 1000,
  "dim": 1536,
  "format": "numpy"
}
```

To decode the numpy array in Python:
```python
import base64
import numpy as np

# Assuming response is the JSON response from the API
binary_data = base64.b64decode(response["embeddings"])
embeddings = np.frombuffer(binary_data, dtype=np.float32).reshape(-1, response["dim"])
```

## Configuration

All settings can be configured via environment variables or the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_ID | HuggingFace model ID | Qwen/Qwen3-Embedding-0.6B |
| DEVICE | Device to run model on (auto, cuda, cpu) | auto |
| DTYPE | Data type (auto, float16, float32) | auto |
| MAX_TEXTS | Maximum texts per request | 32 |
| RETURN_NUMPY | Return embeddings as numpy binary | false |
| CACHE_DIR | Cache directory | ./cache |
| CACHE_SIZE_LIMIT | Cache size limit in bytes | 1073741824 (1GB) |
| CACHE_EVICTION_POLICY | Cache eviction policy | least-recently-stored |
| MAX_CONCURRENT_REQUESTS | Maximum concurrent embedding requests | 10 |
| LOG_LEVEL | Logging level | INFO |
| LOG_CACHE_STATS | Log cache hit/miss statistics | true |

## Using Other Embedding Models

You can easily use other embedding models by changing the `MODEL_ID` in your `.env` file. The service supports any model compatible with the SentenceTransformers library.

Popular alternatives include:

- `BAAI/bge-large-en-v1.5` - BGE large English model
- `intfloat/e5-large-v2` - E5 large model
- `sentence-transformers/all-mpnet-base-v2` - MPNet base model
- `thenlper/gte-large` - GTE large model

Example:
```
MODEL_ID=BAAI/bge-large-en-v1.5
```

For models with different embedding dimensions, the service will automatically adjust.

## Performance Tuning

For optimal performance:

1. **Cache Size**: Adjust `CACHE_SIZE_LIMIT` based on your available disk space and expected text volume
2. **Concurrency**: Set `MAX_CONCURRENT_REQUESTS` based on your hardware capabilities
3. **Device**: Use `DEVICE=cuda` if you have a GPU
4. **Data Type**: Use `DTYPE=float16` for faster GPU inference with minimal precision loss
5. **Return Format**: Use `RETURN_NUMPY=true` for large batches to reduce JSON serialization overhead

## License

This project is free for use under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
