import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Model settings
MODEL_ID = os.getenv('MODEL_ID', 'Qwen/Qwen3-Embedding-0.6B')
DEVICE = os.getenv('DEVICE', 'auto')  # 'auto', 'cuda', or 'cpu'
DTYPE = os.getenv('DTYPE', 'auto')    # 'auto', 'float16', or 'float32'

# API settings
MAX_TEXTS = int(os.getenv('MAX_TEXTS', '32'))
RETURN_NUMPY = os.getenv('RETURN_NUMPY', 'false').lower() == 'true'

# Cache settings
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
CACHE_SIZE_LIMIT = int(os.getenv('CACHE_SIZE_LIMIT', '1073741824'))  # 1GB default
CACHE_EVICTION_POLICY = os.getenv('CACHE_EVICTION_POLICY', 'least-recently-stored')

# Concurrency settings
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_CACHE_STATS = os.getenv('LOG_CACHE_STATS', 'true').lower() == 'true'