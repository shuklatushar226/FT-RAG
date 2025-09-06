# FT+RAG: Fine-tuning + Retrieval Augmented Generation System

A complete system for ingesting Git repositories, creating vector embeddings, and building fine-tuned language models for code-aware question answering.

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **16GB+ RAM** (32GB+ recommended for fine-tuning)
- **Git** installed
- **Gemini API Key** (get from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 1. Clone and Setup

```bash
git clone <your-repo>
cd FT+RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env

# Edit .env with your settings:
# GOOGLE_API_KEY=your_gemini_api_key_here
# GITHUB_TOKEN=your_github_token_for_private_repos (optional)
```

### 3. Start Services

```bash
# Terminal 1: Start Qdrant vector database
./qdrant

# Terminal 2: Start API server
source venv/bin/activate
export GOOGLE_API_KEY="your_key_here"
uvicorn app.api:app --reload --port 8000
```

### 4. Verify Setup

```bash
# Check health
curl http://localhost:8000/health

# Check available models
curl http://localhost:8000/models/info
```

## 📋 System Status

**✅ WORKING COMPONENTS:**
- **FastAPI Server** - Running on http://localhost:8000
- **Qdrant Vector DB** - Running on http://localhost:6333 (2,385 chunks stored)
- **Repository Ingestion** - Optimized batch processing with private repo support
- **RAG Query System** - Using Gemini 2.0 Flash model
- **Fine-tuning Pipeline** - Complete QLoRA setup with Axolotl
- **Training Data Generation** - 2,636+ examples from ingested repositories

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Git Repos     │───▶│   Qdrant Vector  │───▶│   RAG Query     │
│   (GitHub)      │    │   Database       │    │   System        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Fine-tuning    │    │   Embeddings     │    │  Gemini 2.0     │
│  Pipeline       │    │   (BGE Model)    │    │  Flash API      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 API Endpoints

### Core Functionality

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Ask questions (supports `model` parameter) |
| `/ingest/large` | POST | Ingest repositories with background processing |
| `/ingest/status/{job_id}` | GET | Check ingestion status |
| `/ingest/status` | GET | List all jobs |

### Fine-tuning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/info` | GET | List available models |
| `/fine-tune/generate-data` | POST | Generate training data from repos |
| `/fine-tune/start` | POST | Start fine-tuning process |

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/github` | GET | GitHub authentication instructions |

## 💻 Usage Examples

### 1. Ingest a Repository

```bash
# Public repository
curl -X POST http://localhost:8000/ingest/large \
  -H "Content-Type: application/json" \
  -d '["https://github.com/user/repo"]'

# Private repository (with token)
curl -X POST http://localhost:8000/ingest/large \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Token: ghp_your_token" \
  -d '["https://github.com/user/private-repo"]'
```

### 2. Query the System

```bash
# Query with Gemini (default)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "How do I implement payment processing?"}'

# Query with fine-tuned model (after training)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "How do I implement payment processing?", "model": "fine-tuned"}'
```

### 3. Fine-tuning Workflow

```bash
# 1. Generate training data
curl -X POST http://localhost:8000/fine-tune/generate-data

# 2. Start fine-tuning
curl -X POST http://localhost:8000/fine-tune/start

# 3. Check progress
curl http://localhost:8000/ingest/status/{job_id}
```

## 📁 Project Structure

```
FT+RAG/
├── app/                     # Main application
│   ├── api.py              # FastAPI server with all endpoints
│   ├── ingest.py           # Repository ingestion with optimization
│   ├── rag_langchain.py    # RAG system using LangChain
│   ├── settings.py         # Configuration management
│   ├── splitter.py         # Code-aware text splitting
│   └── fine_tuned_model.py # Fine-tuned model integration
├── ft/                      # Fine-tuning pipeline
│   ├── axolotl.yaml        # QLoRA configuration
│   ├── build_sft_data.py   # Training data builder
│   ├── generate_training_data.py # Extract data from repos
│   ├── train_adapter.sh    # Complete training script
│   ├── fine_tuned_model.py # Model loading utilities
│   ├── sft.jsonl          # Generated training data (2,636 examples)
│   └── README.md          # Fine-tuning documentation
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
├── qdrant                 # Qdrant binary
└── README.md             # This file
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Vector Database
QDRANT_URL=http://localhost:6333
COLLECTION=repo_chunks

# Embeddings
EMB_MODEL=BAAI/bge-base-en-v1.5
CHUNK_TOKENS=1000
CHUNK_OVERLAP=200

# API Keys
GOOGLE_API_KEY=your_gemini_api_key_here
GITHUB_TOKEN=your_github_token_here

# Optimization Settings
BATCH_SIZE=32
EMBEDDING_BATCH_SIZE=16
MAX_FILE_SIZE=2000000
MAX_CONCURRENT_INGESTIONS=2

# Security
WEBHOOK_SECRET=your_secure_webhook_secret_here
```

### Supported File Types

- **Documents**: `.md`, `.rst`, `.txt`
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.kt`, `.go`, `.rs`, `.cpp`, `.c`, `.cs`, `.php`, `.rb`
- **Config**: `.sh`, `.yml`, `.yaml`

## 🔧 Fine-tuning Setup

### 1. Hardware Requirements

| Setup | RAM | GPU | Training Time |
|-------|-----|-----|---------------|
| **Minimum** | 16GB | CPU only | 8-12 hours |
| **Recommended** | 16GB+ | 8GB+ VRAM | 2-4 hours |
| **Optimal** | 32GB+ | 24GB+ VRAM | 30-60 min |

### 2. Quick Fine-tuning

```bash
# Generate training data from ingested repos
cd ft
python generate_training_data.py

# Start training (requires GPU for reasonable speed)
./train_adapter.sh
```

### 3. Model Configuration

Default setup uses **Qwen2-7B-Instruct** with:
- **4-bit quantization** for memory efficiency
- **LoRA rank 16** for parameter-efficient training
- **2 epochs** with cosine scheduling
- **Gradient accumulation** for larger effective batch size

## 🚨 Troubleshooting

### Common Issues

**1. API Server Won't Start**
```bash
# Check if port is in use
lsof -i :8000

# Export API key manually
export GOOGLE_API_KEY="your_key"
uvicorn app.api:app --reload --port 8000
```

**2. Qdrant Connection Failed**
```bash
# Start Qdrant manually
./qdrant

# Check if running
curl http://localhost:6333/collections
```

**3. Repository Ingestion Fails**
```bash
# For private repos, set GitHub token
export GITHUB_TOKEN="ghp_your_token"

# Or use header in API call
curl -H "X-GitHub-Token: your_token" ...
```

**4. Fine-tuning Out of Memory**
```bash
# Edit ft/axolotl.yaml
per_device_train_batch_size: 1  # Reduce batch size
gradient_accumulation_steps: 16  # Increase accumulation
load_in_8bit: true  # Use 8-bit instead of 4-bit
```

**5. Slow Embeddings Generation**
```bash
# For Apple Silicon
pip install torch torchvision torchaudio

# For NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependency Issues

```bash
# Install missing packages
pip install pandas datasets transformers accelerate

# For fine-tuning
pip install axolotl[flash-attn]

# Fix LangChain deprecation warnings
pip install langchain-huggingface langchain-qdrant
```

## 📊 Current Data Status

- **Vector Database**: 2,385 chunks stored
- **Training Data**: 2,636 examples generated
- **Repositories**: Multiple repos ingested (vscode-python, hyperswitch-client-core)
- **Models**: Gemini 2.0 Flash (active), Fine-tuned (ready for training)

## 🔄 Development Workflow

### 1. Add New Repository
```bash
curl -X POST http://localhost:8000/ingest/large \
  -H "Content-Type: application/json" \
  -d '["https://github.com/your-org/new-repo"]'
```

### 2. Test Queries
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "Your question about the codebase"}'
```

### 3. Update Training Data
```bash
curl -X POST http://localhost:8000/fine-tune/generate-data
```

### 4. Fine-tune Model
```bash
curl -X POST http://localhost:8000/fine-tune/start
```

## 🔒 Security Notes

- **API Keys**: Store in `.env`, never commit to Git
- **GitHub Tokens**: Use minimal permissions (repo scope only)
- **Private Repos**: Tokens are removed from URLs after cloning
- **Webhooks**: Use strong webhook secrets

## 📚 Additional Resources

- **Fine-tuning Guide**: See `ft/README.md`
- **API Documentation**: Visit http://localhost:8000/docs when server is running
- **Qdrant Dashboard**: Visit http://localhost:6333/dashboard
- **Axolotl Documentation**: https://github.com/OpenAccess-AI-Collective/axolotl

## 🤝 Team Setup Checklist

- [ ] Clone repository
- [ ] Create virtual environment and install dependencies
- [ ] Get Gemini API key and add to `.env`
- [ ] Start Qdrant database (`./qdrant`)
- [ ] Start API server (`uvicorn app.api:app --reload --port 8000`)
- [ ] Test health endpoint (`curl http://localhost:8000/health`)
- [ ] Ingest your first repository
- [ ] Test RAG queries
- [ ] (Optional) Set up fine-tuning pipeline

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all services are running
3. Check API server logs for errors
4. Ensure environment variables are set correctly

---

**System Status**: ✅ **OPERATIONAL** - All components working and tested