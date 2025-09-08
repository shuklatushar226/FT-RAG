from fastapi import FastAPI, Body, Request, Header, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
import hmac, hashlib, os, asyncio
from concurrent.futures import ThreadPoolExecutor
from app.ingest import upsert_repo, upsert_repo_optimized, upsert_repo_incremental
from app.rag_langchain import make_qa_chain
from app.settings import settings
import uuid
from datetime import datetime

app = FastAPI()

# Track background ingestion jobs
ingestion_jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent ingestions

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/auth/github")
async def github_auth_info():
    """Instructions for GitHub authentication for private repositories."""
    return {
        "title": "GitHub Authentication for Private Repositories",
        "methods": [
            {
                "method": "Environment Variable",
                "description": "Set GITHUB_TOKEN in your .env file",
                "example": "GITHUB_TOKEN=ghp_your_personal_access_token_here",
                "steps": [
                    "1. Go to GitHub → Settings → Developer settings → Personal access tokens",
                    "2. Generate new token with 'repo' scope",
                    "3. Add to .env file: GITHUB_TOKEN=your_token",
                    "4. Restart the server"
                ]
            },
            {
                "method": "Request Header",
                "description": "Include X-GitHub-Token header in your API request",
                "example": {
                    "curl": "curl -X POST 'http://localhost:8000/ingest/large' -H 'X-GitHub-Token: ghp_your_token' -H 'Content-Type: application/json' -d '[\"https://github.com/user/private-repo\"]'",
                    "headers": {"X-GitHub-Token": "ghp_your_personal_access_token_here"}
                }
            },
            {
                "method": "URL Authentication",
                "description": "Include token directly in the repository URL",
                "example": "https://ghp_your_token@github.com/user/private-repo.git",
                "note": "Less secure - token visible in logs"
            },
            {
                "method": "SSH Key",
                "description": "Use SSH URLs if you have SSH keys configured",
                "example": "git@github.com:user/private-repo.git",
                "requirement": "SSH keys must be configured on the server"
            }
        ],
        "token_permissions": [
            "repo (Full control of private repositories)",
            "Contents (Read repository contents)"
        ],
        "security_note": "Personal access tokens are securely handled and not logged. Tokens in URLs are removed after cloning."
    }

@app.post("/ingest")
async def ingest(repos: list[str]):
    """Standard ingestion endpoint for small repositories."""
    out = []
    for r in repos:
        result = await asyncio.get_event_loop().run_in_executor(
            executor, upsert_repo, r
        )
        out.append(result)
    return out

@app.post("/ingest/large")
async def ingest_large(
    repos: list[str], 
    background_tasks: BackgroundTasks,
    github_token: Optional[str] = Header(None, alias="X-GitHub-Token")
):
    """Optimized ingestion endpoint for large repositories with background processing and private repo support."""
    job_ids = []
    
    for repo_url in repos:
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        ingestion_jobs[job_id] = {
            "repo_url": repo_url,
            "status": "queued",
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "result": None,
            "error": None
        }
        
        # Add background task with optional GitHub token
        background_tasks.add_task(process_large_repo, job_id, repo_url, github_token)
        job_ids.append(job_id)
    
    return {
        "message": f"Queued {len(repos)} repositories for processing",
        "job_ids": job_ids,
        "status_endpoint": "/ingest/status/{job_id}",
        "note": "For private repositories, include 'X-GitHub-Token' header or set GITHUB_TOKEN in .env"
    }

@app.post("/ingest/incremental")
async def ingest_incremental(
    repos: list[str], 
    background_tasks: BackgroundTasks,
    github_token: Optional[str] = Header(None, alias="X-GitHub-Token")
):
    """Incremental ingestion endpoint that only processes changed files since last update."""
    job_ids = []
    
    for repo_url in repos:
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        ingestion_jobs[job_id] = {
            "repo_url": repo_url,
            "status": "queued",
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "result": None,
            "error": None,
            "type": "incremental"
        }
        
        # Add background task with incremental processing
        background_tasks.add_task(process_incremental_repo, job_id, repo_url, github_token)
        job_ids.append(job_id)
    
    return {
        "message": f"Queued {len(repos)} repositories for incremental processing",
        "job_ids": job_ids,
        "status_endpoint": "/ingest/status/{job_id}",
        "note": "Only changed files will be processed. For private repositories, include 'X-GitHub-Token' header."
    }

@app.get("/ingest/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """Get the status of a background ingestion job."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ingestion_jobs[job_id]

@app.get("/ingest/status")
async def get_all_ingestion_status():
    """Get the status of all ingestion jobs."""
    return {"jobs": ingestion_jobs}

async def process_large_repo(job_id: str, repo_url: str, github_token: str = None):
    """Background task for processing large repositories with authentication support."""
    try:
        # Update job status
        ingestion_jobs[job_id]["status"] = "processing"
        ingestion_jobs[job_id]["progress"] = 10
        
        # Run the optimized ingestion in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            executor, upsert_repo_optimized, repo_url, 'data/repos', github_token
        )
        
        # Update job completion
        ingestion_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Update job error
        ingestion_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

async def process_incremental_repo(job_id: str, repo_url: str, github_token: str = None):
    """Background task for incremental repository processing."""
    try:
        # Update job status
        ingestion_jobs[job_id]["status"] = "processing"
        ingestion_jobs[job_id]["progress"] = 10
        
        # Run the incremental ingestion in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            executor, upsert_repo_incremental, repo_url, 'data/repos', github_token
        )
        
        # Update job completion
        ingestion_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Update job error
        ingestion_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

def verify_github_signature(secret, payload_body, signature_header):
    if not signature_header:
        return False
    sha_name, signature = signature_header.split('=')
    mac = hmac.new(secret.encode(), msg=payload_body, digestmod=hashlib.sha256)
    expected = mac.hexdigest()
    return hmac.compare_digest(expected, signature)

@app.post("/webhook/github")
async def gh_webhook(request: Request, x_hub_signature_256: Optional[str] = Header(None)):
    """GitHub webhook handler with incremental processing and branch filtering."""
    body = await request.body()
    if settings.WEBHOOK_SECRET:
        ok = verify_github_signature(settings.WEBHOOK_SECRET, body, x_hub_signature_256)
        if not ok:
            raise HTTPException(status_code=403, detail="Invalid signature")
    
    payload = await request.json()
    
    # Branch filtering - only process main/master branch pushes
    ref = payload.get('ref', '')
    if ref not in ['refs/heads/main', 'refs/heads/master']:
        return {
            "status": "skipped",
            "reason": f"Only main/master branch pushes are processed. Received: {ref}",
            "ref": ref
        }
    
    # Only process push events
    if payload.get('repository') is None:
        return {
            "status": "skipped", 
            "reason": "Not a repository push event"
        }
    
    repo_url = payload.get('repository', {}).get('html_url', '')
    if repo_url and not repo_url.endswith('.git'):
        repo_url += '.git'
    
    if repo_url:
        # Use incremental ingestion for webhook updates
        print(f"🔔 Webhook triggered for {repo_url} on {ref}")
        result = await asyncio.get_event_loop().run_in_executor(
            executor, upsert_repo_incremental, repo_url
        )
        result['trigger'] = 'webhook'
        result['branch'] = ref
        return result
    
    return {"skipped": True}

@app.post("/query")
async def query(q: str = Body(..., embed=True), model: str = "gemini"):
    """Query using either Gemini or fine-tuned model"""
    if model == "fine-tuned":
        # Use fine-tuned model
        from app.fine_tuned_model import fine_tuned_manager
        from app.rag_langchain import build_retriever
        
        # Check if fine-tuned model is available
        if not fine_tuned_manager.is_available():
            raise HTTPException(
                status_code=404,
                detail="Fine-tuned model not found. Please run the fine-tuning process first."
            )
        
        # Get relevant context from RAG
        retriever = build_retriever(k=6)
        docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response using fine-tuned model
        instruction = f"Answer the following question based on the provided context:\n\nQuestion: {q}"
        response = fine_tuned_manager.generate(instruction, context)
        
        return {
            "answer": response,
            "model": "fine-tuned",
            "context_docs": len(docs)
        }
    else:
        # Use Gemini (default)
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0
        )
        qa = make_qa_chain(llm, k=6)
        res = qa.invoke({"query": q})
        return {
            "answer": res["result"],
            "model": "gemini"
        }

@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    from app.fine_tuned_model import fine_tuned_manager
    
    models = {
        "gemini": {
            "name": "Gemini 2.0 Flash",
            "available": bool(settings.GOOGLE_API_KEY),
            "type": "cloud_api",
            "description": "Google's latest fast and efficient language model"
        },
        "fine_tuned": {
            "name": "Fine-tuned Model",
            "available": fine_tuned_manager.is_available(),
            "type": "local",
            "description": "Custom fine-tuned model based on ingested repositories",
            "info": fine_tuned_manager.get_info() if fine_tuned_manager.is_available() else None
        }
    }
    
    return {"models": models}

@app.post("/fine-tune/generate-data")
async def generate_training_data():
    """Generate training data from ingested repositories"""
    try:
        # Run the data generation script
        result = await asyncio.get_event_loop().run_in_executor(
            executor, _run_generate_data
        )
        return {"message": "Training data generated successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

@app.post("/fine-tune/start")
async def start_fine_tuning(background_tasks: BackgroundTasks):
    """Start the fine-tuning process"""
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    ingestion_jobs[job_id] = {
        "type": "fine_tuning",
        "status": "queued",
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "result": None,
        "error": None
    }
    
    # Add background task for fine-tuning
    background_tasks.add_task(process_fine_tuning, job_id)
    
    return {
        "message": "Fine-tuning process started",
        "job_id": job_id,
        "status_endpoint": f"/ingest/status/{job_id}",
        "note": "Fine-tuning may take several hours depending on data size and hardware"
    }

def _run_generate_data():
    """Helper function to run data generation"""
    import subprocess
    import os
    
    # Change to ft directory
    ft_dir = os.path.join(os.getcwd(), "ft")
    
    # Run the data generation script
    result = subprocess.run(
        ["python", "generate_training_data.py"],
        cwd=ft_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Data generation failed: {result.stderr}")
    
    return result.stdout

async def process_fine_tuning(job_id: str):
    """Background task for fine-tuning process"""
    try:
        # Update job status
        ingestion_jobs[job_id]["status"] = "processing"
        ingestion_jobs[job_id]["progress"] = 10
        
        # Run the fine-tuning script
        result = await asyncio.get_event_loop().run_in_executor(
            executor, _run_fine_tuning
        )
        
        # Update job completion
        ingestion_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Update job error
        ingestion_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

def _run_fine_tuning():
    """Helper function to run fine-tuning"""
    import subprocess
    import os
    
    # Change to ft directory
    ft_dir = os.path.join(os.getcwd(), "ft")
    
    # Make the script executable
    train_script = os.path.join(ft_dir, "train_adapter.sh")
    subprocess.run(["chmod", "+x", train_script])
    
    # Run the training script
    result = subprocess.run(
        ["./train_adapter.sh"],
        cwd=ft_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Fine-tuning failed: {result.stderr}")
    
    return result.stdout