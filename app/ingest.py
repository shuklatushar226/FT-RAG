import os, hashlib, pathlib, asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Iterator, Tuple
from git import Repo
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from app.splitter import split_preserve_code
from app.settings import settings
import numpy as np
from tqdm import tqdm
from fastapi import HTTPException

# Lazy loading to avoid startup delays
_EMB = None
_Q = None

def get_embeddings():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(settings.EMB_MODEL)
    return _EMB

def get_qdrant_client():
    global _Q
    if _Q is None:
        _Q = QdrantClient(url=settings.QDRANT_URL)
    return _Q

ALLOWED = {'.md','.rst','.txt','.py','.js','.ts','.java','.kt','.go','.rs','.cpp','.c','.cs','.php','.rb','.sh','.yml','.yaml'}
BATCH_SIZE = 32  # Process files in batches for better memory management
EMBEDDING_BATCH_SIZE = 16  # Encode embeddings in smaller batches

def ensure_collection():
    Q = get_qdrant_client()
    EMB = get_embeddings()
    try:
        Q.get_collection(settings.COLLECTION)
    except Exception:
        Q.recreate_collection(
            collection_name=settings.COLLECTION,
            vectors_config=qm.VectorParams(size=EMB.get_sentence_embedding_dimension(), distance=qm.Distance.COSINE),
        )

def file_iter(root) -> Iterator[pathlib.Path]:
    """Generator that yields valid files for processing."""
    for p in pathlib.Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in ALLOWED and p.stat().st_size < 2_000_000:
            yield p

def process_file_batch(file_paths: List[pathlib.Path], repo_name: str, repo_path: str, commit_hash: str) -> List[Tuple[str, int, str]]:
    """Process a batch of files and return chunks with metadata."""
    chunks_data = []
    
    for file_path in file_paths:
        try:
            txt = file_path.read_text(errors='ignore')
            if not txt.strip():  # Skip empty files
                continue
                
            chunks = split_preserve_code(txt, max_tokens=settings.CHUNK_TOKENS, overlap=settings.CHUNK_OVERLAP)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only process meaningful chunks
                    rel_path = str(file_path.relative_to(repo_path))
                    chunks_data.append((chunk, i, rel_path, str(file_path), commit_hash))
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
    return chunks_data

def create_embeddings_batch(chunks: List[str]) -> np.ndarray:
    """Create embeddings for a batch of chunks."""
    EMB = get_embeddings()
    return EMB.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

def batch_generator(iterable, batch_size: int):
    """Generator that yields batches from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def upsert_repo_optimized(repo_url: str, base_dir='data/repos', github_token: str = None) -> Dict[str, Any]:
    """Optimized repository ingestion with batch processing and private repo support."""
    ensure_collection()
    Q = get_qdrant_client()
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Handle private repository authentication
    authenticated_url = repo_url
    if github_token or settings.GITHUB_TOKEN:
        token = github_token or settings.GITHUB_TOKEN
        if 'github.com' in repo_url and not token in repo_url:
            # Convert https://github.com/user/repo to https://token@github.com/user/repo
            if repo_url.startswith('https://github.com'):
                authenticated_url = repo_url.replace('https://github.com', f'https://{token}@github.com')
            elif repo_url.startswith('http://github.com'):
                authenticated_url = repo_url.replace('http://github.com', f'https://{token}@github.com')
    
    # Extract repo name for folder
    name = repo_url.rstrip('/').split('/')[-1].replace('.git','')
    path = os.path.join(base_dir, name)

    # Clone or update repository
    print(f"📥 Cloning/updating repository: {name}")
    try:
        if os.path.exists(path):
            repo = Repo(path)
            # Set remote URL with authentication for private repos
            if authenticated_url != repo_url:
                repo.remotes.origin.set_url(authenticated_url)
            repo.remotes.origin.pull()
        else:
            repo = Repo.clone_from(authenticated_url, path)
            # Remove credentials from remote URL after cloning for security
            if authenticated_url != repo_url:
                repo.remotes.origin.set_url(repo_url)
    except Exception as e:
        if "Authentication failed" in str(e) or "Repository not found" in str(e):
            raise HTTPException(
                status_code=401, 
                detail=f"Authentication failed for private repository. Please provide a valid GitHub token."
            )
        raise e
    
    commit_hash = repo.head.commit.hexsha
    
    # Get all files to process
    all_files = list(file_iter(path))
    total_files = len(all_files)
    print(f"📁 Found {total_files} files to process")
    
    if total_files == 0:
        return {"repo": name, "commit": commit_hash, "points_ingested": 0}
    
    total_points = 0
    
    # Process files in batches to manage memory
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for file_batch in batch_generator(all_files, BATCH_SIZE):
            # Process file batch to get chunks
            chunks_data = process_file_batch(file_batch, name, path, commit_hash)
            
            if not chunks_data:
                pbar.update(len(file_batch))
                continue
            
            # Create embeddings in smaller batches
            points = []
            chunks_text = [chunk_data[0] for chunk_data in chunks_data]
            
            for i in range(0, len(chunks_text), EMBEDDING_BATCH_SIZE):
                batch_chunks = chunks_text[i:i + EMBEDDING_BATCH_SIZE]
                batch_metadata = chunks_data[i:i + EMBEDDING_BATCH_SIZE]
                
                # Create embeddings for this batch
                embeddings = create_embeddings_batch(batch_chunks)
                
                # Create points for this embedding batch
                for j, (chunk_text, chunk_id, rel_path, full_path, commit) in enumerate(batch_metadata):
                    uid_hash = hashlib.sha1(f"{name}:{rel_path}:{chunk_id}".encode()).hexdigest()
                    uid = int(uid_hash[:8], 16)
                    
                    points.append(qm.PointStruct(
                        id=uid,
                        vector=embeddings[j].tolist(),
                        payload={
                            'repo': name,
                            'path': rel_path,
                            'full_path': full_path,
                            'chunk_id': chunk_id,
                            'commit': commit,
                            'page_content': chunk_text,
                        }
                    ))
            
            # Upsert points to Qdrant
            if points:
                Q.upsert(collection_name=settings.COLLECTION, points=points)
                total_points += len(points)
                print(f"📤 Uploaded batch: {len(points)} chunks")
            
            pbar.update(len(file_batch))
    
    print(f"✅ Repository '{name}' ingested successfully!")
    print(f"📊 Total chunks processed: {total_points}")
    
    return {"repo": name, "commit": commit_hash, "points_ingested": total_points}

# Keep the original function for backward compatibility
def upsert_repo(repo_url: str, base_dir='data/repos') -> Dict[str, Any]:
    """Legacy function - routes to optimized version."""
    return upsert_repo_optimized(repo_url, base_dir)