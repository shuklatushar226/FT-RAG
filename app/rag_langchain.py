from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from app.settings import settings

# Lazy loading
_emb = None

def get_embeddings():
    global _emb
    if _emb is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _emb = HuggingFaceEmbeddings(model_name=settings.EMB_MODEL)
    return _emb

def build_retriever(k=6):
    emb = get_embeddings()
    client = QdrantClient(url=settings.QDRANT_URL)
    vs = Qdrant(client=client, collection_name=settings.COLLECTION, embeddings=emb)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever

def make_qa_chain(llm, k=6):
    retriever = build_retriever(k=k)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa