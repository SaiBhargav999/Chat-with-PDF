import os
import time
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, ValidationError

from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

#
# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
#
class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL_NAME: Literal["gpt-3.5-turbo", "gpt-4"] = Field(
        "gpt-3.5-turbo", env="OPENAI_MODEL_NAME"
    )
    DOCUMENT_DIR: str = Field("documents", env="DOCUMENT_DIR")
    MAX_CHUNK_SIZE: int = Field(2048, env="MAX_CHUNK_SIZE")
    MAX_CHUNK_OVERLAP: int = Field(1024, env="MAX_CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except ValidationError as e:
    raise RuntimeError(f"Configuration error: {e}")

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

#
# ─── LOGGING ─────────────────────────────────────────────────────────────────────
#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("talk-with-us-api")

#
# ─── APP SETUP ────────────────────────────────────────────────────────────────────
#
app = FastAPI(
    title="Talk With Us API",
    description="RAG-powered resume Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#
# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────────
#
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's natural-language question")
    chunk_size: int = Field(
        512,
        ge=1,
        le=settings.MAX_CHUNK_SIZE,
        description="Token count per chunk for retrieval",
    )
    chunk_overlap: int = Field(
        50,
        ge=0,
        le=settings.MAX_CHUNK_OVERLAP,
        description="Overlap tokens between chunks",
    )
    model: Literal["gpt-3.5-turbo", "gpt-4"] = Field(
        settings.OPENAI_MODEL_NAME,
        description="OpenAI model to use",
    )

class QueryResponse(BaseModel):
    answer: str
    latency: float

#
# ─── ENDPOINT ────────────────────────────────────────────────────────────────────
#
@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    logger.info("Incoming query: %r", request.question)
    t0 = time.perf_counter()

    # 1. Load documents
    doc_dir = settings.DOCUMENT_DIR
    if not os.path.isdir(doc_dir):
        logger.error("Document directory %r not found", doc_dir)
        raise HTTPException(500, "Document directory not found")

    try:
        documents = SimpleDirectoryReader(doc_dir).load_data()
    except Exception as e:
        logger.exception("Failed to load documents")
        raise HTTPException(500, "Could not read documents")

    # 2. Build RAG pipeline
    try:
        llm = OpenAI(model=request.model)  # uses env API key
        svc_ctx = ServiceContext.from_defaults(
            llm=llm,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        index = VectorStoreIndex.from_documents(documents, service_context=svc_ctx)
        query_engine = index.as_query_engine()
        result = query_engine.query(request.question)
        answer = str(result)
    except Exception as e:
        logger.exception("LLM/RAG pipeline error")
        raise HTTPException(500, "Failed to generate answer")

    latency = time.perf_counter() - t0
    logger.info("Answered in %.3fs", latency)

    return QueryResponse(answer=answer, latency=latency)