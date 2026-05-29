import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Replace the failing core import with this precise community path
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ingest import ModelFactory, process_youtube_video

app = FastAPI(title="RAG Diagnostic Query Engine Server")

# In-memory session tracking for vector stores
retriever_cache = {"vector_store": None, "current_url": ""}


class RAGRequest(BaseModel):
    api_key: str
    youtube_url: str
    question: str


@app.post("/api/v1/query")
async def execute_rag_pipeline(payload: RAGRequest):
    if not payload.api_key.strip():
        raise HTTPException(status_code=400, detail="Authorization token missing.")

    try:
        provider = ModelFactory.identify_provider(payload.api_key)
        embeddings = ModelFactory.get_embeddings()

        # 1. Thread-safe Vector Management
        if retriever_cache["current_url"] != payload.youtube_url or retriever_cache["vector_store"] is None:
            retriever_cache["vector_store"] = process_youtube_video(payload.youtube_url, embeddings)
            retriever_cache["current_url"] = payload.youtube_url

        retriever = retriever_cache["vector_store"].as_retriever(search_kwargs={"k": 3})

        # 2. Dynamic Model Execution
        llm = ModelFactory.get_llm(payload.api_key)

        template = """Answer the question based only on the following context derived from the video transcript:
        {context}

        Question: {question}
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        # 3. Running Telemetry Metrics Capture
        start_time = time.time()
        with get_openai_callback() as cb:
            response = chain.invoke(payload.question)
        end_time = time.time()

        elapsed_time = round(end_time - start_time, 2)
        total_tokens = cb.total_tokens
        tokens_per_sec = round(total_tokens / elapsed_time, 1) if elapsed_time > 0 and total_tokens > 0 else "N/A"
        computed_cost = f"${cb.total_cost:.5f}" if cb.total_cost > 0 else "Credits / Comped Tier"

        return {
            "status": "success",
            "data": response,
            "metrics": {
                "latency": f"{elapsed_time}s",
                "tokens": f"{total_tokens}",
                "throughput": f"{tokens_per_sec}/s",
                "cost": computed_cost
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Boot production Uvicorn web server instance
    uvicorn.run(app, host="127.0.0.1", port=8000)