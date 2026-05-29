import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class ModelFactory:
    """Dynamically identifies the provider based on the syntax of the user's API key."""

    @staticmethod
    def identify_provider(api_key: str) -> str:
        api_key = api_key.strip()
        if api_key.startswith("sk-"):
            return "openai"
        elif api_key.startswith("gsk_"):
            return "groq"
        else:
            raise ValueError("Unrecognized API Key format. Ensure you've supplied a valid provider string.")

    @classmethod
    def get_llm(cls, api_key: str):
        provider = cls.identify_provider(api_key)

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)

        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0)

    @staticmethod
    def get_embeddings():
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_youtube_video(url: str, embeddings_model):
    """Downloads and tokenizes YouTube transcripts into a local FAISS vector instance."""
    try:
        # FIX: Added language parameter supporting standard 'en' and regional 'en-US'
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["en", "en-US"]
        )
        docs = loader.load()

        if not docs:
            raise ValueError("No transcript returned for this video target.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        return FAISS.from_documents(chunks, embeddings_model)
    except Exception as e:
        raise RuntimeError(f"Ingest Engine Failure: {str(e)}")