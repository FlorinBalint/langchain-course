import asyncio
import os
import ssl

from typing import Any, Dict, List

import aiohttp
import certifi
import langchain_core
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
from langchain_classic import text_splitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.tracers import langchain
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_info, log_warning, log_error, log_success, log_header)

load_dotenv()

ssk_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False,
    chunk_size=50, retry_min_seconds=10,
)

# chrome
vectorstore = PineconeVectorStore(
     index_name="langchain-doc-index", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=5, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int) -> None:
    """ Process documents in taches asynchronously """
    log_header("Vector storage phase")

    log_info(
        f"VectorStore indexing: Preparing to add {len(documents)} documents..."
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore indexing: Adding {len(batches)} batches of {batch_size} each..."
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"Successfully added batch {batch_num} to VectorStore.",
            )
        except Exception as e:
            log_error(f"Vectorstore indexing: Failed to add batch {batch_num} :- {e}")
            return False
        return True

    # process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(
            f"Successfully indexed {len(batches)} documents to VectorStore.",
        )
    else:
        log_warning(
            f"Vectorstore indexing: Processed {successful} / {len(batches )} documents to VectorStore.",
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com",
        Colors.PURPLE
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "max_depth": 1,
            "extract_depth": "advanced",
            "instructions": "content on ai agents",
        }
    )

    all_docs = [
        Document(
            page_content=r['raw_content'],
            metadata={"source": r["url"]},
        ) for r in res['results']
    ]

    log_success(
        f"Tavily Crawl sucesfully crawled {len(all_docs)} URLS from docs."
    )

    #Split documents into chunks
    log_header("DOCUMENT CHUNKER PHASE")

    log_info(
        f"Text splitter, processing {len(all_docs)} documents with 4000 chunk size"
        f"and 200 overalp",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text splitter: created {len(splitted_docs)} chunks from {len(all_docs)} documents",
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)



if __name__ == "__main__":
    asyncio.run(main())
