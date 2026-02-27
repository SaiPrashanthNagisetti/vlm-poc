import os
import json
import logging
from dotenv import load_dotenv

import chromadb
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext


# ─────────────────────────────────────────────
# LOGGING SWITCH  ← flip this to True/False
# ─────────────────────────────────────────────

ENABLE_LOGS = True

logging.basicConfig(
    level=logging.INFO if ENABLE_LOGS else logging.CRITICAL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("rag")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

load_dotenv()

MERGED_ROOT   = "merged_output"
PERSIST_DIR   = "vector_store"
COLLECTION    = "multimodal_rag"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 3

Settings.llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# ─────────────────────────────────────────────
# INDEXING  — run once to build the vector store
# ─────────────────────────────────────────────

def load_documents():
    """Read merged JSON files and return LlamaIndex Documents."""
    documents = []
    skipped   = 0

    log.info("LOAD | Scanning: %s", MERGED_ROOT)

    json_files = [f for f in os.listdir(MERGED_ROOT) if f.endswith(".json")]
    log.info("LOAD | Found %d JSON file(s)", len(json_files))

    for file in json_files:

        file_path = os.path.join(MERGED_ROOT, file)
        log.info("LOAD | Reading: %s", file)

        with open(file_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        log.info("LOAD | %s -> %d page(s)", file, len(pages))

        for page in pages:

            pdf_name    = page.get("pdf")
            page_number = page.get("page")
            text        = page.get("text", "").strip()
            keywords    = page.get("keywords", [])
            visuals     = page.get("visual_summaries", [])

            if len(text) < 50:
                log.warning("LOAD | Skipping p%s of %s — too short (%d chars)",
                            page_number, pdf_name, len(text))
                skipped += 1
                continue

            log.info("LOAD | p%-3s | %s | %d chars | %d keyword(s) | %d visual(s)",
                     page_number, pdf_name, len(text), len(keywords), len(visuals))

            # ChromaDB requires flat metadata — serialize lists to JSON strings
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "pdf":              pdf_name,
                        "page":             page_number,
                        "keywords":         json.dumps(keywords),
                        "visual_summaries": json.dumps(visuals)
                    }
                )
            )

    log.info("LOAD | Done — %d loaded, %d skipped", len(documents), skipped)
    return documents


def build_index():
    """Chunk → embed → store in ChromaDB. Run once."""
    log.info("INDEX | Starting build ...")

    documents = load_documents()
    log.info("INDEX | %d document(s) to embed", len(documents))

    chroma_client     = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    log.info("INDEX | Embedding and storing in ChromaDB ...")
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    log.info("INDEX | Done — stored at: %s", PERSIST_DIR)


# ─────────────────────────────────────────────
# QUERYING  — run anytime after index is built
# ─────────────────────────────────────────────

def query_rag(question: str, top_k: int = TOP_K):
    """Retrieve chunks from ChromaDB, generate answer via LLM, print citations."""
    log.info("QUERY | Question : %s", question)
    log.info("QUERY | top_k    : %d", top_k)

    chroma_client     = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    StorageContext.from_defaults(vector_store=vector_store)
    index             = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    log.info("QUERY | Retrieving chunks ...")
    response = query_engine.query(question)
    log.info("QUERY | LLM response received — %d chunk(s) used",
             len(response.source_nodes))
    
     # ── question ────────────────────────────────────────────────────────────────
    print("\n================ Question ================\n")
    print("Question:" question)

    # ── Answer ────────────────────────────────────────────────────────────────
    print("\n================ ANSWER ================\n")
    print(response.response)

    # ── Citations ─────────────────────────────────────────────────────────────
    # Only show chunks that actually contributed to the answer.
    # Match by checking how many words from the answer appear in the chunk.
    answer_words = set(response.response.lower().split())

    def contribution_score(chunk_text):
        """Count how many answer words appear in this chunk."""
        chunk_words = set(chunk_text.lower().split())
        return len(answer_words & chunk_words)

    # Sort nodes by contribution and keep only those above threshold
    scored_nodes = [
        (node, contribution_score(node.node.get_content()))
        for node in response.source_nodes
    ]

    # Only show chunks with meaningful overlap with the answer
    contributing = [(node, score) for node, score in scored_nodes if score >= 10]

    # Fallback: if nothing passes threshold, show the top scorer
    if not contributing:
        contributing = [max(scored_nodes, key=lambda x: x[1])]

    log.info("QUERY | %d/%d chunk(s) contributed to the answer",
             len(contributing), len(response.source_nodes))

    # Log contribution scores so you can tune the threshold
    for n, s in scored_nodes:
        log.info("SCORE | contribution=%d words | similarity=%.4f | %s p%s",
                 s,
                 n.score if n.score is not None else 0.0,
                 n.node.metadata.get("pdf", "?"),
                 n.node.metadata.get("page", "?"))

    print("\n================ CITATIONS ================\n")

    for rank, (node, contrib_score) in enumerate(contributing, 1):

        metadata   = node.node.metadata
        score      = node.score if node.score is not None else 0.0
        chunk_text = node.node.get_content()
        pdf_name   = metadata.get("pdf",  "unknown")
        page_num   = metadata.get("page", "?")
        keywords   = json.loads(metadata.get("keywords",         "[]"))
        visuals    = json.loads(metadata.get("visual_summaries", "[]"))

        log.info("NODE  | Rank %d | similarity=%.4f | contribution=%d words | %s p%s",
                 rank, score, contrib_score, pdf_name, page_num)

        print(f"-- Citation #{rank}  (similarity: {score:.4f} | contribution: {contrib_score} words)")
        print(f"   PDF  : {pdf_name}")
        print(f"   Page : {page_num}")

        if keywords:
            print(f"   Keywords : {', '.join(keywords[:5])}")

        #print(f"\n   [ Extracted text — {len(chunk_text)} chars ]\n")
        #print("   " + chunk_text.replace("\n", "\n   "))

        # Show visual citation only when the retrieved chunk contains
        # keywords from the image — meaning the answer is image-related.
        matched_visuals = []
        chunk_lower     = chunk_text.lower()

        for v in visuals:
            img_keywords = [kw.lower() for kw in v.get("image_keywords", [])]

            if not img_keywords:
                continue

            # Count how many image keywords appear in the chunk text
            hits = sum(1 for kw in img_keywords if kw in chunk_lower)

            # Match if at least 2 image keywords found in the chunk
            if hits >= 2:
                matched_visuals.append(v)

        # Filter out visuals with no meaningful data
        matched_visuals = [
            v for v in matched_visuals
            if v.get("image_title", "").strip()
            or v.get("image_citation", "").strip()
        ]

        if matched_visuals:
            print(f"\n   [ Image Citations — {len(matched_visuals)} ]\n")
            for v in matched_visuals:
                title    = v.get("image_title",    "").strip()
                path     = v.get("image_citation", "").strip()
                img_type = v.get("type",           "").strip()
                img_kws  = v.get("image_keywords", [])

                if img_type:
                    print(f"   Type        : {img_type}")
                if title:
                    print(f"   Image Title : {title}")
                if path:
                    print(f"   Image Path  : {path}")
                if img_kws:
                    print(f"   Img Keywords: {', '.join(img_kws[:6])}")

        print("\n" + "-" * 50 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — build index (run once, then comment out)
    #build_index()

    # Step 2 — query
    #query_rag("What are the benefits of injury and illness prevention programs?")
    #query_rag("what is the cost of most disabling injuries during 2002?")
    query_rag("What is EH&S  is responsible for?")
    