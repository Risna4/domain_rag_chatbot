import os
import json
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

VECTOR_STORE_DIR = "vector_store"

INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "faiss.index"
)

METADATA_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "metadata.json"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K = 4

# Minimum similarity required to consider a result relevant.
SIMILARITY_THRESHOLD = 0.35


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

def create_chunks(documents):
    """
    Split extracted page text into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        text = document["text"]

        metadata = document["metadata"]

        split_texts = splitter.split_text(text)

        for chunk_text in split_texts:

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "metadata": metadata
                    }
                )

    return chunks


# --------------------------------------------------
# CREATE FAISS VECTOR STORE
# --------------------------------------------------

def create_vector_store(chunks):
    """
    Convert chunks into embeddings and store them in FAISS.
    """

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # FAISS requires float32
    embeddings = embeddings.astype("float32")

    # Because vectors are normalized,
    # inner product is equivalent to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    # Create directory
    os.makedirs(
        VECTOR_STORE_DIR,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        INDEX_FILE
    )

    # Save text + metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    return len(chunks)


# --------------------------------------------------
# LOAD SAVED VECTOR STORE
# --------------------------------------------------

def load_vector_store():
    """
    Load the locally saved FAISS index and metadata.
    """

    if not os.path.exists(
        INDEX_FILE
    ):
        return None, None

    if not os.path.exists(
        METADATA_FILE
    ):
        return None, None

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks


# --------------------------------------------------
# SEARCH VECTOR STORE
# --------------------------------------------------

def search_vector_store(
    question,
    top_k=TOP_K
):
    """
    Retrieve the most relevant chunks for a question.

    Returns up to 4 chunks by default.
    """

    index, chunks = load_vector_store()

    if index is None or not chunks:

        return []

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = (
        question_embedding.astype("float32")
    )

    # Search FAISS
    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        score = float(score)

        # Ignore weak matches
        if score < SIMILARITY_THRESHOLD:
            continue

        result = chunks[
            index_position
        ].copy()

        result["score"] = score

        results.append(result)

    return results


# --------------------------------------------------
# DELETE VECTOR STORE
# --------------------------------------------------

def delete_vector_store():
    """
    Delete the saved FAISS index and metadata.
    """

    if os.path.exists(INDEX_FILE):

        os.remove(INDEX_FILE)

    if os.path.exists(METADATA_FILE):

        os.remove(METADATA_FILE)