from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": settings.hf_device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    settings.storage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(settings.storage_dir),
    )


def get_vector_store(
    collection_name: str | None = None,
) -> QdrantVectorStore:
    name = collection_name or settings.qdrant_collection

    return QdrantVectorStore(
        client=get_client(),
        collection_name=name,
        embedding=get_embeddings(),
    )


INDEXED_PAYLOAD_FIELDS: dict[str, qmodels.PayloadSchemaType] = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.chunk_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
}


def ensure_collection(
    recreate: bool = False,
    collection_name: str | None = None,
) -> None:
    client = get_client()
    name = collection_name or settings.qdrant_collection

    exists = client.collection_exists(
        collection_name=name,
    )

    if exists and recreate:
        client.delete_collection(
            collection_name=name,
        )
        exists = False

    if not exists:
        embedding_dimension = len(get_embeddings().embed_query("dimension probe"))

        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )

    collection_info = client.get_collection(collection_name=name)

    payload_schema = collection_info.payload_schema or {}

    for field_name, field_schema in INDEXED_PAYLOAD_FIELDS.items():
        if field_name not in payload_schema:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_schema,
            )