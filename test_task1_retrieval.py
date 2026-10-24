
from retrieval.data_loader import MusicCatalogLoader
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever

# Step 1: Load music catalog
catalog = MusicCatalogLoader()

# Step 2: Initialize BM25
bm25 = BM25Retriever(
    catalog=catalog,
    corpus_types=[
        "track_name",
        "artist_name",
        "album_name",
        "tag_list"
    ]
)

# Step 3: Define test query
query = "energetic rock songs"

# Step 4: Test BM25
print("\n--- BM25 RESULTS ---")
bm25_ids = bm25.text_to_item_retrieval(query, topk=5)

for rank, track_id in enumerate(bm25_ids, 1):
    track = catalog.metadata_dict[track_id]
    print(rank, track.get("track_name"),
          track.get("artist_name"))

# Step 5: Test hybrid retrieval
# This initializes the Qwen3 dense model.
hybrid = HybridRetriever(bm25_retriever=bm25)

print("\n--- HYBRID RESULTS ---")
hybrid_ids = hybrid.text_to_item_retrieval(query, topk=5)

for rank, track_id in enumerate(hybrid_ids, 1):
    track = catalog.metadata_dict[track_id]
    print(rank, track.get("track_name"),
          track.get("artist_name"))

