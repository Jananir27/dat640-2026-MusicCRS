"""Run from MusicCRS project root after applying task2_hybrid_evidence.patch."""
from retrieval.hybrid import HybridRetriever
from retrieval.data_loader import MusicCatalogLoader

catalog = MusicCatalogLoader()
hybrid = HybridRetriever()
query = "energetic rock songs"
results = hybrid.retrieve_with_evidence(query, topk=5)

assert len(results) == 5
assert len({r["track_id"] for r in results}) == len(results)
assert all(abs(r["hybrid_score"] - (
    r["bm25_rrf_contribution"] + r["dense_rrf_contribution"]
)) < 1e-12 for r in results)

for position, row in enumerate(results, 1):
    track = catalog.metadata_dict[row["track_id"]]
    print(f"{position}. {track.get('track_name')} — {track.get('artist_name')}")
    print(f"   BM25 rank={row['bm25_rank']} "
          f"Dense rank={row['dense_rank']} "
          f"RRF score={row['hybrid_score']:.6f}")

print("\nTask 2 evidence check passed!")

original_ids = hybrid.text_to_item_retrieval(query, topk=5)
evidence_ids = [item["track_id"] for item in results]

assert original_ids == evidence_ids, (
    "Evidence method changed the ranking!"
)

print("Original ranking preserved!")
