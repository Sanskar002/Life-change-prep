# Day 4 Quick Reference Notes: Hybrid RRF Code, Bitwise Quantization, TurboQuant & Qdrant API

*Concise engineering reference sheet for rapid review of implementation patterns and system behavior.*

---

## 1. Core Code Patterns in 5 Lines Each

### A. Reciprocal Rank Fusion (RRF) Loop
```python
rrf_scores = {}
for rank, (doc_id, _) in enumerate(dense_hits, start=1):  # MUST be 1-based (start=1)
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (w_dense / (60 + rank))
for rank, (doc_id, _) in enumerate(bm25_hits, start=1):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (w_bm25 / (60 + rank))
```

### B. Binary Quantization (`np.packbits` + Bitwise `XOR` + `POPCOUNT`)
```python
# 1. Center around corpus mean and pack 8 boolean signs into 1 uint8 byte (32x compression)
db_packed = np.packbits((db_fp32 - corpus_mean) > 0.0, axis=1)
q_packed  = np.packbits((q_fp32  - corpus_mean) > 0.0)

# 2. Fast Hamming distance via XOR + 256-entry POPCOUNT lookup table
hamming_dists = POPCOUNT_TABLE[np.bitwise_xor(db_packed, q_packed)].sum(axis=1)
```

### C. Google TurboQuant 3-Step Pipeline
```python
z = R @ x                                         # 1. Orthogonal rotation smooths outlier spikes
r, theta = np.hypot(zx, zy), np.arctan2(zy, zx)   # 2. PolarQuant bounds angles in [-pi, pi]
qjl_bits = np.where((S @ (z - z_polar)) >= 0, 1, -1) # 3. 1-bit QJL cancels residual dot-product bias
```

---

## 2. Production Qdrant Hybrid Search Checklist

| Component / Setting | Exact API Usage | Why It Matters |
| :--- | :--- | :--- |
| **Sparse IDF Calculation** | `SparseVectorParams(modifier=models.Modifier.IDF)` | Offloads corpus-wide Inverse Document Frequency calculation to the Qdrant server so clients only send local term-frequency counts. |
| **Parallel Sub-Queries** | `prefetch=[Prefetch(using="dense", limit=100), Prefetch(using="sparse", limit=100)]` | Executes both dense HNSW traversal and sparse inverted-index lookup concurrently inside a single network round-trip. |
| **Prefetch Cutoff Rule** | Set `prefetch.limit` to **$5\times\text{–}10\times$** your final `limit` (e.g., `100` for Top `10`) | Fusion only reorders the union of what the prefetches return. If `prefetch.limit` is too small (e.g., `10`), consensus documents sitting at Rank #12 in both lists get cut off before RRF sees them! |
| **Global Filter Propagation** | Pass `query_filter=rbac_filter` at the root `query_points()` level | Qdrant automatically pushes the top-level RBAC filter down into **every** nested `Prefetch` stage, preventing unauthorized data leakage. |
| **RRF vs. DBSF** | `Fusion.RRF` (Rank-based) vs. `Fusion.DBSF` (Distribution-Based Score Fusion) | `RRF` uses only ordinal ranks ($1/(60+r)$); `DBSF` normalizes each prefetch list using $\mu \pm 3\sigma$ (mean and standard deviation) when score gaps matter. |

---

## 3. Important Questions (Code & Systems Deep-Dive)

**Q1: When implementing Binary Quantization in Python/NumPy, why is subtracting the corpus mean ($\tilde{x} = x - \mu$) before `np.packbits` critical?**  
*Many pretrained bi-encoders suffer from **anisotropy**—their embeddings occupy a narrow cone in vector space where certain dimensions are positive (e.g., $>0.2$) for 99% of all documents. If you binarize at threshold `0.0` without subtracting the corpus mean vector $\mu$, those dimensions become a constant `1` bit across every document, wasting bits and collapsing Hamming distance discrimination.*

**Q2: In Qdrant Hybrid Search, what happens if you set `prefetch.limit = 10` when requesting `limit = 10` final results?**  
*You suffer from **Prefetch Cutoff Starvation**. RRF's biggest strength is boosting a consensus document that ranks #15 in Dense and #15 in BM25 above a noisy outlier that ranks #1 in only one retriever. If `prefetch.limit` is capped at 10, that #15 consensus document is dropped by both prefetches before `FusionQuery` ever runs. Always oversample prefetches to `50–100` candidates for a Top-10 output.*

**Q3: Why does Google's TurboQuant use a 2-stage split (PolarQuant + 1-Bit QJL) instead of spending all 3 bits on PolarQuant?**  
*Spending all 3 bits on deterministic quantization (PolarQuant alone) minimizes reconstruction error ($||x - \hat{x}||_2$), but still leaves a small **systematic directional bias** in the residual error. When computing dot products or attention logits across thousands of keys, that bias compounds and shifts rankings. Allocating 2 bits to PolarQuant and reserving the final **1 bit for randomized QJL sign projection** guarantees a mathematically **unbiased inner-product estimator** ($\mathbb{E}[\langle q, \hat{x} \rangle] = \langle q, x \rangle$).*

**Q4: What is the difference between `Fusion.RRF` and `Fusion.DBSF` (Distribution-Based Score Fusion), and when would you switch?**  
*`RRF` discards raw scores completely and only looks at ordinal ranks (`1, 2, 3...`), making it immune to extreme BM25 outliers. However, `RRF` cannot tell if Rank #1 beat Rank #2 by a tiny margin (`0.851` vs `0.850`) or a massive margin (`0.95` vs `0.42`). `DBSF` preserves score gaps by normalizing each retriever's scores per query using a 3-standard-deviation window ($\text{mean} \pm 3\sigma$) before summing—useful when confidence gaps within a single retriever need to be preserved.*
