# Day 4 Complete Technical Implementation Guide: BM25, Reciprocal Rank Fusion (RRF), Binary/TurboQuant Math & Qdrant Hybrid Search

**Scope:** Step-by-Step Python/NumPy Implementations of Hybrid Retrieval, Bitwise SIMD Quantization, Google TurboQuant & Single-Call Server-Side Fusion  
**Prerequisites:** None — every algorithm and line of code is explained from first principles with toy numbers.  
**Reading Time:** 50 Minutes (100% self-contained)

---

## The Big Picture: What Are We Building Today?

In Day 3, we learned the concepts behind **Hybrid Search (BM25 + Dense Vectors + RRF)** and **Vector Compression (Binary Quantization + Google TurboQuant)**. Today, we turn every one of those concepts into clean, runnable Python and NumPy code so you can see exactly how the numbers move under the hood:

1. **Block 1 (BM25 + RRF in Pure Python/NumPy):** How to build a fast keyword search engine from scratch and fuse its rankings with dense vector search in 15 lines of code.
2. **Block 2 (Binary Quantization & Rescoring in NumPy):** How to pack 32-bit decimal numbers into 1-bit `uint8` bytes ($32\times$ smaller) and score them using bitwise `XOR` + `POPCOUNT`.
3. **Block 3 (Google TurboQuant in NumPy):** How to code the 3-step TurboQuant pipeline (**Random Rotation $\rightarrow$ PolarQuant Angle Compression $\rightarrow$ 1-Bit QJL Error Correction**) and watch it fix spiky outlier vectors live.
4. **Block 4 (Production Qdrant Hybrid Search):** How to configure named Dense + Sparse (BM25) vectors in Qdrant and run parallel `Prefetch` + `RRF` fusion with RBAC filters in a single API call.

---

## Block 1: Coding BM25 & Reciprocal Rank Fusion (RRF) from Scratch (Minutes 0–15)

### 1.1 How BM25 Works in Code (Step-by-Step Before You Read the Code)
Suppose we have 3 short documents:
* `Doc 0`: `"bms thermal limit is 45A"` (5 words)
* `Doc 1`: `"bms error 0x84f2 occurs at -20C"` (6 words)
* `Doc 2`: `"error 0x84f2 log error 0x84f2"` (5 words — repeats `0x84f2` twice)

To score a query like `"0x84f2 thermal"` without scanning every word of every document slowly, our code pre-builds three simple lookup tables during setup (`__init__`):
1. **`doc_lengths`:** A list storing how many words each document has: `[5, 6, 5]`. The average length (`avgdl`) is $\frac{5+6+5}{3} = 5.33$.
2. **`term_freqs`:** For each document, a dictionary counting how many times each word appears (e.g., for `Doc 2`, `{"error": 2, "0x84f2": 2}`).
3. **`idf` (Rarity Score per Word):** Computed once for every unique word using:
   $$\text{IDF}(w) = \ln\left(1 + \frac{N - \text{docs\_with\_word} + 0.5}{\text{docs\_with\_word} + 0.5}\right)$$
   Rare words like `"thermal"` (in 1 doc) get a high IDF score ($\approx 0.98$), while common words get a low score.

---

### 1.2 Complete, Runnable Code: Pure Python/NumPy `BM25Index` + `HybridRRFRanker`

```python
import math
import re
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np


class BM25Index:
    """
    Clean, dependency-free BM25 implementation using precomputed IDF and length normalization.
    """

    def __init__(self, documents: List[str], k1: float = 1.2, b: float = 0.4):
        """
        Args:
            documents: List of raw chunk strings.
            k1: Term frequency saturation (1.2 caps repetition bonus at 1.2 + 1 = 2.2x).
            b: Document length penalty (0.4 is ideal for uniform RAG chunks).
        """
        self.documents = documents
        self.num_docs = len(documents)
        self.k1 = k1
        self.b = b

        self.doc_lengths: List[int] = []
        self.term_freqs: List[Counter] = []
        self.doc_freqs: Counter = Counter()  # How many docs contain each word
        self.idf: Dict[str, float] = {}

        self._build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases and splits text into alphanumeric tokens (preserves hex codes like 0x84f2)."""
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _build_index(self) -> None:
        total_length = 0
        for doc in self.documents:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            counts = Counter(tokens)
            self.term_freqs.append(counts)

            # Increment document frequency once per unique word in this document
            for unique_word in counts.keys():
                self.doc_freqs[unique_word] += 1

        self.avgdl = total_length / max(1, self.num_docs)

        # Precompute Robertson-Sparck Jones IDF for every unique word in corpus
        for word, df in self.doc_freqs.items():
            self.idf[word] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Scores all documents against the query and returns [(doc_id, bm25_score), ...] sorted highest first.
        """
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.num_docs, dtype=np.float64)

        for token in query_tokens:
            if token not in self.idf:
                continue  # Skip query words that never appear in our corpus
            token_idf = self.idf[token]

            for doc_id in range(self.num_docs):
                freq = self.term_freqs[doc_id].get(token, 0)
                if freq == 0:
                    continue

                doc_len = self.doc_lengths[doc_id]
                # BM25 core saturation + length normalization formula
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_id] += token_idf * (numerator / denominator)

        # Sort document IDs by score descending, keeping only docs with score > 0
        ranked_indices = np.argsort(scores)[::-1]
        return [
            (int(idx), round(float(scores[idx]), 4))
            for idx in ranked_indices[:top_k]
            if scores[idx] > 0.0
        ]


class HybridRRFRanker:
    """
    Combines Dense Vector Search and Sparse BM25 Search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k: int = 60, dense_weight: float = 1.0, bm25_weight: float = 1.0):
        """
        Args:
            k: Smoothing constant (default 60 prevents single-list #1 outliers from dominating).
            dense_weight: Relative weight for semantic vector rankings.
            bm25_weight: Relative weight for lexical BM25 rankings.
        """
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    @staticmethod
    def dense_cosine_search(
        query_vec: np.ndarray, doc_matrix: np.ndarray, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Computes exact batched cosine similarity using L2-normalized matrix-vector multiplication.
        """
        # Normalize query (1D) and document matrix (2D: [num_docs, dim])
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        d_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9)

        # Single SIMD-accelerated matrix-vector dot product
        similarities = d_norms @ q_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), round(float(similarities[idx]), 4)) for idx in top_indices]

    def fuse(
        self,
        dense_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """
        Applies RRF(d) = w_dense / (k + rank_dense) + w_bm25 / (k + rank_bm25).
        Note: Ranks are 1-based (1, 2, 3...).
        """
        rrf_scores: Dict[int, float] = {}
        rank_provenance: Dict[int, Dict[str, object]] = {}

        # 1. Accumulate Dense Ranks (1-based: enumerate starts at rank=1)
        for rank, (doc_id, raw_score) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (self.dense_weight / (self.k + rank))
            rank_provenance.setdefault(doc_id, {})["dense_rank"] = rank
            rank_provenance[doc_id]["dense_raw"] = raw_score

        # 2. Accumulate BM25 Ranks (1-based: enumerate starts at rank=1)
        for rank, (doc_id, raw_score) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (self.bm25_weight / (self.k + rank))
            rank_provenance.setdefault(doc_id, {})["bm25_rank"] = rank
            rank_provenance[doc_id]["bm25_raw"] = raw_score

        # 3. Sort by fused RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        return [
            {
                "doc_id": doc_id,
                "rrf_score": round(score, 5),
                "dense_rank": rank_provenance[doc_id].get("dense_rank", None),
                "bm25_rank": rank_provenance[doc_id].get("bm25_rank", None),
            }
            for doc_id, score in sorted_docs
        ]


# --- Verification Demo ---
if __name__ == "__main__":
    corpus = [
        "Overview of general electric vehicle battery cooling systems.",              # Doc 0: Semantic match only
        "The ID.4 BMS controller triggers error 0x84f2 when cold current > 45A.",   # Doc 1: Consensus Match!
        "System log dump: error 0x84f2 repeated in module 3, error 0x84f2 cleared.", # Doc 2: Keyword-stuffed log
    ]
    bm25 = BM25Index(corpus, k1=1.2, b=0.4)
    bm25_hits = bm25.search("ID.4 battery thermal error 0x84f2", top_k=3)

    # Mock 4D embeddings where Doc 0 is #1 in Dense, and Doc 1 is #2 in Dense
    doc_vecs = np.array([
        [0.95, 0.10, 0.00, 0.00],  # Doc 0 (Dense #1)
        [0.90, 0.20, 0.00, 0.00],  # Doc 1 (Dense #2, BM25 #1)
        [0.10, 0.90, 0.00, 0.00],  # Doc 2 (Dense #3, BM25 #2)
    ])
    q_vec = np.array([0.94, 0.12, 0.00, 0.00])

    ranker = HybridRRFRanker(k=60)
    dense_hits = ranker.dense_cosine_search(q_vec, doc_vecs, top_k=3)
    fused = ranker.fuse(dense_hits, bm25_hits, top_k=3)
    print("Fused RRF Ranking:", fused)
```

#### Two Subtle Code Details to Notice:
1. **`enumerate(results, start=1)`:** Python indices start at `0`, os if you forget `start=1`, Rank #1 gets `rank = 0` (`1 / (60 + 0)` instead of `1 / (60 + 1)`). Always use 1-based ranks in RRF.
2. **Preventing Zero-Division in Cosine Similarity:** Adding `+ 1e-9` to `np.linalg.norm(...)` guarantees that an all-zero vector never crashes the pipeline with a `NaN` division-by-zero error.

---

## Block 2: Coding Binary Quantization (BQ) & Two-Stage Rescoring in NumPy (Minutes 15–30)

### 2.1 How `np.packbits` and `XOR` Work Step-by-Step
Suppose we have an 8-number decimal vector:
$$x = [+0.4, \; -0.2, \; +0.9, \; +0.1, \; -0.8, \; -0.5, \; +0.3, \; -0.1]$$

1. **Step 1 — Binarize (`x > 0`):** Turn positive numbers into `1` and negative/zero numbers into `0`:
   $$\text{Bits} = [1, \; 0, \; 1, \; 1, \; 0, \; 0, \; 1, \; 0]$$
   * **Wait!** In Python/NumPy, a boolean array `[True, False, ...]` still uses **1 full Byte (8 bits) per boolean**! If you don't pack the bits together, you only get $4\times$ compression instead of $32\times$ compression.
2. **Step 2 — Bit Packing (`np.packbits`):** `np.packbits` packs **8 separate `0/1` bits inside a single 1-Byte `uint8` integer**:
   $$[1, 0, 1, 1, 0, 0, 1, 0]_2 \xrightarrow{\text{np.packbits}} \mathbf{178} \quad (\text{1 single byte in RAM!})$$
   For a 1024-dimension FP32 vector ($4,096\text{ Bytes}$), `np.packbits` compresses it into $\frac{1024}{8} = \mathbf{128\text{ Bytes}}$ ($32\times$ RAM reduction).
3. **Step 3 — Fast Bitwise Hamming Distance (`XOR` + `POPCOUNT`):**  
   To compare two packed `uint8` bytes, we apply bitwise `XOR` (`q_packed ^ db_packed`), which puts a `1` bit wherever the two vectors disagree, and then look up the number of `1` bits in a precomputed 256-entry `POPCOUNT` table.

---

### 2.2 Complete, Runnable Code: `BinaryQuantizerIndex` with Mean-Centering & Rescoring

```python
from typing import List, Tuple
import numpy as np


# Precomputed lookup table: POPCOUNT_TABLE[b] = number of '1' bits in byte b (0..255)
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


class BinaryQuantizerIndex:
    """
    Implements 32x compressed Binary Quantization with:
    1. Corpus Mean-Centering (fixes anisotropic embedding offset)
    2. Bit-packing via np.packbits (8 dimensions per uint8 byte)
    3. Two-Stage Search: Fast Bitwise XOR+POPCOUNT Shortlist -> Exact FP32 Rescoring
    """

    def __init__(self, full_precision_vectors: np.ndarray):
        """
        Args:
            full_precision_vectors: Shape [num_docs, dim], where dim is a multiple of 8 (e.g., 768, 1024).
        """
        assert full_precision_vectors.shape[1] % 8 == 0, "Dimension must be divisible by 8 for byte packing."
        # Normalize vectors so dot product == cosine similarity during Stage 2 rescoring
        norms = np.linalg.norm(full_precision_vectors, axis=1, keepdims=True) + 1e-9
        self.fp32_vectors = (full_precision_vectors / norms).astype(np.float32)

        # Compute corpus mean vector to center each dimension around 0.0 before binarizing
        self.corpus_mean = np.mean(self.fp32_vectors, axis=0, keepdims=True)

        # Center and pack into 1-bit uint8 matrix of shape [num_docs, dim // 8]
        centered_docs = self.fp32_vectors - self.corpus_mean
        self.binary_codes = np.packbits(centered_docs > 0.0, axis=1)

    def memory_savings_report(self) -> Dict[str, float]:
        fp32_bytes = self.fp32_vectors.nbytes
        binary_bytes = self.binary_codes.nbytes
        return {
            "fp32_bytes": fp32_bytes,
            "binary_bytes": binary_bytes,
            "compression_ratio": round(fp32_bytes / binary_bytes, 1),
        }

    def search(
        self, query_vec: np.ndarray, top_k: int = 5, oversampling_factor: int = 3
    ) -> List[Tuple[int, float, int]]:
        """
        Executes Two-Stage Quantized Retrieval:
        Stage 1: Bitwise Hamming Distance over in-RAM binary codes to shortlist (oversampling * top_k).
        Stage 2: Exact cosine rescoring on only the shortlisted candidates.
        """
        # Normalize and center the query vector using the exact same corpus mean
        q_norm = (query_vec / (np.linalg.norm(query_vec) + 1e-9)).astype(np.float32)
        q_centered = q_norm - self.corpus_mean[0]
        q_packed = np.packbits(q_centered > 0.0)  # Shape: [dim // 8]

        # STAGE 1: Bitwise XOR across all docs, then sum set bits via POPCOUNT lookup
        xor_bytes = np.bitwise_xor(self.binary_codes, q_packed)  # Shape: [num_docs, dim // 8]
        hamming_distances = POPCOUNT_TABLE[xor_bytes].sum(axis=1) # Lower distance = more similar!

        shortlist_size = min(len(self.fp32_vectors), top_k * oversampling_factor)
        # argpartition finds the lowest Hamming distances in O(N) time without sorting the whole array
        shortlist_ids = np.argpartition(hamming_distances, shortlist_size - 1)[:shortlist_size]

        # STAGE 2: Fetch ONLY shortlisted FP32 vectors (in production, via SSD mmap) & rescore
        shortlist_fp32 = self.fp32_vectors[shortlist_ids]
        exact_scores = shortlist_fp32 @ q_norm

        # Sort the shortlisted candidates by exact cosine score descending
        best_local_order = np.argsort(exact_scores)[::-1][:top_k]
        return [
            (
                int(shortlist_ids[i]),
                round(float(exact_scores[i]), 4),
                int(hamming_distances[shortlist_ids[i]]),
            )
            for i in best_local_order
        ]


# --- Verification Demo ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Simulate 1,000 vectors of dimension 1024
    mock_db = rng.standard_normal((1000, 1024), dtype=np.float32)
    bq_index = BinaryQuantizerIndex(mock_db)

    print("Memory Report:", bq_index.memory_savings_report())
    # Query with a slightly noisy version of Document #42
    query = mock_db[42] + rng.normal(0, 0.1, size=1024).astype(np.float32)
    hits = bq_index.search(query, top_k=3, oversampling_factor=4)
    print("Top-3 Rescored Hits (doc_id, exact_cosine, hamming_dist):", hits)
```

---

## Block 3: Coding Google TurboQuant (Rotation + PolarQuant + 1-Bit QJL) in NumPy (Minutes 30–42)

### 3.1 Tracing the 3 Steps on a Spiky Vector (Zero-Prerequisite Walkthrough)
Remember from your Day 3 Addendum why standard low-bit quantization fails on raw vectors: **activation outliers** (e.g., one dimension is `+18.5` while all other dimensions are tiny numbers like `0.03`).

Let's trace how our code implements **Google's 3-Bit TurboQuant** (2 bits for **PolarQuant** + 1 bit for **QJL Error Correction**) without any dataset training:

1. **Step 1 — Orthogonal Random Rotation ($z = R \cdot x$):**
   We generate a random orthogonal rotation matrix $R$ (where $R^T R = I$, meaning rotating two vectors never changes the angle or dot product between them). Multiplying spiky vector $x$ by $R$ blends the `+18.5` spike evenly across all dimensions so every coordinate becomes smooth and balanced.
2. **Step 2 — PolarQuant (2 Bits per Coordinate via Polar Angles):**
   We group the rotated coordinates into pairs $(z_{2i}, z_{2i+1})$ and convert each pair from $(X, Y)$ grid coordinates into a **Radius** $r_i = \sqrt{z_{2i}^2 + z_{2i+1}^2}$ and a **Compass Angle** $\theta_i = \text{atan2}(z_{2i+1}, z_{2i}) \in [-\pi, \pi]$.
   * Because angles are strictly bounded between $-\pi$ and $+\pi$, we quantize each angle into uniform buckets without needing per-block min/max metadata, and reconstruct the primary approximation $\hat{z}_{\text{polar}}$.
3. **Step 3 — 1-Bit QJL (Quantized Johnson-Lindenstrauss Residual Sign Bit):**
   We compute the tiny rounding error left behind by Step 2:
   $$e = z - \hat{z}_{\text{polar}}$$
   Instead of throwing that error away, we project $e$ with a random Gaussian matrix $S$ and store **just 1 sign bit (`+1` or `-1`) per dimension**: $q_{\text{qjl}} = \text{sign}(S \cdot e)$.
   When computing the dot product with a query vector $q$, adding the scaled 1-bit QJL correction cancels out the rounding bias!

---

### 3.2 Complete, Runnable Code: `TurboQuantizer` vs. Naive Quantization on Spiky Vectors

```python
import numpy as np


class TurboQuantizer:
    """
    Educational, runnable NumPy implementation of Google's TurboQuant pipeline:
    1. Orthogonal Random Rotation (blends spiky outlier dimensions with zero training)
    2. PolarQuant (converts coordinate pairs to polar radii & quantized angles)
    3. 1-Bit QJL (Quantized Johnson-Lindenstrauss sign-bit residual bias correction)
    """

    def __init__(self, dim: int = 64, angle_bits: int = 4, seed: int = 2026):
        assert dim % 2 == 0, "Dimension must be even to pair Cartesian (x, y) into Polar (r, theta)."
        self.dim = dim
        self.angle_levels = 2 ** angle_bits  # Number of discrete angle buckets in [-pi, pi]

        rng = np.random.default_rng(seed)
        # 1. Pre-generate fixed Orthogonal Rotation Matrix R via QR decomposition (R.T @ R == I)
        random_matrix = rng.standard_normal((dim, dim))
        q_mat, _ = np.linalg.qr(random_matrix)
        self.R = q_mat.astype(np.float32)

        # 2. Pre-generate QJL projection matrix S ~ N(0, 1) for 1-bit residual sign checking
        self.S = rng.standard_normal((dim, dim)).astype(np.float32)

    def compress_and_reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Compresses vector x using Rotation + PolarQuant + 1-Bit QJL and returns the
        unbiased reconstructed vector in the original coordinate space.
        """
        # --- STEP 1: Orthogonal Random Rotation (Smooths out spiky outliers) ---
        z = self.R @ x  # Exact norm and inner-product preserving rotation

        # --- STEP 2: PolarQuant (Cartesian pairs -> Polar Radius + Quantized Angle) ---
        z_pairs = z.reshape(-1, 2)
        zx, zy = z_pairs[:, 0], z_pairs[:, 1]

        radii = np.sqrt(zx ** 2 + zy ** 2)        # Magnitude per pair
        angles = np.arctan2(zy, zx)               # Strictly bounded in [-pi, +pi]!

        # Quantize angles uniformly across [-pi, +pi] with ZERO min/max scaling metadata
        step_size = (2.0 * np.pi) / self.angle_levels
        angle_indices = np.clip(
            np.floor((angles + np.pi) / step_size), 0, self.angle_levels - 1
        )
        quantized_angles = -np.pi + (angle_indices + 0.5) * step_size

        # Quantize radii using global L2 norm (1 single float scalar for the whole vector)
        rmax = np.max(radii) + 1e-9
        quantized_radii = np.round((radii / rmax) * 7.0) / 7.0 * rmax  # 3-bit radius level

        # Reconstruct Stage 1 Polar approximation
        z_polar = np.empty_like(z_pairs)
        z_polar[:, 0] = quantized_radii * np.cos(quantized_angles)
        z_polar[:, 1] = quantized_radii * np.sin(quantized_angles)
        z_polar = z_polar.reshape(-1)

        # --- STEP 3: 1-Bit QJL Residual Error Checker ---
        residual_error = z - z_polar
        residual_norm = np.linalg.norm(residual_error)

        # Project residual error and keep ONLY 1 sign bit (+1 or -1) per dimension
        qjl_sign_bits = np.where((self.S @ residual_error) >= 0.0, 1.0, -1.0)

        # Unbiased QJL estimator scaling factor: sqrt(pi / 2) / dim * ||e||_2
        qjl_scale = (np.sqrt(np.pi / 2.0) / self.dim) * residual_norm
        e_reconstructed = qjl_scale * (self.S.T @ qjl_sign_bits)

        # Combine PolarQuant + 1-Bit QJL correction and rotate back (R.T)
        z_final = z_polar + e_reconstructed
        return self.R.T @ z_final


# --- Verification Demo: Spiky Outlier Vector Test ---
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    dim = 64
    # Create a spiky vector: 62 tiny dimensions (~0.05) and 2 massive outlier spikes (+18.5, -14.2)
    spiky_doc = rng.normal(0.0, 0.05, size=dim).astype(np.float32)
    spiky_doc[0] = 18.5
    spiky_doc[17] = -14.2

    query = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    true_dot_product = float(query @ spiky_doc)

    # Naive 3-Bit Uniform Quantization (8 buckets stretched across [-18.5, +18.5])
    max_val = np.max(np.abs(spiky_doc))
    naive_3bit = np.round((spiky_doc / max_val) * 3.0) / 3.0 * max_val
    naive_dot = float(query @ naive_3bit)

    # Google TurboQuant (Rotation + PolarQuant + 1-Bit QJL)
    tq = TurboQuantizer(dim=dim)
    tq_reconstructed = tq.compress_and_reconstruct(spiky_doc)
    tq_dot = float(query @ tq_reconstructed)

    print(f"True Exact Dot Product:       {true_dot_product:.4f}")
    print(f"Naive 3-Bit Dot Product:      {naive_dot:.4f} (Error: {abs(true_dot_product - naive_dot):.4f})")
    print(f"TurboQuant Dot Product:       {tq_dot:.4f} (Error: {abs(true_dot_product - tq_dot):.4f})")
```

---

## Block 4: Production Qdrant Hybrid Search (`Prefetch` + Sparse BM25 + Server-Side `RRF`) (Minutes 42–50)

In a production service, you don't pull 100 BM25 results and 100 Dense results over the network to fuse them in Python—that wastes network bandwidth. Instead, Qdrant executes **both searches and the RRF fusion inside the database engine in a single API call**.

### 4.1 How Qdrant Stores Dense and Sparse (BM25) Together
Each point in Qdrant can hold **multiple named vectors**:
1. **`"dense"` (`VectorParams`):** A fixed-length list of floats (e.g., 768 floats from your bi-encoder).
2. **`"sparse"` (`SparseVectorParams` with `Modifier.IDF`):** Instead of storing 50,000 zeros for words that *don't* appear in a chunk, a **Sparse Vector** only stores two tiny lists for the words that *do* appear:
   * `indices`: The integer vocabulary IDs of the words present in the chunk (e.g., `[104, 8821]`).
   * `values`: The term frequency weights of those words (e.g., `[1.0, 2.2]`).
   * Setting `modifier=models.Modifier.IDF` tells Qdrant's engine to automatically multiply those weights by Inverse Document Frequency (IDF) at query time!

```text
Single Qdrant Point (ID: "doc_1"):
 ├── payload: {"text": "BMS 0x84f2 limit...", "allowed_groups": ["grp_eng"]}
 └── vector:
      ├── "dense":  [0.12, -0.44, 0.81, 0.05]                 <-- Semantic Bi-Encoder
      └── "sparse": SparseVector(indices=[42, 991], values=[1.0, 1.8]) <-- Lexical BM25
```

---

### 4.2 Complete, Runnable Code: Single-Call Qdrant Hybrid Search with RBAC Propagation

```python
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models


class ProductionHybridQdrantStore:
    """
    Demonstrates Qdrant's Universal Query API combining:
    1. Named Dense + Sparse (BM25 with Server-Side IDF Modifier) vectors
    2. Parallel sub-query execution via models.Prefetch
    3. Server-side Reciprocal Rank Fusion (FusionQuery / RRF)
    4. Automatic global RBAC filter propagation into all Prefetch stages
    """

    def __init__(self, collection_name: str = "hybrid_enterprise_rag", dense_dim: int = 4):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self._init_hybrid_collection(dense_dim)

    def _init_hybrid_collection(self, dense_dim: int) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            # 1. Configure Named Dense Vector
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            # 2. Configure Named Sparse Vector with automatic Server-Side IDF weighting
            sparse_vectors_config={
                "sparse_bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # Pre-create RBAC payload index for single-stage filtered graph/inverted-index traversal
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="allowed_groups",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )

    def upsert_hybrid_points(self, items: List[Dict[str, Any]]) -> None:
        points = [
            models.PointStruct(
                id=item["id"],
                vector={
                    "dense": item["dense_vec"],
                    "sparse_bm25": models.SparseVector(
                        indices=item["sparse_indices"],
                        values=item["sparse_values"],
                    ),
                },
                payload={
                    "text": item["text"],
                    "allowed_groups": item["allowed_groups"],
                },
            )
            for item in items
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def hybrid_rrf_search(
        self,
        dense_query: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        user_groups: List[str],
        prefetch_limit: int = 50,
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel Dense + Sparse prefetches and fuses them via RRF on the server.
        Note: Passing query_filter at the top level automatically propagates the RBAC filter
        into BOTH the dense prefetch and the sparse_bm25 prefetch!
        """
        global_rbac_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="allowed_groups",
                    match=models.MatchAny(any=user_groups),
                )
            ]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Sub-query 1: Dense Semantic Search
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=prefetch_limit,
                ),
                # Sub-query 2: Sparse BM25 Lexical Search
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse_bm25",
                    limit=prefetch_limit,
                ),
            ],
            # Server-side Reciprocal Rank Fusion over the union of both prefetches
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=global_rbac_filter,  # Propagates automatically to all prefetches!
            limit=final_top_k,
            with_payload=True,
        )

        return [
            {
                "id": point.id,
                "rrf_score": round(point.score, 4),
                "text": point.payload.get("text"),
            }
            for point in response.points
        ]


# --- Verification Demo ---
if __name__ == "__main__":
    store = ProductionHybridQdrantStore(dense_dim=4)
    # Vocabulary IDs: 101="bms", 205="0x84f2", 309="thermal"
    store.upsert_hybrid_points([
        {
            "id": 1,
            "dense_vec": [0.95, 0.10, 0.0, 0.0],
            "sparse_indices": [101, 309],
            "sparse_values": [1.0, 1.0],
            "text": "General BMS thermal management overview.",
            "allowed_groups": ["grp_eng"],
        },
        {
            "id": 2,
            "dense_vec": [0.88, 0.22, 0.0, 0.0],
            "sparse_indices": [101, 205, 309],
            "sparse_values": [1.0, 2.1, 1.0],
            "text": "BMS error 0x84f2 triggers thermal current limit of 45A.",
            "allowed_groups": ["grp_eng"],
        },
    ])

    results = store.hybrid_rrf_search(
        dense_query=[0.92, 0.15, 0.0, 0.0],
        sparse_indices=[205, 309],  # User searches "0x84f2 thermal"
        sparse_values=[1.0, 1.0],
        user_groups=["grp_eng"],
        prefetch_limit=10,
        final_top_k=2,
    )
    print("Qdrant Server-Side Hybrid RRF Results:", results)
```
