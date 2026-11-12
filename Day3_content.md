# Day 3 Complete Self-Contained Study Guide: HNSW vs. IVF-PQ, Vector Quantization Math & Hybrid Search (BM25 + RRF)

**Target Role Level:** Senior / Lead AI Engineer (₹55L–₹95L+ TC at Atlassian, Intuit, Microsoft, Adobe, SAP Labs, Target)  
**Study Time:** 50 Minutes (100% self-contained; links & videos are optional)  
**Your Unique Narrative Hook:** Combining your Volkswagen large-scale enterprise retrieval background with your Samsung representation learning and quantization depth to optimize RAM vs. Recall vs. Latency at 100M+ vector scale.

---

## Block 1: Vector Index Internals & Memory Math — HNSW vs. IVF-PQ (Minutes 0–20)

At 1 million vectors, any default vector index works. In senior system design interviews, the interviewer will scale the prompt to **50 million to 100 million vectors** and ask: *"How much RAM does this take, why is your cloud bill $8,000/month, and how do you cut memory by 80% without destroying p99 latency or recall?"*

### 1.1 HNSW (Hierarchical Navigable Small World) Memory & Scaling Profile
* **How it works:** Maintains a multi-layered proximity graph where the base layer ($L_0$) holds all $N$ vectors, and each node stores up to $2M$ bidirectional graph edges (typically $M=16 \rightarrow 32$ neighbor pointers of 4 or 8 bytes each).
* **Exact HNSW RAM Formula:**
  $$\text{RAM}_{\text{HNSW}} = N \times \left( \underbrace{d \times \text{Bytes}_{\text{float}}}_{\text{Raw Vector}} + \underbrace{2 \times M \times 4\text{ B} \times 1.33}_{\text{Graph Edges Across Layers}} \right)$$
* **Concrete 50-Million Vector Example ($d = 768$, FP32 = 4 bytes, $M = 16$):**
  1. **Raw Vectors:** $50,000,000 \times 768 \times 4\text{ B} = \mathbf{153.6\text{ GB}}$
  2. **Graph Edges:** $50,000,000 \times (32 \times 4\text{ B} \times 1.33) \approx 50\text{M} \times 170\text{ B} = \mathbf{8.5\text{ GB}}$
  3. **Total RAM (with 15% allocator/payload overhead):** **$\sim 186\text{ GB RAM}$**
* **Strength:** Sub-10ms latency at 95–99% Recall@10; supports **streaming incremental inserts** without global retraining.
* **Weakness:** Massive RAM footprint; graph links alone consume 8.5 GB, and unquantized vectors must reside in fast memory during graph hops.

---

### 1.2 IVF-PQ (Inverted File Index with Product Quantization) Step-by-Step
When RAM is strictly capped and you cannot afford HNSW graph links, **IVF-PQ** (popularized in FAISS and Milvus) compresses both the search space and the vector bytes via two stages:

#### Stage A: IVF (Inverted File Clustering via Voronoi Cells)
1. **Training:** Run k-means on a sample of vectors to partition the vector space into $nlist$ Voronoi centroids (typically $nlist = \sqrt{N}$ to $4\sqrt{N}$; for $N = 50\text{M}$, $nlist \approx 8,192$ to $16,384$ centroids).
2. **Assigning & Residual Calculation:** Each vector $x$ is assigned to its closest centroid $c_i$. Instead of storing $x$, we compute the **residual vector**:
   $$r = x - c_i$$
   * **Why Residual Encoding Matters (Senior Interview Signal):** Because all vectors inside cell $i$ are close to $c_i$, the residual $r$ has a much smaller variance (dynamic range) than raw vector $x$. Quantizing the residual $r$ yields significantly lower reconstruction error than quantizing $x$ directly!
3. **Query Time (`nprobe`):** Given query $q$, find the top `nprobe` (e.g., 32 out of 16,384) closest centroids, and **only scan the inverted lists of those 32 cells** (scanning just $\frac{32}{16384} \approx 0.2\%$ of the dataset).

#### Stage B: PQ (Product Quantization) — Compressing 3,072 Bytes into 96 Bytes
Instead of storing the 768-dim FP32 residual vector $r$ ($3,072\text{ bytes}$), PQ slices and quantizes it:

```text
Original 768-dim FP32 Vector (3,072 Bytes):
[ d_0 ... d_7 | d_8 ... d_15 | ... | d_760 ... d_767 ]
       │              │                     │
  Split into m = 96 Sub-Vectors (each of dimension d* = 768 / 96 = 8 floats)
       ▼              ▼                     ▼
  [Sub-Vec 1]    [Sub-Vec 2]    ...    [Sub-Vec 96]
       │              │                     │
  Map each 8-dim sub-vector to nearest of k* = 256 Codebook Centroids (0..255)
       ▼              ▼                     ▼
  [ 142 (1B) ]   [  19 (1B) ]   ...    [ 251 (1B) ]  --> Total = 96 Bytes (uint8 array)!
```

1. **Slice into $m$ Sub-Spaces:** Split the $d=768$ vector into $m=96$ contiguous sub-vectors, each of dimension $d^* = \frac{768}{96} = 8$.
2. **Train Sub-Space Codebooks:** Run k-means independently inside each of the 96 sub-spaces with $k^* = 256$ centroids. Because $256 = 2^8$, each centroid ID fits inside a **single `uint8` byte (8 bits)**.
3. **Encode:** Replace each 8-float sub-vector ($32\text{ bytes}$) with its 1-byte centroid ID.
   * **Compression Math:** $3,072\text{ bytes} \longrightarrow \mathbf{96\text{ bytes}}$ (**$32\times$ compression!**). For 50 million vectors, the entire index shrinks from **186 GB (HNSW) down to ~5.2 GB (IVF-PQ)**!

#### Stage C: Asymmetric Distance Computation (ADC) via Lookup Tables (LUT)
How does IVF-PQ compute distances at query time without decompressing millions of vectors back into FP32?
1. When query $q$ arrives, slice $q$ into $m = 96$ sub-vectors $[q^{(1)}, q^{(2)}, \dots, q^{(96)}]$.
2. Precompute a **Distance Lookup Table (LUT)** of size $m \times k^* = 96 \times 256$ floats by calculating the exact squared distance between each query sub-vector $q^{(j)}$ and all 256 codebook centroids in sub-space $j$. (Only $96 \times 256 = 24,576$ distance calculations total!).
3. Now, to score any compressed database vector $x = [b_1, b_2, \dots, b_{96}]$, simply perform **96 array lookups and additions**:
   $$\hat{d}(q, x) = \sum_{j=1}^{96} \text{LUT}[j][b_j]$$
   Zero floating-point multiplications are needed during the candidate scan!

---

### 1.3 Production Failure Modes: When IVF-PQ Breaks vs. HNSW

| Dimension | HNSW (Graph-Based) | IVF-PQ (Cluster + Codebook-Based) |
| :--- | :--- | :--- |
| **50M Vector RAM (768d)** | ~186 GB (FP32) / ~50 GB (INT8) | **~5.2 GB** ($32\times$ smaller) |
| **Recall@10 (Without Rerank)** | **95% – 99%** | **70% – 85%** (Lossy quantization error) |
| **Query Latency (p99)** | **2 – 8 ms** (Logarithmic graph hops) | **15 – 40 ms** (Linear scan over `nprobe` lists) |
| **Streaming Inserts & Concept Drift** | **Excellent:** New vectors wire into graph on the fly with zero retraining. | **Brittle:** Requires initial k-means training; suffers from **Centroid Staleness** under domain shift. |

#### The Senior Interview Failure Story: "Concept Drift & Centroid Staleness in IVF-PQ"
Suppose you trained your IVF-PQ centroids and codebooks in January on Volkswagen combustion-engine manuals. In June, you ingest 5 million new documents on **Solid-State EV Batteries**.
* **What goes wrong in IVF-PQ:** All 5 million new EV battery chunks fall into just 2 or 3 existing Voronoi cells (creating ** unbalanced "jumbo lists"** that slow down `nprobe` scans by $10\times$), and the PQ sub-space codebooks—which never saw battery terminology embeddings during training—suffer massive quantization distortion. Recall@10 drops from 85% to 50% until you run an expensive offline **Full Index Re-Clustering** job.
* **Why HNSW avoids this:** HNSW is non-parametric—it requires zero k-means training and dynamically links new EV battery vectors to their nearest neighbors as they arrive.

#### Optional Reference Links & Videos (Block 1)
* **Articles:** [FAISS Index Selection Guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) | [Product Quantization for Nearest Neighbor Search (Pinecone)](https://www.pinecone.io/learn/series/faiss/product-quantization/)
* **Videos:** [How Vector Databases Actually Work: ANN, HNSW, IVF & Product Quantization](http://www.youtube.com/watch?v=YitjkFRN20Q) | [Vector Search & ANN: FAISS HNSW & IVF](http://www.youtube.com/watch?v=chz74Mtd1AA) | [Product Quantization Explained: HNSW vs IVF vs PQ](http://www.youtube.com/watch?v=4FZvqtnQ5Hw)

---

## Block 2: Vector Quantization Deep-Dive — SQ8 vs. PQ vs. Binary Quantization (BQ) (Minutes 20–35)

In modern vector databases (Qdrant, Weaviate, Milvus), the dominant production architecture is **HNSW + In-RAM Quantization + On-Disk Rescoring**—giving you HNSW's dynamic updates and graph speed with IVF-PQ's low memory cost.

### 2.1 Comparing the Three Vector Quantization Schemes

#### 1. Scalar Quantization (SQ8 / INT8 — $4\times$ Compression)
* **Mechanism:** Maps each FP32 dimension ($4\text{ bytes}$) independently to a signed 8-bit integer (`int8`, $1\text{ byte}$) using the 99th-percentile (`quantile = 0.99`) min/max range to clip activation outliers:
  $$x_{\text{int8}} = \text{round}\left( 127 \times \frac{\text{clip}(x, q_{0.005}, q_{0.995})}{\max(\vert{}q_{0.005}\vert{}, \vert{}q_{0.995}\vert{})} \right)$$
* **Trade-off:** Cuts vector RAM by **$4\times$** ($153.6\text{ GB} \rightarrow 38.4\text{ GB}$ for 50M vectors) with **$<1\%$ Recall@10 loss**. Always your safest default in production.

#### 2. Binary Quantization (BQ / 1-Bit — $32\times$ Compression + $40\times$ Faster Distance Math)
* **Mechanism:** Collapses every floating-point dimension into a **single bit** based on its sign:
  $$b_i = \begin{cases} 1 & \text{if } x_i > 0 \\ 0 & \text{if } x_i \le 0 \end{cases}$$
* **Memory & CPU Math:**
  * A 768-dim FP32 vector ($3,072\text{ bytes}$) packs into twelve 64-bit integers (`uint64`) = **$96\text{ bytes}$** ($32\times$ compression).
  * Instead of 768 floating-point multiply-adds, distance between two binary vectors $A$ and $B$ is the **Hamming Distance** (number of differing bits), computed in hardware via bitwise `XOR` followed by `POPCOUNT` (population count instruction):
    $$d_{\text{Hamming}}(A, B) = \text{POPCOUNT}(A \oplus B)$$
  * A single CPU instruction (`_mm512_popcnt_epi64`) processes 512 bits in **1 clock cycle**, making BQ distance calculations up to **$40\times$ faster** than FP32 cosine similarity!

#### The Senior Interview Trap: When Does Binary Quantization Fail?
If an interviewer asks: *"Can we turn on Binary Quantization for any embedding model to get a 32x RAM reduction?"*
* **Your Winning Answer:** **No.** Binary Quantization discards all magnitude information within each dimension and only preserves the quadrant sign $(+/-)$. It only maintains high recall under two strict conditions:
  1. **High Dimensionality ($d \ge 1024$):** Models like `bge-large` (1024d), `jina-embeddings-v3` (1024d), Cohere v3, or OpenAI `text-embedding-3-large` (1536d/3072d) have enough angular redundancy across dimensions that sign bits approximate cosine angle closely. On a 384-dim model (`all-MiniLM-L6-v2`), BQ loses 15–25% recall!
  2. **Zero-Centered Isotropic Distribution:** If an encoder has an anisotropic mean shift (e.g., dimension 42 is $>0$ for 99% of inputs), that bit is always `1` and carries zero information. In production, you must **subtract the corpus mean vector** ($\tilde{x} = x - \mu_{\text{corpus}}$) before binarizing, or train the encoder explicitly with a **Binary Contrastive / Matryoshka loss** (something you can directly tie to your Samsung encoder training background).

---

### 2.2 Two-Stage Quantized Retrieval: Oversampling + `mmap` Rescoring
To recover the 5–10% recall lost during Binary or Product Quantization without keeping FP32 vectors in RAM:

```text
[Query Vector q (FP32)]
          │
          ▼
Stage 1: Quantize q -> Search In-RAM Binary/INT8 HNSW Graph
         Fetch Oversampled Candidates: Top (α × K) = (3.0 × 10) = 30 IDs  [~2 ms, 0 Disk I/O]
          │
          ▼
Stage 2: Read ONLY those 30 Full-Precision (FP16/FP32) Vectors from NVMe SSD via mmap
         Recompute Exact Cosine Similarity (q, x_exact) -> Sort & Return Top K = 10  [~2 ms]
```

* **Concrete Math:** Suppose you want $K = 10$ results and set `oversampling = 3.0` (or `4.0` for Binary Quantization).
  * Stage 1 traverses the HNSW graph entirely in RAM using 96-byte binary vectors and retrieves the top $3 \times 10 = 30$ candidate IDs in **1.5 ms**.
  * Stage 2 reads only $30 \times 3\text{ KB} = 90\text{ KB}$ from NVMe SSD (`mmap`) to rescore those 30 candidates in exact FP32, restoring Recall@10 from **82% back to 98.5%** while keeping 97% of your vector data off RAM!

---

## Block 3: Hybrid Search — BM25 Internals, Score Normalization & Reciprocal Rank Fusion (RRF) (Minutes 35–50)

### 3.1 Why Dense Embeddings Fail Alone on Enterprise Data
Dense bi-encoders compress an entire 256-token paragraph into a single fixed vector via mean pooling. While great for semantic synonyms (*"automobile battery overheating"* $\leftrightarrow$ *"EV thermal runaway"*), dense embeddings fail on:
1. **Exact Part Numbers, SKUs & Hex Codes:** `VW-BMS-0x84F2` vs `VW-BMS-0x84F3` get tokenized into noisy subwords and map to nearly identical dense vectors.
2. **Acronyms & Rare Domain Entities:** An internal project codename like `Project-Krypton-v4` was never seen during pre-training.
3. **Short Keyword Queries:** When an engineer searches a single error token `ERR_CAN_TIMEOUT_504`, lexical exact matching is mandatory.

---

### 3.2 BM25 (Best Matching 25) Mathematical Anatomy
Interviewers frequently ask: *"How does BM25 improve upon classic TF-IDF, and what do parameters $k_1$ and $b$ control?"*

For a query $Q$ containing terms $q_1, \dots, q_n$, the BM25 score of document (chunk) $D$ is:
$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}D\vert{}}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$ = Term frequency (count of term $q_i$ inside chunk $D$).
* $\vert{}D\vert{} / \text{avgdl}$ = Ratio of chunk $D$'s token length to the average chunk length across the corpus.
* **$\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$:** Inverse Document Frequency. Common words like `"the"` appear in $n(q_i) \approx N$ chunks and get $\text{IDF} \approx 0$; rare serial numbers like `0x84F2` appear in 3 chunks out of 1,000,000 and get a massive $\text{IDF} \approx 12.7$.
* **Parameter 1 — $k_1 \in [1.2, 1.5]$ (Term Frequency Saturation):**
  * **The TF-IDF Flaw:** In classic TF-IDF, a log file repeating `"ERROR_404"` 100 times scores $100\times$ higher than a clean troubleshooting guide mentioning `"ERROR_404"` once.
  * **How $k_1$ fixes it:** In BM25, as $f(q_i, D) \to \infty$, the term frequency fraction asymptotically approaches an upper bound of **$k_1 + 1$** (e.g., $1.2 + 1 = 2.2$). Mentioning a keyword 2 times gives a big boost; mentioning it 50 times yields diminishing returns!
* **Parameter 2 — $b \in [0, 1]$ (Document Length Normalization, default $0.75$):**
  * Penalizes long rambling chunks that happen to match a word just because they have 1,000 tokens.
  * **RAG Tuning Tip:** Because RAG chunks are already split to uniform sizes (e.g., 256–400 tokens), $\vert{}D\vert{} \approx \text{avgdl}$. Setting a lower $b \approx 0.3\text{–}0.5$ prevents over-penalizing slightly longer table chunks.

---

### 3.3 Why Linear Score Combination Breaks — And How Reciprocal Rank Fusion (RRF) Solves It

Suppose you retrieve top candidates from both Dense Vector Search and BM25. Why is the naive convex combination $\text{Score} = \alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{BM25}}$ broken in production?

#### Concrete Failure Example of Raw / Min-Max Score Fusion:
* **Distribution Mismatch:** Cosine similarity $S_{\text{dense}}$ is strictly bounded in $[-1, 1]$, and for an uncalibrated bi-encoder, top-50 hits tightly cluster between **$0.78$ and $0.86$**. Meanwhile, $S_{\text{BM25}}$ is unbounded $[0, \infty)$—on a 2-word query it might range from $2.1$ to $5.4$, while on a rare part-number query it spikes to **$42.0$**.
* **Why Min-Max Normalization per Query Fails:** Suppose Chunk A is a spammy log file that matches 4 query tokens and gets a freak BM25 score of $50.0$, while Chunk B (the actual answer) gets a solid BM25 score of $12.0$ and Ranks #2. If you min-max normalize BM25 scores into $[0, 1]$ per query, the outlier score of $50.0$ crushes Chunk B's normalized BM25 score down to $\frac{12}{50} = 0.24$!

#### The Production Fix: Reciprocal Rank Fusion (RRF)
RRF completely ignores raw score magnitudes and fuses results purely based on each item's **ordinal rank position** ($1, 2, 3, \dots$) in each retriever:

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$

* $\text{rank}_m(d) \in \{1, 2, 3, \dots\}$ is the 1-based position of chunk $d$ in retriever $m$'s sorted list (if chunk $d$ is not in retriever $m$'s top-$K$, its term for $m$ is $0$).
* $w_m$ is an optional retriever weight (default $w_{\text{dense}} = 1.0, w_{\text{BM25}} = 1.0$).
* **$k = 60$ (The Smoothing Constant):** Why do we add $k = 60$ to the denominator instead of computing $\frac{1}{\text{rank}}$?
  * **Without $k$ ($k=0$):** Rank #1 scores $\frac{1}{1} = 1.0$, while Rank #2 scores $\frac{1}{2} = 0.50$. A document ranked **#1 in BM25 alone** (score $1.0$) would beat a document ranked **#2 in BOTH Dense and BM25** ($\frac{1}{2} + \frac{1}{2} = 1.0$)! A single noisy Rank #1 hit from keyword matching would hijack the top spot.
  * **With $k=60$:** The gap between Rank #1 ($\frac{1}{61} = 0.01639$) and Rank #2 ($\frac{1}{62} = 0.01613$) is smoothed out. Consensus across *both* retrievers is mathematically guaranteed to beat a lone outlier in a single retriever!

#### Worked Numeric RRF Table ($k = 60$, Equal Weights $w = 1.0$)

| Chunk ID | Dense Rank ($\text{rank}_{\text{dense}}$) | BM25 Rank ($\text{rank}_{\text{BM25}}$) | Dense RRF Term ($\frac{1}{60 + r_d}$) | BM25 RRF Term ($\frac{1}{60 + r_b}$) | Total RRF Score | Final RRF Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DOC_A`** (Consensus Hit) | Rank **#2** | Rank **#2** | $\frac{1}{62} = 0.01613$ | $\frac{1}{62} = 0.01613$ | **$0.03226$** | **#1 (Winner!)** |
| **`DOC_B`** (Consensus Hit) | Rank **#4** | Rank **#3** | $\frac{1}{64} = 0.01563$ | $\frac{1}{63} = 0.01587$ | **$0.03150$** | **#2** |
| **`DOC_C`** (BM25 #1 Outlier) | Not in Top-100 ($\infty$) | Rank **#1** | $0.00000$ | $\frac{1}{61} = 0.01639$ | **$0.01639$** | **#3** |
| **`DOC_D`** (Dense #1 Only) | Rank **#1** | Not in Top-100 ($\infty$) | $\frac{1}{61} = 0.01639$ | $0.00000$ | **$0.01639$** | **#4** |

> **Look at `DOC_A` vs `DOC_C`:** Even though `DOC_C` was #1 in BM25, `DOC_A` (which ranked #2 in both semantic dense search and BM25 lexical search) wins easily ($0.03226$ vs $0.01639$).

#### Optional Reference Links & Videos (Block 3)
* **Articles:** [Elastic RRF & Hybrid Search Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) | [Qdrant Hybrid Search & Sparse Vectors](https://qdrant.tech/articles/hybrid-search/)
* **Videos:** [The Complete Guide to Hybrid Search in RAG (BM25 + Embeddings + Reranker)](http://www.youtube.com/watch?v=XvKiTfd6Xvo) | [Understanding Reciprocal Rank Fusion in Hybrid Search](http://www.youtube.com/watch?v=6dDvfGrxFns) | [What is Reciprocal Rank Fusion in Hybrid Search?](http://www.youtube.com/watch?v=2uBcjEecr38)

---

## Day 3 Complete Whiteboard Solution (Memorize This Blueprint)

> **Interview Prompt:** *"Design a hybrid search index for 100 million technical documents (1024-dim embeddings) where users search both natural-language symptoms and exact hex error codes. Keep RAM under 40 GB and p99 retrieval latency under 25 ms."*

1. **Back-of-the-Envelope Memory Sizing (Why Uncompressed HNSW Fails):**
   * Uncompressed FP32 vectors: $100\text{M} \times 1024 \times 4\text{ B} = \mathbf{409.6\text{ GB}}$ (10x over our 40 GB RAM budget!).
   * **Solution — Binary Quantization (BQ) in RAM + FP16 on NVMe SSD (`mmap`):**
     * Because our embeddings are 1024-dimensional, we mean-center and apply **1-bit Binary Quantization** ($1024\text{ bits} = \mathbf{128\text{ bytes/vector}}$).
     * **In-RAM Binary Vectors:** $100\text{M} \times 128\text{ B} = \mathbf{12.8\text{ GB}}$.
     * **In-RAM HNSW Graph Links ($M=16$, 4-byte internal IDs):** $100\text{M} \times 32 \times 4\text{ B} \times 1.33 \approx \mathbf{17.0\text{ GB}}$.
     * **Sparse BM25 Inverted Index (on disk with hot posting lists cached):** $\sim\mathbf{6\text{ GB RAM}}$.
     * **Total RAM:** $12.8 + 17.0 + 6.0 = \mathbf{35.8\text{ GB RAM}}$ (Fits comfortably under our 40 GB ceiling!).
     * Full FP16 vectors ($204.8\text{ GB}$) live on low-cost NVMe SSD via memory-mapped files (`mmap`).
2. **Dual-Path Parallel Query Execution (<20 ms):**
   * **Path A (Dense Semantic with Oversampled Rescoring):**
     * Query $q$ is binarized and traverses the in-RAM Binary HNSW graph using `XOR + POPCOUNT` to fetch an oversampled pool of $4 \times 50 = \mathbf{200\text{ candidates}}$ in ~4 ms.
     * Engine fetches those 200 FP16 vectors from NVMe `mmap` ($200 \times 2\text{ KB} = 400\text{ KB}$ sequential/cached read, ~3 ms) and rescores them via exact cosine similarity to emit the **Top 50 Dense Hits**.
   * **Path B (Sparse Lexical BM25):**
     * In parallel, execute BM25 with $k_1 = 1.2, b = 0.4$ (tuned for uniform chunk lengths) over the sparse inverted index to capture exact hex codes (`0x84F2`) and emit the **Top 50 Lexical Hits** in ~5 ms.
3. **Reciprocal Rank Fusion (RRF) & Cross-Encoder Handoff:**
   * Fuse the Top 50 Dense and Top 50 BM25 lists in <1 ms using **RRF ($k=60$)** to eliminate score calibration drift between unbounded BM25 scores and bounded cosine similarities.
   * Pass the top 20 RRF winners to our distilled Cross-Encoder Reranker (Days 5–6) for final precision ordering.
