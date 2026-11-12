# Day 3 Quick Revision Cheat Sheet: HNSW vs. IVF-PQ, Quantization & Hybrid RRF

*Scan this 5-minute sheet before System Design and Retrieval Architecture rounds.*

---

## 1. Index & Quantization Memory Formulas (1024-dim or 768-dim)

| Scheme | Bytes per 768d Vector | Bytes per 1024d Vector | Compression Ratio | Recall@10 Impact | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (Baseline)** | $3,072\text{ B}$ | $4,096\text{ B}$ | $1\times$ | $100\%$ (Exact baseline) | Small indices ($<2\text{M}$ vectors) |
| **FP16 / BF16** | $1,536\text{ B}$ | $2,048\text{ B}$ | $2\times$ | $\sim 99.9\%$ (Zero loss) | Default on-disk `mmap` rescoring storage |
| **Scalar Quantization (SQ8 / INT8)** | $768\text{ B}$ | $1,024\text{ B}$ | **$4\times$** | **$>99\%$** (with $1.5\times\text{–}2\times$ oversampling) | **Default production choice** for any dimension |
| **Product Quantization (PQ, $m=64\text{ or }96$)** | $96\text{ B}$ ($m=96$) | $64\text{ B}$ ($m=64$) | **$32\times\text{–}64\times$** | $75\%\text{–}88\%$ (Suffers from codebook drift) | Extreme RAM limits when $d < 1024$ |
| **Binary Quantization (BQ, 1-bit)** | $96\text{ B}$ ($768\text{ bits}$) | $128\text{ B}$ ($1024\text{ bits}$) | **$32\times$** | **$>95\%$** (Requires $d \ge 1024$ + $3\times\text{–}4\times$ rescoring) | **1024d+ models** (`bge-large`, `jina-v3`, Cohere, OpenAI) |

* **HNSW Graph Link Overhead Formula:**
  $$\text{Graph RAM} \approx N \times 2M \times 4\text{ Bytes} \times 1.33 \quad (\text{For } M=16: \approx \mathbf{170\text{ Bytes per vector}})$$

---

## 2. Core Equations to Write on the Whiteboard

### A. BM25 Scoring Formula
$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* **$k_1 \in [1.2, 1.5]$:** Controls **Term Frequency Saturation** (as $f \to \infty$, term weight plateaus at $k_1 + 1$, preventing keyword-stuffed logs from dominating).
* **$b \in [0, 1]$ (Default $0.75$, RAG-tuned $0.3\text{–}0.5$):** Controls **Document Length Penalty** (lowered in RAG because chunks already have uniform token lengths).

### B. Reciprocal Rank Fusion (RRF) Formula
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense, BM25}\}} \frac{w_m}{k + \text{rank}_m(d)} \quad (\text{where } k = 60)$$
* **Why $k = 60$?** Smooths the steep drop between Rank #1 and Rank #2 so a document ranked **#2 in both Dense and BM25** ($\frac{1}{62} + \frac{1}{62} = 0.03226$) beats a noisy outlier ranked **#1 in only one retriever** ($\frac{1}{61} + 0 = 0.01639$).

---

## 3. Rapid-Fire Interview Trap Questions & Answers

**Q1: "Why does IVF-PQ quantize the residual vector ($r = x - c_i$) instead of the raw vector $x$?"**  
> **Answer:** "Because every vector inside Voronoi cell $i$ lies in the immediate neighborhood of centroid $c_i$, subtracting $c_i$ centers the vectors around the origin and drastically shrinks their variance. Quantizing a narrow-variance residual distribution with 256 codebook centroids produces much lower MSE reconstruction error than trying to cover the entire global vector space with the same 256 centroids."

**Q2: "When would you choose HNSW over IVF-PQ, and when does IVF-PQ break in production?"**  
> **Answer:** "HNSW is non-parametric and supports real-time streaming inserts at 95–99% recall, whereas IVF-PQ requires offline k-means training for both Voronoi cells and PQ codebooks. In production, IVF-PQ breaks under **Concept Drift (Centroid Staleness)**—when we ingest a new product domain that wasn't in the initial training sample, new vectors pile into unbalanced Voronoi cells and suffer severe quantization distortion until we run an expensive full re-index."

**Q3: "Why can't we just use $\alpha \cdot \text{Cosine} + (1-\alpha) \cdot \text{BM25}$ instead of RRF?"**  
> **Answer:** "Cosine similarity is bounded in $[-1, 1]$ and tightly clustered (e.g., $0.78\text{–}0.86$), whereas BM25 is unbounded $[0, \infty)$ and shifts wildly depending on query length and term rarity (IDF). Even if you apply per-query Min-Max normalization to BM25, a single keyword-stuffed chunk with an outlier BM25 score of $60.0$ will crush the denominator and compress the normalized score of the true #2 hit down to near zero. RRF is non-parametric and invariant to score scale."

**Q4: "How do you get 98% recall out of 1-bit Binary Quantization?"**  
> **Answer:** "Three steps: First, ensure the embedding model is high-dimensional ($d \ge 1024$) and mean-centered around zero. Second, run the HNSW graph search in RAM using `XOR + POPCOUNT` Hamming distance with an **oversampling factor of $3\times\text{ to }4\times$** (e.g., fetching 120 candidates for a Top-30 target). Third, read only those 120 full-precision FP16 vectors from NVMe SSD via `mmap` and **rescore** them with exact cosine similarity."
