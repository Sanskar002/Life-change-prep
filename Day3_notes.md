# Day 3 Quick Reference Notes: HNSW, IVF-PQ, Quantization & Hybrid RRF

*Concise reference sheet covering core mental models, memory formulas, and key architecture questions.*

---

## 1. Core Mental Models

* **HNSW (Hierarchical Navigable Small World):** *Express Flights $\rightarrow$ Highways $\rightarrow$ Local Streets.* Traverses sparse upper layers to drop into the target neighborhood on Layer 0 (where all vectors reside). Supports live streaming inserts, adding ~170 Bytes/vector in graph link RAM ($M=16$).
* **IVF (Inverted File Index):** *Postal Pin-Code Zones.* Partitions vectors into `nlist` k-means clusters. At query time, scans only the closest `nprobe` clusters.
* **Product Quantization (PQ):** *Paint-Shop Swatch Book.* Splits a 768-dim vector into 96 sub-vectors and replaces each with a 1-byte Pattern ID (`0–255`) from a trained codebook ($32\times$ smaller).
* **Binary Quantization (BQ):** *Sign Bit Compression.* Maps values $>0$ to `1` and $\le 0$ to `0` ($32\times$ smaller). Computes Hamming distance via CPU `XOR + POPCOUNT` ($40\times$ faster), followed by SSD `mmap` rescoring.
* **Reciprocal Rank Fusion (RRF):** *Rank-Based Consensus.* Bypasses raw score scale differences between Cosine (`0.8`) and BM25 (`35.0`) by summing reciprocal ranks: $\frac{1}{60 + \text{Rank}_{\text{Dense}}} + \frac{1}{60 + \text{Rank}_{\text{BM25}}}$.

---

## 2. Vector Memory Reference Table (Per 768-Dim Vector)

| Format | Storage Mechanism | Size of 1 Vector (768d) | Compression | Typical Recall@10 |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 (Raw Float)** | 4 Bytes per dimension | $768 \times 4\text{ B} = \mathbf{3,072\text{ B}}$ | $1\times$ | $100\%$ |
| **FP16 (Half Float)** | 2 Bytes per dimension | $768 \times 2\text{ B} = \mathbf{1,536\text{ B}}$ | $2\times$ | $99.9\%$ |
| **SQ8 / INT8 (Scalar)** | 1 Byte per dimension (`-127..127`) | $768 \times 1\text{ B} = \mathbf{768\text{ B}}$ | **$4\times$** | **$>99\%$** (Standard Default) |
| **PQ (96 Sub-Vectors)** | 1 Byte per 8-float sub-vector | $96 \times 1\text{ B} = \mathbf{96\text{ B}}$ | **$32\times$** | $75\text{–}88\%$ |
| **BQ (1-Bit Sign)** | 1 Bit (`0` or `1`) per dimension | $768\text{ bits} = \mathbf{96\text{ B}}$ | **$32\times$** | **$>95\%$** (with $3\times$ rescore & $d \ge 1024$) |

* **HNSW Graph Link Overhead ($M=16$):** Add **$\sim 170\text{ Bytes per vector}$** ($2 \times 16 \times 4\text{ B} \times 1.33$).

---

## 3. Core Equations

### A. BM25 Scoring Equation
$$\text{BM25}(D, Q) = \sum \text{IDF}(q_i) \cdot \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* **$\text{IDF}(q_i)$:** Weighs rare tokens (`0x84F2`) heavily and common stop-words (`the`) near zero.
* **$k_1 = 1.2$ (Term Frequency Saturation):** Caps the repetition multiplier at $k_1 + 1 = 2.2\times$ so repetitive log files do not dominate clean documentation.
* **$b = 0.4\text{–}0.75$ (Length Normalization):** Normalizes for document length (tuned lower to $0.4$ when RAG chunks have uniform token counts).

### B. Reciprocal Rank Fusion (RRF)
$$\text{RRF}(d) = \frac{1}{60 + \text{Rank}_{\text{Dense}}(d)} + \frac{1}{60 + \text{Rank}_{\text{BM25}}(d)}$$
* **Purpose of $k = 60$:** Prevents a single Rank #1 outlier in one retriever ($\frac{1}{61} = 0.01639$) from beating a chunk that achieves strong consensus (e.g., Rank #2) across both retrievers ($\frac{1}{62} + \frac{1}{62} = 0.03226$).

---

## 4. Top 3 Important Questions

**Q1: Why does IVF-PQ subtract the cluster centroid ($r = x - c_i$) before quantizing?**  
*Encoding a vector's offset from its local cluster center produces values with much smaller variance than global coordinates, allowing the 256 PQ codebook centroids to represent the vector with significantly lower reconstruction error.*

**Q2: When does HNSW + Quantization outperform IVF-PQ in production systems?**  
*IVF-PQ requires offline k-means training and degrades under **Concept Drift (Centroid Staleness)** when new document domains are added, requiring full re-indexing. HNSW is non-parametric, requires no centroid training, and supports real-time streaming inserts without accuracy degradation.*

**Q3: How does Binary Quantization maintain >95% recall while discarding decimal magnitudes?**  
*By combining high-dimensional mean-centered embeddings ($d \ge 1024$) with **Two-Stage Oversampled Rescoring**: searching the in-RAM 1-bit graph via `XOR + POPCOUNT` to shortlist $3\times\text{ to }4\times$ more candidates (e.g., 30 candidates for Top 10), then fetching only those 30 FP16 vectors from NVMe SSD (`mmap`) for exact cosine rescoring.*
