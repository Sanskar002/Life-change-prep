# Day 3 Complete Technical Study Guide: HNSW, IVF-PQ, Vector Quantization & Hybrid Search (BM25 + RRF)

**Scope:** Vector Index Internals, Memory Compression Math & Hybrid Retrieval Fusion  
**Prerequisites:** None — explained from first principles with step-by-step numeric examples.  
**Reading Time:** 50 Minutes (100% self-contained)

---

## The Big Picture: What Problem Are We Solving?

Suppose a system stores **50 million text chunks** from enterprise technical documents. Passing each chunk through an embedding model (such as a 768-dimension bi-encoder) turns every chunk into a list of **768 decimal numbers** (a **vector**):
$$\text{Chunk}_1 \longrightarrow [0.12, -0.45, 0.88, \dots, 0.03] \quad (768\text{ numbers})$$

When a user submits a query, the system converts that query into a 768-number vector $q$ and searches for the **10 most similar chunks**. Comparing $q$ against all 50 million vectors one by one (**Brute-Force Search**) creates two major bottlenecks:
1. **High Latency:** Executing $50,000,000 \times 768$ floating-point operations per query takes hundreds of milliseconds of CPU time.
2. **High Memory Footprint:** Storing 50 million uncompressed vectors in server RAM requires massive memory capacity.

Three core engineering techniques solve these bottlenecks:
* **Block 1 (Index Structures — HNSW vs. IVF):** Finding the top 10 vectors by inspecting only **0.1%** of the database.
* **Block 2 (Quantization — SQ8, PQ, Binary):** Compressing vectors so they consume **$4\times$ to $32\times$ less RAM**.
* **Block 3 (Hybrid Search — BM25 + RRF):** Combining semantic vector search with exact keyword search so part numbers, SKUs, and error codes are never missed.

---

## Block 1: How Vector Indexes Work — Brute-Force vs. HNSW vs. IVF (Minutes 0–20)

### 1.1 How Much RAM Does One Vector Actually Take? (Step-by-Step Math)
In computer memory, a standard decimal number (**32-bit Floating Point**, or **FP32**) occupies **4 Bytes** of RAM.
* If 1 vector contains **768 decimal numbers (FP32)**:
  $$\text{Size of 1 Vector} = 768\text{ numbers} \times 4\text{ Bytes} = \mathbf{3,072\text{ Bytes}} \approx \mathbf{3\text{ KB}}$$
* Scaling to **50 million vectors**:
  $$50,000,000 \times 3,072\text{ Bytes} = 153,600,000,000\text{ Bytes} = \mathbf{153.6\text{ GB of RAM}}$$
Holding the raw numbers alone requires over 150 GB of memory. Next, let's look at how to search those vectors without scanning all 50 million.

---

### 1.2 HNSW (Hierarchical Navigable Small World) — "The Flight-to-Street Map"

#### Conceptual Analogy
Imagine navigating to a specific house in a small village without an address book—you can only ask locals, *"Which of your connected neighbors lives closer to my destination?"*
* **Layer 2 (Top Layer — Express Flights):** Only 5 major hub nodes exist here (e.g., Delhi, Mumbai, Bengaluru, Kolkata, Chennai) connected by long-distance links. You start at Delhi, check which hub is closest to your destination coordinates, and jump straight to **Bengaluru**.
* **Layer 1 (Middle Layer — State Highways):** You drop down to the regional highway network around Bengaluru. Now regional nodes like Mysuru, Hosur, and Tumakuru appear. You hop along the highway to **Mysuru**.
* **Layer 0 (Bottom Layer — Local Streets):** Every single data point (all 50 million vectors) resides on this bottom layer. Because the upper layers already dropped you into the exact neighborhood, you only traverse ~10 local edges to reach the exact target vector.

```text
Layer 2 (Few Nodes, Long Jumps):     [Node A] ───────────────────────► [Node B]
                                                                          │ (Drop Down)
Layer 1 (Medium Nodes, Medium Jumps): [Node A] ──► [Node C] ─────────► [Node B] ──► [Node D]
                                                                                       │ (Drop Down)
Layer 0 (ALL 50M Vectors, Short Hops): [A]─[E]─[C]─[F]─[G]─[H]─[B]─[I]─[D]─► [TARGET!]
```

#### The Two Key HNSW Parameters
1. **$M$ (Max Edges per Node, typically $16$):** How many neighbor links each vector stores in Layer 0.
   * Higher $M$ (e.g., 32 or 64) increases recall accuracy because there are more paths to navigate, **but consumes more RAM** to store the graph pointers.
   * **Memory Math for HNSW Links:** With $M = 16$, each node stores up to $2 \times M = 32$ bidirectional links. Each link is a 4-byte internal ID, plus ~33% overhead for upper layers:
     $$\text{Link RAM per Vector} \approx 32 \times 4\text{ Bytes} \times 1.33 \approx \mathbf{170\text{ Bytes per vector}}$$
     Across 50 million vectors, the graph edges alone consume $50\text{M} \times 170\text{ B} = \mathbf{8.5\text{ GB of additional RAM}}$ on top of the $153.6\text{ GB}$ of raw vectors.
2. **`efSearch` (Exploration Queue Width, typically $64$ to $128$):** How many top candidate nodes the walker tracks in its priority queue while exploring Layer 0.
   * Higher `efSearch` improves recall accuracy at the cost of higher query latency.

---

### 1.3 IVF (Inverted File Index) — "The Postal Pin-Code Buckets"

When the extra $8.5\text{ GB}$ of RAM for HNSW graph links is too expensive, **IVF** provides a cluster-based alternative.

#### Conceptual Analogy
Instead of wiring vectors together with road links, IVF divides all 50 million vectors into **10,000 Postal Pin-Code Zones** (**Voronoi Cells** or **Clusters**) using k-means clustering. Each zone has a central reference point (a **Centroid**).
* **Parameter 1 — `nlist` (Total Number of Zones, e.g., $10,000$):** During index creation, every vector is assigned to the bucket of its nearest centroid.
* **Parameter 2 — `nprobe` (Number of Zones Scanned at Query Time, e.g., $20$):**
  When a query vector $q$ arrives:
  1. Compare $q$ against the 10,000 zone centroids.
  2. Select the **top `nprobe = 20` closest zones** (handling boundary cases where the true nearest neighbor sits just across a zone border).
  3. Scan **only the vectors inside those 20 buckets** ($\frac{20}{10,000} = 0.2\%$ of the dataset).

```text
All Vectors Divided into Pin-Code Buckets (nlist = 4 shown):
 ┌─────────────────┬─────────────────┐
 │  Zone 1 (• C1)  │  Zone 2 (• C2)  │   Query (★) lands near the border of Zone 2 & Zone 4.
 │                 │        ★ Query  │   With nprobe = 2, the engine ONLY scans Zone 2 and Zone 4,
 ├─────────────────┼─────────────────┤   skipping Zone 1 and Zone 3 completely.
 │  Zone 3 (• C3)  │  Zone 4 (• C4)  │
 └─────────────────┴─────────────────┘
```

---

## Block 2: Shrinking Vector RAM — SQ8, Product Quantization (PQ) & Binary Quantization (BQ) (Minutes 20–35)

Even with HNSW or IVF, 50 million raw vectors still require **153.6 GB of RAM**. **Quantization** compresses the numbers inside each vector so they occupy fewer bytes.

---

### 2.1 Method 1: Scalar Quantization (SQ8 / INT8) — "Rounding Decimals to Whole Numbers"
* **Mechanism:** Instead of storing each dimension as a 4-byte float like `0.4829104` (FP32), we scale and round each value to a signed 8-bit integer between **`-127` and `+127`** (**INT8**, which takes **1 Byte**).
* **Toy Example (4-number vector):**
  $$\text{FP32 (16 Bytes): } [-1.00, \; -0.25, \; +0.50, \; +1.00] \xrightarrow{\times 127} \text{INT8 (4 Bytes): } [-127, \; -32, \; +64, \; +127]$$
* **Impact:** Reduces vector RAM by **$4\times$** ($153.6\text{ GB} \rightarrow \mathbf{38.4\text{ GB}}$) with **less than 1% recall loss**. This is the safest default compression method in production.

---

### 2.2 Method 2: Product Quantization (PQ) — "The Paint-Shop Color Swatch Book"
When $38.4\text{ GB}$ is still too large and 50 million vectors must fit into **~5 GB of RAM**, combining IVF with **Product Quantization (IVF-PQ)** achieves **$32\times$ compression**.

#### Step 1: Subtract the Local Centroid First ("Residual Vector")
Before compressing a vector $x$ inside IVF Zone $c_i$, the engine subtracts the zone centroid's coordinates:
$$r = x - c_i$$
* **Why Residual Encoding Matters:** Describing a building's position relative to its local neighborhood center ("200m north of the station") requires much smaller numbers than giving global coordinates from the Equator. Subtracting $c_i$ centers all vectors in the cell around zero and reduces variance, lowering quantization rounding error.

#### Step 2: Slice and Replace with Codebook IDs (Toy Example)
Suppose a vector has **8 decimal numbers** ($8 \times 4\text{ Bytes} = \mathbf{32\text{ Bytes}}$).
1. **Split into $m = 2$ sub-vectors** of 4 numbers each:
   $$\underbrace{[0.11, \; 0.89, \; -0.52, \; 0.04]}_{\text{Sub-Vector 1 (16 Bytes)}} \quad \Big\vert{} \quad \underbrace{[-0.77, \; 0.21, \; 0.19, \; -0.81]}_{\text{Sub-Vector 2 (16 Bytes)}}$$
2. **Match against a 256-Pattern Swatch Book (Codebook):** During training, k-means learns 256 representative 4-number patterns (numbered `0` to `255`).
   * Sub-Vector 1 is closest to **Pattern `#42`** (`[0.10, 0.90, -0.50, 0.05]`).
   * Sub-Vector 2 is closest to **Pattern `#198`** (`[-0.75, 0.20, 0.20, -0.80]`).
3. **Store ONLY the Pattern IDs:** Because an integer from `0` to `255` fits in **1 Byte (`uint8`)**, we discard the 32 bytes of floats and store just two bytes:
   $$\mathbf{[42, \; 198]} \quad (\text{Only }\mathbf{2\text{ Bytes}}\text{ instead of 32 Bytes})$$
* **Production Scale (768 dimensions):** A 768-float vector ($3,072\text{ Bytes}$) is split into **96 sub-vectors** of 8 floats each. Each sub-vector becomes a 1-byte Pattern ID:
  $$3,072\text{ Bytes} \longrightarrow \mathbf{96\text{ Bytes per vector}} \quad (\mathbf{32\times\text{ smaller: } 153.6\text{ GB} \to 4.8\text{ GB}})$$

#### Step 3: Fast Query Scoring via Asymmetric Distance Lookup Tables (LUT)
At query time, IVF-PQ never decompresses `[42, 198]` back into floats:
1. When query $q$ arrives, it is split into sub-vectors and compared against all 256 patterns in each sub-space codebook, storing the results in a small **Lookup Table (LUT)**.
2. Scoring any compressed vector `[42, 198]` requires only array lookups and addition:
   $$\text{Distance} = \text{LUT}_1[42] + \text{LUT}_2[198]$$

#### Operational Limitation of IVF-PQ: Concept Drift & Centroid Staleness
Because IVF-PQ relies on pre-trained cluster centroids and pre-trained PQ codebooks, it degrades under **Concept Drift (Centroid Staleness)**. If an index is trained on legacy documents in January, and millions of documents covering a brand-new product domain are ingested in June, the new vectors pile into a few unbalanced clusters and poorly match the old codebook patterns. Recall drops sharply until the entire index is re-clustered offline. **HNSW avoids this issue** because it is non-parametric and requires no k-means training.

---

### 2.3 Method 3: Binary Quantization (BQ) — "Turning Every Number into 1 or 0"
To achieve **$32\times$ compression** inside HNSW *without* PQ codebooks, engines use **Binary Quantization**.

* **Mechanism:** Inspect each of the 768 numbers. If the value is **greater than 0**, store bit **`1`**. If it is **0 or negative**, store bit **`0`**.
* **Toy 4-Number Example:**
  $$\text{Query } q = [+0.82, \; -0.14, \; +0.05, \; +0.91] \longrightarrow \mathbf{1011}$$
  $$\text{Doc } A = [+0.75, \; -0.60, \; -0.02, \; +0.88] \longrightarrow \mathbf{1001}$$
* **Fast Bitwise Distance (Hamming Distance):**
  Count how many bit positions differ between `1011` and `1001`:
  * Position 1 (`1` vs `1`), Position 2 (`0` vs `0`), and Position 4 (`1` vs `1`) match.
  * Position 3 (`1` vs `0`) differs $\rightarrow$ **Distance = 1 bit**.
  Modern CPUs execute bitwise `XOR` + `POPCOUNT` across 512 bits in a single clock cycle, making Binary Quantization up to **$40\times$ faster** than float math and **$32\times$ smaller in RAM**.

#### Restoring High Recall: Two-Stage Oversampling + Disk Rescoring
Binarizing floats loses magnitude precision (and works best when embedding dimension $d \ge 1024$ with mean-centered vectors). To recover **98%+ recall** without holding FP32 vectors in RAM:
1. **Stage 1 (Fast Binary Shortlist in RAM):** To return the **Top 10** chunks, search the 1-bit Binary HNSW graph in RAM with an oversampling factor of $3\times$ to fetch the **Top 30** binary candidates in ~2 ms.
2. **Stage 2 (Exact Rescore from SSD Disk):** Read the full-precision vectors for **only those 30 candidates** from NVMe SSD (`mmap`), compute exact cosine similarity, and return the top 10.

---

## Block 3: Hybrid Search — BM25 Keyword Math + Reciprocal Rank Fusion (RRF) (Minutes 35–50)

### 3.1 Why Vector Search Alone Is Insufficient
Dense embeddings capture **semantic meaning** (*"battery overheating"* matches *"thermal runaway"*), but struggle with **exact tokens**:
* Searching for hex error code **`ERR_BMS_0x84F2`** often retrieves chunks containing **`ERR_BMS_0x84F9`** because subword tokenizers split codes into similar fragments.
* Production retrieval pipelines therefore run two searches in parallel (**Hybrid Search**):
  1. **Dense Vector Search** (captures semantic similarity).
  2. **BM25 Keyword Search** (captures exact terms, SKUs, and error codes).

---

### 3.2 BM25 Explained via 3 Core Rules
**BM25 (Best Matching 25)** scores how well document chunk $D$ matches the exact words in query $Q$:

$$\text{BM25}(D, Q) = \sum_{\text{word } q_i \in Q} \underbrace{\text{IDF}(q_i)}_{\text{Rule 1: Rarity}} \times \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}D\vert{}}{\text{avgdl}}\right)}$$

1. **Rule 1 — Word Rarity ($\text{IDF}$):**  
   In a query like `"the 0x84F2 error"`:
   * `"the"` appears in almost every chunk $\rightarrow$ **~0 points**.
   * `"error"` appears in 10% of chunks $\rightarrow$ **~2 points**.
   * `"0x84F2"` appears in 3 chunks out of 1,000,000 $\rightarrow$ **~13 points**.
2. **Rule 2 — Diminishing Returns for Repetition ($k_1$, typically $1.2$):**  
   Here $f$ is the count of the word inside chunk $D$.
   * **Problem with raw counts (TF-IDF):** A noisy log file repeating `"0x84F2"` 500 times would score $250\times$ higher than a clean specification mentioning `"0x84F2"` twice.
   * **How $k_1 = 1.2$ solves it:** The repetition multiplier is capped at an upper bound of **$k_1 + 1 = 2.2\times$**:
     * Appears **1 time** ($f=1$, average length): Multiplier $= \frac{1 \times 2.2}{1 + 1.2} = \mathbf{1.0\times}$.
     * Appears **5 times** ($f=5$): Multiplier $= \frac{5 \times 2.2}{5 + 1.2} = \mathbf{1.77\times}$.
     * Appears **500 times** ($f=500$): Multiplier $= \frac{500 \times 2.2}{500 + 1.2} = \mathbf{2.19\times}$ (never exceeds $2.2\times$).
3. **Rule 3 — Length Normalization ($b$, typically $0.75$ or $0.4$ in RAG):**  
   $\vert{}D\vert{} / \text{avgdl}$ compares chunk $D$'s length to the average chunk length. Because RAG chunks are already split to uniform sizes (e.g., 256–400 tokens), setting a lower $b \approx 0.4$ avoids over-penalizing slightly longer table chunks.

---

### 3.3 Reciprocal Rank Fusion (RRF) — Combining Dense and BM25 Results

#### Why Raw Score Addition Fails
* **Dense Cosine Similarity Scores** sit in a narrow bounded range between `-1.0` and `+1.0` (e.g., `0.84`, `0.81`).
* **BM25 Keyword Scores** are unbounded `[0, ∞)` (e.g., `38.5`, `14.2`).
* Adding `0.84 + 38.5` allows BM25 to dominate completely. Even per-query Min-Max normalization fails when a single keyword-stuffed chunk gets an outlier BM25 score of `50.0`, squashing the second-best BM25 hit (`14.2`) down to `14.2 / 50.0 = 0.28`.

#### How RRF Works: Rank-Based Consensus
**Reciprocal Rank Fusion (RRF)** ignores raw scores and merges lists using only each chunk's **ordinal position (Rank #1, Rank #2, Rank #3...)** in each retriever:

$$\text{RRF\_Score}(\text{Chunk}) = \frac{1}{60 + \text{Rank}_{\text{Dense}}} + \frac{1}{60 + \text{Rank}_{\text{BM25}}}$$

*(If a chunk does not appear in a retriever's top-$K$ list, its term for that retriever is $0$).*

#### Step-by-Step Numeric Example
For the query `"BMS 0x84F2 thermal limit"`, compare three candidate chunks:

* **Chunk A (Relevant Spec Chunk):** Ranked **#2 in Dense Search** AND Ranked **#2 in BM25 Search**.
  $$\text{RRF\_Score}(A) = \frac{1}{60 + 2} + \frac{1}{60 + 2} = \frac{1}{62} + \frac{1}{62} = 0.01613 + 0.01613 = \mathbf{0.03226} \quad \longrightarrow \mathbf{\text{FINAL RANK \#1}}$$
* **Chunk B (Noisy Log File with `0x84F2`):** Ranked **#1 in BM25 Search**, missing from Dense Search ($0$).
  $$\text{RRF\_Score}(B) = 0 + \frac{1}{60 + 1} = \frac{1}{61} = \mathbf{0.01639} \quad \longrightarrow \mathbf{\text{FINAL RANK \#2}}$$
* **Chunk C (General Thermal Overview without `0x84F2`):** Ranked **#1 in Dense Search**, missing from BM25 Search ($0$).
  $$\text{RRF\_Score}(C) = \frac{1}{60 + 1} + 0 = \frac{1}{61} = \mathbf{0.01639} \quad \longrightarrow \mathbf{\text{FINAL RANK \#3}}$$

> **Why is the smoothing constant $k = 60$ necessary?**  
> Without $60$ (using $\frac{1}{\text{Rank}}$), Chunk B (Rank #1 in BM25 only) would score $\frac{1}{1} + 0 = \mathbf{1.0}$, beating a chunk ranked #2 and #3 across both retrievers ($\frac{1}{2} + \frac{1}{3} = \mathbf{0.83}$). Adding $60$ flattens the gap between Rank #1 ($\frac{1}{61} = 0.01639$) and Rank #2 ($\frac{1}{62} = 0.01613$), ensuring **consensus across both retrieval engines always wins**.

---

## Architecture Design Scenario: 100M Vectors Under 40 GB RAM

> **Design Requirement:** *Index 100 million chunks (1024-dim vectors) supporting both semantic queries and exact hex error codes. Keep total RAM under 40 GB and p99 retrieval latency under 25 ms.*

1. **Raw Memory Baseline:**  
   $100\text{M} \times 1024 \times 4\text{ Bytes (FP32)} \approx \mathbf{410\text{ GB}}$ (exceeds the 40 GB budget by $10\times$).
2. **In-RAM Binary HNSW + SSD `mmap` Storage:**  
   * Apply mean-centering and **1-Bit Binary Quantization** to the 1024-dim vectors, compressing each vector from $4,096\text{ Bytes}$ down to $1024\text{ bits} = \mathbf{128\text{ Bytes}}$ ($32\times$ reduction).
   * **Binary Vectors in RAM:** $100\text{M} \times 128\text{ Bytes} = \mathbf{12.8\text{ GB}}$.
   * **HNSW Graph Links in RAM ($M=16$):** $100\text{M} \times 170\text{ Bytes} = \mathbf{17.0\text{ GB}}$.
   * **BM25 Inverted Index Cache:** $\sim\mathbf{6.0\text{ GB}}$.
   * **Total RAM Footprint:** $12.8 + 17.0 + 6.0 = \mathbf{35.8\text{ GB}}$ (within the 40 GB target), while full FP16 vectors reside on NVMe SSD via `mmap`.
3. **Parallel Query Execution & RRF (<15 ms):**  
   * **Dense Path:** Search the in-RAM Binary HNSW graph via `XOR + POPCOUNT` with $4\times$ oversampling (fetching 200 candidates in ~4 ms), then rescore those 200 vectors from SSD `mmap` in FP16 (~3 ms) to produce the Top 50 Dense hits.
   * **Lexical Path:** Run BM25 ($k_1=1.2, b=0.4$) in parallel to fetch the Top 50 Lexical hits (~5 ms).
   * **Fusion:** Merge both Top-50 lists via **RRF ($k=60$)** in <1 ms and pass the top 20 candidates to the downstream reranker.
