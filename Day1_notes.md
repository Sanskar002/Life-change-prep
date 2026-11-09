# Day 1 Quick Revision Cheat Sheet: Ingestion, RBAC & Indexing

*Read this 5-minute sheet before system design rounds to load key formulas, metrics, and trade-offs into working memory.*

---

## 1. Core Concepts in One Line

* **Layout-Aware Chunking:** Parse DOM/AST first; never split tables without cloning `<thead>` headers, and prepend `H1 > H2 > H3` breadcrumbs to every chunk.
* **Anthropic Contextual Retrieval:** Prepend a 50–100 token LLM-generated summary to each chunk using **Prefix KV Caching**; fixes **both** Dense Vectors and BM25 lexical search.
* **Jina Late Chunking:** Pass full 8k-token document through a long-context bi-encoder *before* mean-pooling by chunk spans; zero LLM generation cost, but **does not fix BM25** because raw chunk text remains unchanged.
* **Post-Filtering Flaw:** Search top-$K$ globally then filter by ACL $\rightarrow$ causes **Zero-Result Starvation** when user role matches <1% of corpus.
* **Naive Pre-Filtering Flaw:** Mask out unauthorized nodes before HNSW search $\rightarrow$ **shatters HNSW bridge edges** and collapses recall.
* **Filterable HNSW:** Builds extra payload-specific subgraph edges at ingestion time so filtered HNSW searches stay connected in a single stage.
* **Adaptive Query Planner:** Checks metadata posting-list cardinality first; if matched vectors < `full_scan_threshold` (~1,000–5,000), skips HNSW and runs an exact **AVX-512 SIMD flat scan** (<1 ms, 100% recall).
* **Two-Level Content Hashing:** Check `SHA256(doc)` then `SHA256(chunk)` against a PostgreSQL Record Manager so document edits only re-embed modified paragraphs.
* **HNSW Tombstones:** Deleted vectors are soft-flagged (`is_deleted = 1`) and still traversed during search, polluting the `efSearch` candidate queue until **Vacuum Compaction** rebuilds the segment.

---

## 2. Numbers & Configs to Drop in Interviews

| Metric / Parameter | Value to Quote | Why It Signals Seniority |
| :--- | :--- | :--- |
| **Contextual Retrieval Error Reduction** | **35%** (Dense only) $\rightarrow$ **49%** (+BM25) $\rightarrow$ **67%** (+Reranker) | Shows you know the exact Anthropic failure-rate reduction cascade (from 5.7% down to 1.9%). |
| **Raw Vector RAM Formula** | $\text{Vectors} \times d \times \text{Bytes} \times 1.5\text{ (HNSW overhead)}$ | E.g., $20\text{M} \times 768 \times 2\text{ B (FP16)} \approx 30.7\text{ GB raw} \rightarrow \sim 45\text{ GB total RAM}$ (or $\sim 24\text{ GB}$ with INT8). |
| **SIMD Brute-Force Speed** | **~1 ms per 1,000 vectors** (768-dim, AVX-512) | Justifies why `full_scan_threshold = 2000` beats HNSW graph traversal on restrictive filters. |
| **Qdrant Vacuum Thresholds** | `deleted_threshold = 0.20`, `vacuum_min_vector_number = 1000` | Proves you have tuned vector DB compaction in high-churn production environments. |

---

## 3. Rapid-Fire Interview Trap Questions & Answers

**Q1: "Since you trained bi-encoders at Samsung, why not use Late Chunking everywhere instead of paying for LLM Contextual Retrieval?"**  
> **Answer:** "Late Chunking only enriches the dense vector representation via token-level self-attention pooling—it leaves the raw chunk text untouched. In enterprise RAG where we rely on Hybrid Search (Dense + BM25), BM25 will still fail on pronoun-heavy chunks like *'it limits current to 45A'*. We use Late Chunking when ingestion GPU budget is tight, but we explicitly inject document titles and section breadcrumbs into the raw text so BM25 doesn't miss entity names."

**Q2: "Why can't we just store `allowed_user_ids` in the vector payload for RBAC?"**  
> **Answer:** "Storing user IDs directly creates an ACL Update Storm—when one employee switches departments, we'd have to mutate metadata across hundreds of thousands of chunks, triggering massive segment rewrites. Instead, we use **Group Indirection**: chunks store immutable `allowed_group_ids`, and we resolve `user_id -> group_ids` at query time in <1 ms via Redis or JWT claims."

**Q3: "Users complain that search recall drops and latency spikes every Friday after heavy Confluence edits, even though total document count hasn't grown. Why?"**  
> **Answer:** "That's **HNSW Tombstone Bloat**. Every edited chunk soft-deletes the old vector via a tombstone bitmask, but the HNSW walker still visits and computes distances on dead nodes during graph traversal. Those dead nodes consume slots in the `efSearch` candidate priority queue, pushing out live neighbors. We fix this by tuning the Vacuum Optimizer (`deleted_threshold = 0.15–0.20`) to compact segments and purge tombstones asynchronously."

---

## 4. 4-Step Whiteboard Checklist (Ingestion & Retrieval)

Whenever an interviewer asks you to design an ingestion/retrieval pipeline, draw these 4 boxes in order:
1. **Kafka (Partitioned by `doc_id`)** $\rightarrow$ Prevents race conditions when the same doc is edited twice in 1 second.
2. **DOM/AST Chunker + Record Manager (PostgreSQL)** $\rightarrow$ Clone table headers, inject breadcrumbs, and diff `SHA256(chunk)` to skip unchanged chunks.
3. **Contextualizer / Bi-Encoder** $\rightarrow$ Prefix-cached Contextual Retrieval or 8k Late Chunking.
4. **Vector DB with Filterable HNSW + Vacuum Optimizer** $\rightarrow$ Group-level RBAC payload index, adaptive cardinality planner (`full_scan_threshold`), and automated tombstone compaction.
