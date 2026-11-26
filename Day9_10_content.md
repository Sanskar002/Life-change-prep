# Days 9–10 Complete Technical Study Guide: The RAG Triad, Offline Retrieval Metrics, Failure Debugging & Capacity/Cost Math

**Scope:** Automated RAG Evaluation (RAG Triad & LLM-as-a-Judge), Offline Ranking Metrics (`Recall@K`, `MRR`, `NDCG@K`), Systematic Failure Diagnosis, and Rapid Architecture Math (QPS, Storage, Cost per 1,000 Queries)  
**Prerequisites:** None — every metric, formula, and calculation is built from scratch with step-by-step numeric examples.  
**Reading Time:** 60 Minutes (100% self-contained)

---

## The Big Picture: Moving From "Vibe Checks" to Measurable Engineering

In Days 1–8, we built the components of a production RAG pipeline: Chunking, RBAC Vector Filtering, Hybrid BM25 + RRF Search, Cross-Encoder Reranking, and Knowledge Graphs.

Now suppose you change your chunk size from `512` tokens to `256` tokens, or swap in a new embedding model. How do you know if the system actually got **better** or silently **broke** on 20% of edge cases? Looking at 5 manual examples ("vibe checking") fails at scale.

Production AI systems rely on **four engineering pillars**:
1. **Block 1 (The RAG Triad & LLM-as-a-Judge):** Automatically grading Retrieval Quality, Hallucinations, and Final Answer Utility.
2. **Block 2 (Offline Retrieval Metrics):** Mathematical scores (`Recall@K`, `MRR`, `NDCG@K`) to measure Stage-1 search and Stage-2 reranking independently before calling an LLM.
3. **Block 3 (Systematic Failure Debugging):** A step-by-step diagnostic playbook to pinpoint *which* component broke instead of guessing with random prompt edits.
4. **Block 4 (2-Minute Capacity & Cost Math):** Rapid mental formulas to estimate QPS, RAM/SSD sizing, and dollars per 1,000 queries.

---

## Block 1: The RAG Triad & LLM-as-a-Judge Mechanics (Minutes 0–15)

### 1.1 The Three Sides of the RAG Triad
Every RAG execution has exactly **three text artifacts**:
1. **Query ($Q$):** What the user asked.
2. **Retrieved Context ($C$):** The top chunks fetched by your search + reranker pipeline.
3. **Generated Answer ($A$):** The final response written by the LLM.

The **RAG Triad** evaluates the **three links** connecting these artifacts so you immediately know whether a bad answer was caused by your **Search Engine** or by your **LLM**:

```text
                        [User Query (Q)]
                           /        \
                          /          \
   1. CONTEXT RELEVANCE  /            \  3. ANSWER RELEVANCE
   (Did Retrieval fetch /              \ (Does the final response
    useful chunks?)    /                \ actually answer Q?)
                      /                  \
       [Retrieved Context (C)] ─────── [Generated Answer (A)]
                       2. GROUNDEDNESS / FAITHFULNESS
                      (Is every claim in A strictly backed by C,
                       with zero LLM hallucination?)
```

#### Concrete Walkthrough Using the Triad
Suppose a user asks **$Q$:** *"What is the cold-weather charging limit for the BMS at -20°C?"*

* **Metric 1 — Context Relevance ($Q \leftrightarrow C$):** Measures whether the retrieved chunks actually contain the facts needed to answer $Q$ (without flooding the prompt with irrelevant paragraphs).
  * *High Score:* Retrieved chunk says *"At -20°C, the BMS caps charging current at 45A."*
  * *Low Score:* Retrieved chunks only talk about vehicle paint colors and seat heaters.
* **Metric 2 — Groundedness / Faithfulness ($C \leftrightarrow A$):** Measures whether **every single factual claim** in the LLM's answer is strictly supported by Context $C$—detecting **hallucinations**.
  * *How Automated Frameworks Score Faithfulness (Atomic Claim Decomposition):*
    1. An evaluator model breaks the LLM's answer $A$ into individual atomic statements:
       * Claim 1: *"The BMS caps current at 45A at -20°C."* $\rightarrow$ **Supported by $C$ (Yes)**
       * Claim 2: *"This limit prevents lithium plating."* $\rightarrow$ **Supported by $C$ (Yes)**
       * Claim 3: *"Charging resumes full speed after 30 minutes."* $\rightarrow$ **NOT in $C$ (Hallucination!)**
    2. Compute the ratio:
       $$\text{Faithfulness} = \frac{\text{Supported Claims}}{\text{Total Claims}} = \frac{2}{3} = \mathbf{0.67}$$
* **Metric 3 — Answer Relevance ($Q \leftrightarrow A$):** Measures whether the final answer actually addresses the user's question directly.
  * *Example of High Faithfulness but Low Answer Relevance:* The chunk states both the battery weight (`450 kg`) and the current limit (`45A`). If the LLM replies *"The battery weighs 450 kg"*, that statement is 100% faithful to the context (zero hallucination!), yet Answer Relevance is **0.0** because it ignored the user's question about current limits.

---

### 1.2 LLM-as-a-Judge Biases & How to Neutralize Them
Because paying humans to grade 5,000 queries after every code commit is impossible, engineering teams use a strong LLM (or fine-tuned judge SLM) to score the RAG Triad automatically. However, naive LLM judges have **three well-known failure modes**:

| Judge Bias | What Goes Wrong | Production Mitigation |
| :--- | :--- | :--- |
| **1. Position Bias** | When comparing `System A` vs. `System B` side-by-side, LLM judges favor whichever answer appears **first** (Option A) ~60–70% of the time. | **Position Swapping:** Run every pairwise comparison twice—once as `(A, B)` and once as `(B, A)`. Only count a win if the judge picks the same system in **both** orders (otherwise declare a tie). |
| **2. Verbosity Bias** | Judges reward long, polite, bullet-heavy answers over short, exact, factual answers—even when the long answer contains fluff. | Use **Atomic Claim Decomposition** (boolean `Yes/No` per fact) and strict **1-to-5 Rubric Anchors** explicitly instructing the judge not to reward length. |
| **3. Self-Preference Bias** | A model from Family X tends to rate outputs written by Family X higher than outputs written by Family Y. | Use a **Cross-Family Judge** and calibrate judge scores against a human-labeled **Golden Dataset (100–200 queries)** until human-to-judge agreement reaches **Cohen's Kappa $\kappa \ge 0.75$**. |

---

## Block 2: Offline Retrieval Metrics — `Recall@K`, `MRR` & `NDCG@K` (Minutes 15–30)

Before you ever run an expensive LLM generator, you must evaluate your **Search + Reranker layers** offline against a **Golden Test Set** (a list of 200+ real queries paired with the exact document IDs that contain the answers).

There are **three core metrics**, and each one measures a completely different engineering property:

---

### 2.1 Metric 1: `Recall@K` — "Did the Right Chunks Make It Into the Shortlist?"
* **Best Used For:** Evaluating **Stage-1 Candidate Generation** (Bi-Encoder + BM25 + RRF, e.g., `Recall@50`).
* **Plain-English Definition:** Out of all the truly relevant documents that exist in the entire database for Query $Q$, what fraction showed up inside our Top $K$ retrieved results?
* **Formula:**
  $$\text{Recall@K} = \frac{\text{Number of Relevant Docs Found in Top } K}{\text{Total Relevant Docs in the Database for Query } Q}$$

#### Toy Numeric Example
* Suppose the entire 10M-document database contains **4 truly relevant chunks** for Query $Q$: `{D1, D2, D3, D4}`.
* Your Stage-1 Search returns **Top $K = 5$** chunks: `[D9, D2, D7, D1, D4]`.
* How many of the 4 true chunks appeared in your Top 5? `{D2, D1, D4}` = **3 chunks**.
  $$\text{Recall@5} = \frac{3}{4} = \mathbf{0.75} \; (75\%)$$

* **Crucial Blind Spot of `Recall@K`:** It completely ignores **ranking order**! Whether your system returns `[D2, D1, D4, D9, D7]` (relevant items at the very top) or `[D9, D7, D2, D1, D4]` (junk at the top), `Recall@5` is identically `0.75`. That is why we pair `Recall@K` with rank-aware metrics below.

---

### 2.2 Metric 2: `MRR` (Mean Reciprocal Rank) — "How High Is the *First* Correct Answer?"
* **Best Used For:** Single-answer lookup or navigational queries (e.g., *"What is error code `0x84F2`?"*) where the user only needs **one** exact chunk at Rank #1.
* **Plain-English Definition:** For each query, find the 1-based rank position of the **very first relevant chunk** ($\text{rank}_i$), take its reciprocal ($\frac{1}{\text{rank}_i}$), and average across all queries:
  $$\text{MRR} = \frac{1}{\vert{}Q\vert{}} \sum_{i=1}^{\vert{}Q\vert{}} \frac{1}{\text{rank}_i}$$
  *(If zero relevant documents are returned for a query, its Reciprocal Rank is `0.0`).*

#### Toy Numeric Example (3 Queries)
* **Query 1:** First relevant chunk lands at **Rank #1** $\longrightarrow \text{RR}_1 = \frac{1}{1} = \mathbf{1.000}$
* **Query 2:** First relevant chunk lands at **Rank #2** $\longrightarrow \text{RR}_2 = \frac{1}{2} = \mathbf{0.500}$
* **Query 3:** First relevant chunk lands at **Rank #4** $\longrightarrow \text{RR}_3 = \frac{1}{4} = \mathbf{0.250}$
$$\text{MRR} = \frac{1.000 + 0.500 + 0.250}{3} = \frac{1.750}{3} = \mathbf{0.5833}$$

* **Crucial Blind Spot of `MRR`:** It stops looking the instant it sees the **first** relevant chunk! If a multi-part question requires 3 chunks to synthesize an answer, and Chunks #2 and #3 are buried at Rank 45, `MRR` still scores a deceptive `1.0` just because Chunk #1 was at Rank 1.

---

### 2.3 Metric 3: `NDCG@K` (Normalized Discounted Cumulative Gain) — The Gold Standard for Rerankers & RAG
* **Best Used For:** Evaluating **Stage-2 Cross-Encoder Rerankers** and multi-chunk RAG (`NDCG@5` or `NDCG@10`).
* **Why `NDCG@K` Beats `Recall` and `MRR`:** It solves **both** blind spots at once:
  1. **Graded Relevance (Not Just Binary Yes/No):** Chunks can be labeled on a scale (e.g., `0` = Irrelevant, `1` = Slightly useful, `2` = Relevant, `3` = Exact Gold Answer).
  2. **Position Discounting:** Putting a Grade-3 chunk at Rank #1 earns far more credit than putting it at Rank #4.

#### Building the Formula One Piece at a Time (No Prior Knowledge Needed)
Break the scary name **N-D-C-G** into its 4 simple English words:

1. **Gain ($2^{\text{rel}_i} - 1$):** Converts the relevance grade of the chunk at position $i$ into points (using exponential gain so a Grade-3 exact answer earns much more than a Grade-1 vague hint):
   * Grade `0` $\rightarrow 2^0 - 1 = \mathbf{0\text{ pts}}$
   * Grade `1` $\rightarrow 2^1 - 1 = \mathbf{1\text{ pt}}$
   * Grade `2` $\rightarrow 2^2 - 1 = \mathbf{3\text{ pts}}$
   * Grade `3` $\rightarrow 2^3 - 1 = \mathbf{7\text{ pts}}$
2. **Discounted ($\div \log_2(i + 1)$):** Divides the points by a penalty that grows as rank $i$ gets lower:
   * Rank $i = 1$: divide by $\log_2(2) = \mathbf{1.0000}$ (100% credit!)
   * Rank $i = 2$: divide by $\log_2(3) \approx \mathbf{1.5850}$ (~63% credit)
   * Rank $i = 3$: divide by $\log_2(4) = \mathbf{2.0000}$ (50% credit)
   * Rank $i = 4$: divide by $\log_2(5) \approx \mathbf{2.3219}$ (~43% credit)
3. **Cumulative (Sum from $i=1$ to $K$):** Add the discounted points for positions $1$ through $K$ to get **DCG@K**:
   $$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
4. **Normalized ($\div \text{IDCG@K}$):** Different queries have different numbers of relevant docs, so raw DCG scores aren't comparable across queries. Divide your actual **DCG@K** by the **Ideal DCG (IDCG@K)**—the maximum possible score achieved if the **best relevant chunks in the entire labeled corpus** were sorted perfectly from highest grade to lowest grade:
   $$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}} \in [0.0, \; 1.0]$$

#### Step-by-Step Hand Calculation (`NDCG@4`)
Suppose our labeled database for Query $Q$ has 4 chunks with ground-truth grades: `{D1: 1, D2: 3, D3: 0, D4: 2}`.  
Our search engine returns them in this imperfect order: **`[D1 (rel=1), D2 (rel=3), D3 (rel=0), D4 (rel=2)]`**.

**Step A: Compute Actual `DCG@4`**
| Rank ($i$) | Returned Doc | Grade ($\text{rel}_i$) | Gain ($2^{\text{rel}_i} - 1$) | Discount ($\log_2(i+1)$) | Discounted Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `D1` | `1` | `1` | `1.0000` | $1 / 1.0000 = \mathbf{1.0000}$ |
| **2** | `D2` | `3` | `7` | `1.5850` | $7 / 1.5850 = \mathbf{4.4165}$ |
| **3** | `D3` | `0` | `0` | `2.0000` | $0 / 2.0000 = \mathbf{0.0000}$ |
| **4** | `D4` | `2` | `3` | `2.3219` | $3 / 2.3219 = \mathbf{1.2920}$ |
| | | | | **Total `DCG@4`** | **$6.7085$** |

**Step B: Compute Ideal `IDCG@4` (Sort all ground-truth corpus grades highest first: `[3, 2, 1, 0]`)**
| Rank ($i$) | Ideal Doc | Grade ($\text{rel}_i$) | Gain ($2^{\text{rel}_i} - 1$) | Discount ($\log_2(i+1)$) | Ideal Discounted Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `D2` | `3` | `7` | `1.0000` | $7 / 1.0000 = \mathbf{7.0000}$ |
| **2** | `D4` | `2` | `3` | `1.5850` | $3 / 1.5850 = \mathbf{1.8928}$ |
| **3** | `D1` | `1` | `1` | `2.0000` | $1 / 2.0000 = \mathbf{0.5000}$ |
| **4** | `D3` | `0` | `0` | `2.3219` | $0 / 2.3219 = \mathbf{0.0000}$ |
| | | | | **Total `IDCG@4`** | **$9.3928$** |

**Step C: Divide to Normalize**
$$\text{NDCG@4} = \frac{6.7085}{9.3928} = \mathbf{0.7142} \; (71.42\%)$$

> **Common Code Bug to Avoid:** Never compute `IDCG@K` by merely sorting the $K$ documents that your retriever happened to return! If your retriever returned 4 irrelevant docs (`[0, 0, 0, 0]`) or missed the Grade-3 document, sorting only the returned items creates a fake denominator. Always compute `IDCG@K` from the **top $K$ grades across the entire ground-truth label set (`qrels`)** for that query.

---

## Block 3: Systematic RAG Pipeline Failure Debugging (Minutes 30–42)

When a user reports *"The AI gave the wrong answer for Query X"*, junior engineers immediately start tweaking the system prompt. Senior engineers treat RAG as a **5-stage diagnostic funnel** and isolate the exact broken layer in under 5 minutes:

```text
[Wrong Answer Reported]
          │
          ▼
Check 1: Does the true answer exist cleanly in the indexed chunks?
   ├── NO  ──► INGESTION / CHUNKING BUG
   │           (Table header decapitated, PDF OCR error, stale un-synced doc)
   │           Fix: DOM/AST chunking, table header cloning, CDC hash check.
   └── YES
          ▼
Check 2: Did the true chunk appear in Stage-1 Top-50 (Recall@50)?
   ├── NO  ──► STAGE-1 RETRIEVAL BUG
   │           (Exact SKU/hex code missed, pronoun coreference lost, ACL starvation)
   │           Fix: Add BM25 + RRF, Contextual/Late Chunking, Filterable HNSW.
   └── YES
          ▼
Check 3: Did Stage-2 Reranker push the chunk into Top-5 (NDCG@5)?
   ├── NO  ──► STAGE-2 RERANKING BUG
   │           (Chunk truncated at 256 tokens by reranker, domain vocabulary gap)
   │           Fix: Increase reranker max_length to 512, distill domain cross-encoder.
   └── YES
          ▼
Check 4: Was the chunk buried in the middle of 15 noisy chunks?
   ├── YES ──► CONTEXT PACKING BUG ("Lost in the Middle")
   │           Fix: Cut K from 15 to 5, apply Sigmoid threshold filter (prob >= 0.20),
   │           and place highest-scoring chunks at the start and end of context.
   └── NO
          ▼
Check 5: The exact fact is clearly visible in Top-5, yet LLM invented something else!
   └── YES ──► GENERATOR / FAITHFULNESS BUG (Hallucination / Parametric Leakage)
               Fix: Set temperature=0.0, enforce inline [Chunk_ID] citations,
               add explicit abstention rule ("If not in context, say I don't know").
```

### Diagnostic Mapping via the RAG Triad

| Context Relevance | Groundedness (Faithfulness) | Answer Relevance | Exact Root Cause |
| :---: | :---: | :---: | :--- |
| **LOW** | **HIGH** | **LOW** | **Retrieval Failure:** Retriever fetched junk; LLM honestly refused to answer ("I don't know"). Fix Stage 1/2 Search, NOT the prompt! |
| **HIGH** | **LOW** | **HIGH / LOW** | **Hallucination:** Retriever brought the exact right doc, but LLM ignored it and leaked outdated pre-training facts. Fix prompt grounding / lower temperature / upgrade generator. |
| **HIGH** | **HIGH** | **LOW** | **Synthesis / Prompt Drift:** Retriever got the right doc and LLM stayed faithful to it, but summarized the wrong paragraph instead of answering the specific question. |
| **LOW** | **LOW** | **HIGH** | **Silent Parametric Leakage (Most Dangerous!):** Retriever completely failed, yet the LLM answered anyway from pre-training memory. It looks right today, but will silently return outdated company data tomorrow! |

---

## Block 4: 2-Minute Architecture Math — QPS, Storage & Cost per 1,000 Queries (Minutes 42–55)

Every large-scale system design requires fast, clean back-of-the-envelope estimates. Memorize **one anchor constant** ($1\text{ Day} = 86,400\text{ sec} \approx 10^5\text{ sec}$) and practice this 3-part template.

### Scenario Baseline
* **Users:** **100,000 Daily Active Users (DAU)**, averaging **10 queries/user/day**.
* **Corpus:** **10 Million document chunks** embedded with a **1024-dim** model.
* **Prompt Shape per Query:** `500` system/instruction tokens + `2,000` retrieved context tokens (5 chunks $\times$ 400 tokens) + `500` query/history tokens = **`3,000` input tokens**; **`500` output tokens**.

---

### Calculation 1: Average QPS, Peak QPS & GPU Replica Sizing
1. **Daily Query Volume:**
   $$100,000\text{ users} \times 10\text{ queries} = \mathbf{1,000,000\text{ queries/day}} \; (30\text{M/month})$$
2. **Average QPS (Queries Per Second):**
   $$\text{Avg QPS} = \frac{1,000,000}{86,400\text{ sec}} \approx \mathbf{11.6\text{ QPS}} \quad (\text{Quick mental estimate: } 10^6 / 10^5 = 10\text{ QPS})$$
3. **Peak QPS (Enterprise Traffic Spike Multiplier = $4\times$):**  
   Enterprise traffic is not flat 24/7—80% of requests hit during core business hours (9 AM–6 PM) with sharp morning spikes:
   $$\text{Peak QPS} = 11.6 \times 4 \approx \mathbf{46\text{–}50\text{ QPS}}$$
4. **Reranker GPU Sizing:**  
   If 1 L4 GPU runs your distilled Stage-2 Cross-Encoder at `35 ms` per query ($\approx 28\text{ QPS}$ single-stream, or $\sim 55\text{ QPS}$ with dynamic micro-batching), deploying **2 active GPU replicas** (plus 1 failover replica for $N+1$ high availability) handles 50 Peak QPS comfortably with zero queue buildup.

---

### Calculation 2: Disk & RAM Storage Sizing (10M Chunks, 1024-Dim)
1. **Raw Text + Metadata on SSD Disk (~2 KB per chunk):**
   $$10,000,000 \times 2\text{ KB} = \mathbf{20\text{ GB SSD}}$$
2. **Uncompressed FP32 Vectors (Baseline Reference):**
   $$10\text{M} \times 1024 \times 4\text{ Bytes} = \mathbf{40.96\text{ GB}}$$
3. **Production In-RAM Footprint (INT8 Scalar Quantization + HNSW Links):**
   * **INT8 Vectors ($1\text{ Byte/dim}$):** $10\text{M} \times 1,024 \times 1\text{ B} = \mathbf{10.24\text{ GB}}$
   * **HNSW Graph Edges ($M=16 \implies \sim 170\text{ B/vector}$):** $10\text{M} \times 170\text{ B} = \mathbf{1.70\text{ GB}}$
   * **Total RAM (adding 30% OS/payload/BM25 buffer):** $(10.24 + 1.70) \times 1.30 \approx \mathbf{15.5\text{ GB RAM}}$  
   $\rightarrow$ Fits easily inside a single **32 GB RAM** node (or 64 GB total across a 2-node high-availability replica pair).

---

### Calculation 3: Unit Economics — Cost per 1,000 Queries (FinOps)
First, compute total token volume for **1,000 queries**:
* **Input Tokens per 1,000 Queries:** $1,000 \times 3,000 = \mathbf{3.0\text{ Million Input Tokens}}$
* **Output Tokens per 1,000 Queries:** $1,000 \times 500 = \mathbf{0.5\text{ Million Output Tokens}}$

Now plug in reference API pricing tiers (plus amortized `~$0.15` per 1,000 queries for self-hosted embedding + reranker GPUs):

| Serving Strategy | Reference Token Pricing (In / Out per 1M) | Cost per 1,000 Queries | Monthly Cost (at 30M Queries/Mo) |
| :--- | :---: | :---: | :---: |
| **1. All-Frontier Model** | $\$3.00 \text{ in } / \; \$15.00 \text{ out}$ | $(3 \times \$3) + (0.5 \times \$15) + \$0.15 = \mathbf{\$16.65}$ | **$\$499,500 / \text{mo}$** |
| **2. All-Fast / Distilled SLM** | $\$0.20 \text{ in } / \; \$0.80 \text{ out}$ | $(3 \times \$0.20) + (0.5 \times \$0.80) + \$0.15 = \mathbf{\$1.15}$ | **$\$34,500 / \text{mo}$** |
| **3. Hybrid Router (75% SLM + 25% Frontier)** | Blended Tier | $(0.75 \times \$1.15) + (0.25 \times \$16.65) = \mathbf{\$5.03}$ | **$\$150,750 / \text{mo}$** (Saves ~$\$349\text{k/mo!}$) |
| **4. Hybrid Router + 20% Semantic Cache** | 20% served at ~$\$0$ LLM cost | $\$5.03 \times 0.80 = \mathbf{\$4.02}$ | **$\$120,600 / \text{mo}$** |

> **Engineering Insight:** Look at the cost ratio—even in a RAG system where you send 6x more input tokens (`3,000`) than output tokens (`500`), **output tokens cost 4x to 5x more per token than input tokens**, making output generation nearly half your total LLM bill! Enforcing concise responses and routing 75% of routine lookups to a distilled SLM cuts cloud spend by **70–75%**.

---

## Block 5: Complete, Runnable Python Evaluator & FinOps Calculator (Minutes 55–60)

This self-contained Python script implements `Recall@K`, `MRR`, and `NDCG@K` (using full ground-truth `qrels` for exact `IDCG@K`) plus a reusable unit-economics calculator, verifying every hand calculation above:

```python
import math
from dataclasses import dataclass
from typing import Dict, List, Set


class OfflineRetrievalEvaluator:
    """
    Production-accurate offline evaluator for Stage-1 and Stage-2 RAG pipelines.
    Computes Recall@K, MRR, and NDCG@K (exponential gain with full-corpus IDCG).
    """

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
        return round(hits / len(relevant_ids), 4)

    @staticmethod
    def mean_reciprocal_rank(
        queries_retrieved: List[List[str]], queries_relevant: List[Set[str]]
    ) -> float:
        if not queries_retrieved:
            return 0.0
        rr_sum = 0.0
        for retrieved, relevant in zip(queries_retrieved, queries_relevant):
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    rr_sum += 1.0 / rank
                    break  # MRR stops at the FIRST relevant document
        return round(rr_sum / len(queries_retrieved), 4)

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int) -> float:
        """
        Args:
            retrieved_ids: Ordered list of returned document IDs.
            qrels: Full ground-truth relevance grades for this query {doc_id: grade (0..3)}.
            k: Cutoff rank.
        """
        # 1. Actual DCG@K over returned documents
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            rel = qrels.get(doc_id, 0)
            dcg += ((2 ** rel) - 1) / math.log2(rank + 1)

        # 2. Ideal DCG@K over the ENTIRE ground-truth qrels pool (never just the returned list!)
        ideal_grades = sorted(qrels.values(), reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_grades, start=1):
            idcg += ((2 ** rel) - 1) / math.log2(rank + 1)

        return round(dcg / idcg, 4) if idcg > 0.0 else 0.0


@dataclass
class RAGFinOpsEstimate:
    avg_qps: float
    peak_qps: float
    vector_db_ram_gb: float
    cost_per_1k_queries_usd: float
    monthly_cost_usd: float


def estimate_rag_capacity_and_cost(
    dau: int = 100_000,
    queries_per_user: int = 10,
    peak_multiplier: float = 4.0,
    num_chunks: int = 10_000_000,
    vector_dim: int = 1024,
    input_tokens_per_query: int = 3_000,
    output_tokens_per_query: int = 500,
    in_price_per_1m: float = 3.00,
    out_price_per_1m: float = 15.00,
    infra_cost_per_1k: float = 0.15,
) -> RAGFinOpsEstimate:
    daily_queries = dau * queries_per_user
    avg_qps = daily_queries / 86_400.0
    peak_qps = avg_qps * peak_multiplier

    # INT8 Quantized Vectors (1B/dim) + HNSW Edges (170B/vec) + 30% buffer
    raw_ram_bytes = num_chunks * (vector_dim * 1 + 170) * 1.30
    ram_gb = raw_ram_bytes / 1e9

    # Cost per 1,000 queries
    in_tokens_1k = (1_000 * input_tokens_per_query) / 1e6
    out_tokens_1k = (1_000 * output_tokens_per_query) / 1e6
    cost_1k = (in_tokens_1k * in_price_per_1m) + (out_tokens_1k * out_price_per_1m) + infra_cost_per_1k
    monthly_cost = (daily_queries * 30 / 1_000) * cost_1k

    return RAGFinOpsEstimate(
        avg_qps=round(avg_qps, 2),
        peak_qps=round(peak_qps, 2),
        vector_db_ram_gb=round(ram_gb, 2),
        cost_per_1k_queries_usd=round(cost_1k, 2),
        monthly_cost_usd=round(monthly_cost, 2),
    )


# --- Verification Demo (Matches Every Hand Calculation Above!) ---
if __name__ == "__main__":
    ev = OfflineRetrievalEvaluator()
    print("Recall@5 (Expected 0.75):", ev.recall_at_k(["D9", "D2", "D7", "D1", "D4"], {"D1", "D2", "D3", "D4"}, k=5))
    print(
        "MRR (Expected 0.5833):",
        ev.mean_reciprocal_rank(
            [["D1", "D9"], ["D8", "D2"], ["D7", "D8", "D9", "D4"]],
            [{"D1"}, {"D2"}, {"D4"}],
        ),
    )
    print(
        "NDCG@4 (Expected 0.7142):",
        ev.ndcg_at_k(["D1", "D2", "D3", "D4"], {"D1": 1, "D2": 3, "D3": 0, "D4": 2}, k=4),
    )
    print("Frontier FinOps Estimate:", estimate_rag_capacity_and_cost())
```
