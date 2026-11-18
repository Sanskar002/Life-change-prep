# Days 5–6 Complete Technical Study Guide: Two-Stage Retrieval Architecture, Cross-Encoder Reranking, ColBERT & <100ms Latency Engineering

**Scope:** Bi-Encoders vs. Cross-Encoders, Multi-Vector Late Interaction (ColBERT MaxSim), Latency SLA Optimization & Resilient Two-Stage Pipelines  
**Prerequisites:** None — every concept, equation, and code block is explained from first principles with toy numeric examples.  
**Reading Time:** 60 Minutes (100% self-contained)

---

## The Big Picture: Why Do Production RAG Systems Need Two Stages?

In Days 1–4, we built a **Hybrid Search Engine** (combining a Dense Bi-Encoder with BM25 Keyword Search and Reciprocal Rank Fusion). That engine can scan **10 million document chunks in ~12 milliseconds**.

So why not just hand the Top 5 chunks from that search directly to the LLM?  
Because Stage-1 search engines suffer from a fundamental **compression bottleneck**: to make search fast, a Bi-Encoder has to squish an entire 300-word paragraph into a **single list of 768 numbers** *before* it ever sees the user's question.

### The Everyday Analogy: Resume Keyword Filter vs. Live Technical Discussion
Imagine hiring an engineer from a pool of **10,000 applicants**:
* **Stage 1 — The Fast Shortlist (Bi-Encoder + BM25):** An automated script scans all 10,000 one-page resume summaries in 10 seconds and picks the **Top 50 candidates** who broadly match the skills. This stage is ultra-fast, but its summaries are blurry—it cannot tell the difference between someone who *"led a migration from X to Y"* and someone who *"assisted on Y before moving to X"*.
* **Stage 2 — The Deep Comparison (Cross-Encoder Reranker):** A panel sits down with **only those 50 shortlisted candidates** and compares every requirement of the project against every sentence of their experience side-by-side. Doing this for all 10,000 people is impossible, but doing it for **50 people** is fast and finds the **Top 5 exact matches**.

```text
[10,000,000 Document Chunks]
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Candidate Generation (Bi-Encoder + BM25 + RRF)  │
│  • Goal: High Recall@50 (Don't miss the right chunk!)    │
│  • Speed: ~10–15 ms (Vectors are pre-computed offline)   │
└──────────────────────────────────────────────────────────┘
             │  Outputs Top N = 50 Shortlisted Chunks
             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Precision Reranking (Cross-Encoder / ColBERT)   │
│  • Goal: High Precision / NDCG@5 (Put best 5 at the top) │
│  • Speed: ~25–40 ms (Scores 50 [Query, Chunk] pairs live)│
└──────────────────────────────────────────────────────────┘
             │  Outputs Top K = 5 Verified Chunks
             ▼
     [Generator LLM]  (Total Retrieval + Rerank Time: < 70 ms!)
```

---

## Block 1: Bi-Encoders vs. Cross-Encoders — How They Work Under the Hood (Minutes 0–15)

### 1.1 Architecture 1: The Bi-Encoder ("Two Separate Rooms")
In a **Bi-Encoder** (like `bge-base` or `jina-embeddings-v3`), the Query and the Document are processed in **two completely separate Transformer forward passes**:

```text
   [Query Q]                          [Document Chunk D]
  (Live at Query Time)               (Pre-computed Offline!)
       │                                      │
       ▼                                      ▼
┌──────────────┐                       ┌──────────────┐
│ Transformer  │                       │ Transformer  │
│  (No look at │                       │  (No look at │
│   Document)  │                       │    Query!)   │
└──────┬───────┘                       └──────┬───────┘
       │ Mean Pool                            │ Mean Pool
       ▼                                      ▼
  Vector v_Q (768 floats)                Vector v_D (768 floats)
       │                                      │
       └──────────────► Dot Product ◄─────────┘
                      Score = v_Q · v_D
```

#### Why Bi-Encoders Lose Fine-Grained Precision (Concrete Failure Example)
Suppose a user asks:
* **Query:** `"What is the current limit at -20C?"`

And your database has two very similar chunks:
* **Chunk A (Wrong Answer):** *"At **+25C**, the controller allows a current limit of **150A**, while the cooling fan remains off down to **-20C**."*
* **Chunk B (Right Answer):** *"While normal operation supports 150A, at **-20C** the current limit drops strictly to **45A**."*

**Why the Bi-Encoder gets tricked:**
1. Both Chunk A and Chunk B contain the exact same keywords: `"current"`, `"limit"`, `"150A"`, and `"-20C"`.
2. When the Bi-Encoder compresses Chunk A into a single 768-number vector offline, it **averages (mean-pools)** all the words together. It loses the grammatical link showing that `"-20C"` was talking about the *cooling fan*, not the *current limit*!
3. As a result, Chunk A and Chunk B get almost identical Bi-Encoder cosine similarity scores (`0.862` vs `0.859`), and the wrong chunk can easily land at Rank #1.

---

### 1.2 Architecture 2: The Cross-Encoder ("Same Room, Word-by-Word Inspection")
A **Cross-Encoder** (like `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) does **not** create vectors to store in a database. Instead, it glues the Query and Document together into **one single sequence** separated by a `[SEP]` token and feeds them into a Transformer **at the same time**:

$$\text{Input} = \texttt{[CLS]} \; \text{Query Words} \; \texttt{[SEP]} \; \text{Document Words} \; \texttt{[SEP]}$$

```text
  [CLS]  What  is  limit  at  -20C  [SEP]  At  -20C  limit  drops  to  45A  [SEP]
    │     │    │     │    │     │           │    │     │      │     │   │
    └─────┴────┴─────┴────┴─────┴───────────┴────┴─────┴──────┴─────┴───┘
               ▲                                           ▲
               └── Every Query word directly attends to ───┘
                   every Document word across ALL layers!
                                   │
                                   ▼
                     Linear Head on [CLS] Token
                                   │
                                   ▼
                   Single Raw Score (Logit z = +4.85)
```

* **Why the Cross-Encoder solves the Chunk A vs. Chunk B problem:**  
  Inside the Transformer's self-attention layers, the query token `"-20C"` looks directly at the document tokens around `"-20C"`. In Chunk B, it sees `"-20C"` is directly attached to `"limit drops to 45A"`. In Chunk A, it sees `"-20C"` is attached to `"cooling fan"`.
* **Result:** The Cross-Encoder gives Chunk B a massive score of **`+5.20`** and drops Chunk A to **`-1.40`**.

---

### 1.3 Converting Cross-Encoder Logits into Probabilities (The Sigmoid Rule)
Unlike Bi-Encoders (which output cosine similarity between `-1.0` and `+1.0`), Cross-Encoders output an unbounded raw number called a **logit** $z \in (-\infty, +\infty)$, such as `-6.5` or `+4.8`.

To use this score as a **relevance filter** (for example, throwing away any chunk with less than `25%` relevance so the LLM doesn't hallucinate on irrelevant text), convert the logit $z$ into a probability between `0.0` and `1.0` using the **Sigmoid Function**:

$$P(\text{relevant}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

* **Quick Mental Reference Table for Logits:**
  * Logit $z = -4.0 \longrightarrow \sigma(-4.0) \approx \mathbf{0.018}$ (**1.8% relevant** — Discard!)
  * Logit $z = 0.0 \longrightarrow \sigma(0.0) = \mathbf{0.500}$ (**50.0% relevant** — Borderline)
  * Logit $z = +2.0 \longrightarrow \sigma(+2.0) \approx \mathbf{0.881}$ (**88.1% relevant** — Keep!)
  * Logit $z = +4.0 \longrightarrow \sigma(+4.0) \approx \mathbf{0.982}$ (**98.2% relevant** — Strong Match!)

---

## Block 2: The Best of Both Worlds — ColBERT & Late Interaction (`MaxSim`) (Minutes 15–30)

### 2.1 The Question: Can We Get Cross-Encoder Accuracy at Bi-Encoder Speed?
Look at the trade-off we just saw:
* **Bi-Encoder:** Fast (pre-computes 1 vector per chunk offline), but blurry (mean-pooling squishes all words together).
* **Cross-Encoder:** Sharp (compares every word to every word), but slow (cannot pre-compute anything offline; must run a heavy neural network on 50 full paragraphs at query time).

**ColBERT (Contextualized Late Interaction over BERT)** bridges this gap using a clever middle ground:
1. **Like a Bi-Encoder:** It encodes the Document **offline** and the Query **at query time** separately.
2. **Unlike a Bi-Encoder:** It **deletes the Mean-Pooling step!** Instead of squishing a 100-word chunk into 1 vector, ColBERT saves **1 small vector for every single token** in the chunk (typically shrunk to $d = 128$ numbers per token).

```text
Bi-Encoder:     100-token chunk ──► Mean Pool ──► 1 Vector   (Shape: [1, 768])
ColBERT:        100-token chunk ──► Keep All! ──► 100 Vectors (Shape: [100, 128])
```

---

### 2.2 How ColBERT Scores Chunks in Microseconds: The `MaxSim` Operator
At query time, suppose your Query $Q$ has **2 tokens** (`"45A"`, `"-20C"`) and a candidate Document $D$ has **4 tokens** (`"limit"`, `"is"`, `"45A"`, `"-20C"`).

How do you compare a 2-token query against a 4-token document? Using **Maximum Similarity (`MaxSim`)**:
1. For **Query Token 1** (`"45A"`), compute its dot product against all 4 Document tokens, and keep **only the single highest score (the Max)**.
2. For **Query Token 2** (`"-20C"`), compute its dot product against all 4 Document tokens, and keep **only the single highest score (the Max)**.
3. **Add those maximums together!**

$$\text{Score}_{\text{MaxSim}}(Q, D) = \sum_{i=1}^{N_q} \max_{j=1 \dots N_d} \left( q_i \cdot d_j^T \right)$$

#### Step-by-Step Toy Numeric Example (Verify by Hand!)
Look at the dot-product similarity table between our 2 Query tokens (rows) and 4 Document tokens (columns):

| Query Token $\backslash$ Doc Token | $d_1$: `"limit"` | $d_2$: `"is"` | $d_3$: `"45A"` | $d_4$: `"-20C"` | **Row Maximum ($\max_j$)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$q_1$: `"45A"`** | `0.15` | `0.02` | **`0.96`** | `0.10` | **`0.96`** (Matched $d_3$) |
| **$q_2$: `"-20C"`** | `0.12` | `0.01` | `0.08` | **`0.94`** | **`0.94`** (Matched $d_4$) |
| **Total `MaxSim` Score** | | | | | **$0.96 + 0.94 = \mathbf{1.90}$** |

* **Why is this called "Late Interaction"?**  
  In a Cross-Encoder, Query and Document tokens interact *early* (inside all 12 or 24 Transformer layers). In ColBERT, Query and Document tokens only interact at the **very last step** via a simple matrix multiplication ($Q \cdot D^T$) and row-max sum.
* **Speed Advantage:** Because the Document token vectors were already computed offline, scoring 50 candidate documents with ColBERT `MaxSim` at query time requires **zero Transformer layers on the documents**—just a tiny matrix multiplication that finishes in **<2 milliseconds**!

---

### 2.3 Comparing All Three Retrieval Paradigms

| Feature | Bi-Encoder (Single Vector) | ColBERT (Multi-Vector Late Interaction) | Cross-Encoder (Full Interaction) |
| :--- | :--- | :--- | :--- |
| **Offline Pre-computation?** | **Yes** (1 vector per chunk) | **Yes** (1 vector per *token* in chunk) | **No** (Must run live on `[Q, D]`) |
| **Query-Time Neural Net Work** | 1 pass on short Query $Q$ only | 1 pass on short Query $Q$ only | **$N$ passes on long `[Q + D]` pairs** |
| **Latency for 50 Candidates** | **< 1 ms** (Dot product) | **1 – 3 ms** (`MaxSim` matrix multiply) | **20 – 80 ms** (Full Transformer batch) |
| **Storage Footprint** | **1x** (Baseline) | **10x – 15x** (Even with 2-bit residual compression) | **0 Bytes** (Uses raw text from Stage 1) |
| **Precision (NDCG@10)** | Baseline (Good) | Very High (Near Cross-Encoder) | **Highest (Gold Standard)** |

> **Production Hybrid Pattern:** Because storing ColBERT token vectors for 50 million chunks takes $10\times$ more storage, many production teams use a **Bi-Encoder + BM25** to fetch the Top 50 chunks from disk/RAM, and then use a **Cross-Encoder** (or **ColBERT**) to rerank only those 50 chunks in memory!

---

## Block 3: Engineering the `<100ms` Latency SLA (Minutes 30–45)

If you take an off-the-shelf Cross-Encoder (like `bge-reranker-large`) and pass it 50 chunks of 512 tokens in default PyTorch, it will take **350 to 600 milliseconds**—blowing past your **100ms SLA**!

Here is how we budget every millisecond and apply **5 engineering optimizations** to cut reranking latency from **400ms down to ~30ms**.

### 3.1 The 100ms End-to-End Latency Budget

```text
Total Allowed Retrieval + Reranking SLA: 100 ms
├── 1. Query Bi-Encoder Embedding (FP16 ONNX/GPU):      8 ms
├── 2. Stage-1 Hybrid Search (Qdrant HNSW+BM25+RRF):   12 ms
├── 3. Internal gRPC Serialization & Hop:               5 ms
├── 4. Stage-2 Cross-Encoder Reranking (50 chunks):    35 ms  <-- The Primary Bottleneck!
└── 5. P99 Jitter & Safety Buffer:                     10 ms
-------------------------------------------------------------
Total P99 Execution Time:                              70 ms (30 ms under budget!)
```

---

### 3.2 The 5 Engineering Levers to Cut Reranker Latency from 400ms to 30ms

#### Optimization 1: Length-Sorted Micro-Batching ("Dynamic Padding") — *Saves ~60% Compute!*
* **The Naive Padding Trap:**  
  Suppose you pass a batch of 50 `[Query, Chunk]` pairs to the GPU.
  * 48 of those pairs are short (**120 tokens**).
  * Just **2 pairs** are long tables (**512 tokens**).
  Because GPU tensors must be rectangular, standard batching pads **all 50 pairs out to 512 tokens** by filling the empty space with `[PAD]` zeros!
* **The Quadratic Attention Penalty ($O(L^2)$):**  
  Transformer self-attention cost grows with the **square of sequence length ($L^2$)**. Comparing a 512-token padded sequence to a 128-token sequence:
  $$\left(\frac{512}{128}\right)^2 = 4^2 = \mathbf{16\times\text{ more attention math per chunk!}}$$
  Those 2 long chunks force the GPU to waste $16\times$ more compute on the 48 short chunks!
* **The Fix (Length-Sorted Micro-Batching):**
  1. Sort the 50 candidate pairs by token length **shortest to longest** before tokenizing.
  2. Split them into 2 or 3 micro-batches (e.g., Batch 1 has the 35 shortest chunks padded only to `130 tokens`; Batch 2 has the 15 longer chunks padded to `380 tokens`).
  * **Result:** Total GPU FLOPs drop by **55–70%** with **zero change in model output**!

```text
WITHOUT LENGTH SORTING (1 Giant Batch — All 4 items padded to longest = 512 tokens):
Item 1 (100 tok): [████████░░░░░░░░░░░░░░░░░░░░░░░░]  <-- 412 wasted [PAD] tokens!
Item 2 (512 tok): [████████████████████████████████]
Item 3 (110 tok): [█████████░░░░░░░░░░░░░░░░░░░░░░░]  <-- 402 wasted [PAD] tokens!
Item 4 (120 tok): [██████████░░░░░░░░░░░░░░░░░░░░░░]  <-- 392 wasted [PAD] tokens!

WITH LENGTH SORTING + MICRO-BATCHING:
Micro-Batch A (Items 1, 3, 4 — Padded only to 120 tokens!):
Item 1 (100 tok): [████████░░]  <-- 16x faster attention!
Item 3 (110 tok): [█████████░]
Item 4 (120 tok): [██████████]
Micro-Batch B (Item 2 — Padded to 512 tokens):
Item 2 (512 tok): [████████████████████████████████]
```

#### Optimization 2: Choosing the Right Model Size (Distilled 6-Layer vs. 24-Layer)
You do not need a 560-million-parameter model (`bge-reranker-large`, 24 layers) for a `<100ms` SLA:
* **24-Layer Large Reranker (560M params):** ~120 ms for 50 pairs on an L4/T4 GPU.
* **12-Layer Base Reranker (110M–278M params):** ~35 ms for 50 pairs.
* **6-Layer Distilled MiniLM (`ms-marco-MiniLM-L-6-v2`, 22M params):** **~12 ms** for 50 pairs, while retaining ~96% of the ranking quality!

#### Optimization 3: Graph Compilation (ONNX Runtime / TensorRT + FP16 + FlashAttention)
Running a model in standard PyTorch (`FP32`) launches hundreds of separate tiny GPU kernels (one for Multiply, one for Add, one for LayerNorm, one for GELU), spending more time waiting on CPU-to-GPU communication than doing math.
* **Exporting to TensorRT / ONNX Runtime with FP16:** Fuses multiple layers into single GPU kernels and uses Tensor Cores (16-bit floats), cutting latency in half (**$2\times\text{ to }3\times$ speedup**) with zero accuracy loss.

#### Optimization 4: Score-Gap Adaptive Candidate Cutoff
Why always rerank 50 chunks if Stage 1 already found an obvious winner?
* If Stage-1 RRF shows that the Top 15 chunks matched **both** Dense and BM25, while chunks 16–50 only had weak single-list scores, dynamically cap the Stage-2 rerank pool at **`N = 20` instead of `50`**, cutting rerank latency by another 60%.

#### Optimization 5: Strict Timeout Circuit Breaker (Graceful Degradation)
What if the Reranker GPU spikes to 120ms due to a traffic burst?
* Wrap the Stage-2 Reranker call in an **80ms async timeout circuit breaker**. If Stage 2 does not finish within 80ms, **immediately return the Top 5 chunks from Stage-1 Hybrid RRF**! The user still gets a good answer within the 100ms SLA instead of an error screen.

---

## Block 4: Complete, Runnable Python/PyTorch Code Implementations (Minutes 45–60)

### 4.1 Implementation 1: Production Cross-Encoder with Length-Sorted Micro-Batching & Sigmoid Filter

```python
import time
import math
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ProductionCrossEncoderReranker:
    """
    Latency-optimized Cross-Encoder Reranker implementing:
    1. Length-sorted micro-batching (eliminates quadratic O(L^2) padding waste)
    2. FP16 mixed-precision inference + torch.inference_mode()
    3. Sigmoid probability calibration + minimum relevance score gating
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 384,
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Enable FP16 automatically when running on a CUDA GPU for 2x Tensor Core speedup
        if self.device.startswith("cuda"):
            self.model.half()

    @staticmethod
    def sigmoid(x: float) -> float:
        """Converts unbounded cross-encoder logit z into a calibrated probability in (0, 1)."""
        return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        micro_batch_size: int = 16,
        min_prob_threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Reranks Stage-1 candidates by sorting lengths into tight micro-batches first.
        """
        if not candidates:
            return []

        # Step 1: Tag each candidate with its original index and sort by character length
        # Sorting ensures each micro-batch only pads to the longest item in THAT micro-batch!
        indexed_candidates = list(enumerate(candidates))
        indexed_candidates.sort(key=lambda item: len(item[1]["text"]))

        logits_by_original_idx: Dict[int, float] = {}

        # Step 2: Process in length-grouped micro-batches with dynamic padding="longest"
        for start_idx in range(0, len(indexed_candidates), micro_batch_size):
            batch_slice = indexed_candidates[start_idx : start_idx + micro_batch_size]
            batch_queries = [query] * len(batch_slice)
            batch_docs = [item[1]["text"] for item in batch_slice]

            encoded = self.tokenizer(
                batch_queries,
                batch_docs,
                padding="longest",       # Pads ONLY to the longest sequence in this micro-batch!
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Forward pass through Cross-Encoder -> outputs shape: [batch_size, 1]
            batch_logits = self.model(**encoded).logits.squeeze(-1).cpu().tolist()
            if isinstance(batch_logits, float):
                batch_logits = [batch_logits]

            for (orig_idx, _), logit in zip(batch_slice, batch_logits):
                logits_by_original_idx[orig_idx] = float(logit)

        # Step 3: Calibrate scores via Sigmoid, filter out low-confidence noise, and sort Top-K
        scored_results: List[Dict[str, Any]] = []
        for orig_idx, cand in enumerate(candidates):
            raw_logit = logits_by_original_idx[orig_idx]
            prob = self.sigmoid(raw_logit)

            if prob >= min_prob_threshold:
                enriched = dict(cand)
                enriched["rerank_logit"] = round(raw_logit, 4)
                enriched["rerank_prob"] = round(prob, 4)
                enriched["stage1_rank"] = orig_idx + 1
                scored_results.append(enriched)

        scored_results.sort(key=lambda x: x["rerank_logit"], reverse=True)
        return scored_results[:top_k]
```

---

### 4.2 Implementation 2: ColBERT `MaxSim` Late Interaction Scorer in Pure NumPy (With Circuit Breaker)

This self-contained script runs immediately without downloading external weights. It demonstrates **exact ColBERT `MaxSim` tensor math** and a **production `<100ms` SLA Timeout Circuit Breaker**:

```python
import time
from typing import List, Dict, Any
import numpy as np


class ColBERTMaxSimReranker:
    """
    Implements ColBERT Late Interaction scoring:
    Score(Q, D) = sum_{i in Q} max_{j in D} (q_i · d_j^T)
    """

    @staticmethod
    def compute_maxsim_batch(
        query_token_vecs: np.ndarray,
        doc_token_vecs_list: List[np.ndarray],
    ) -> List[float]:
        """
        Args:
            query_token_vecs: L2-normalized array of shape [N_q, dim]
            doc_token_vecs_list: List of L2-normalized arrays, each of shape [N_d, dim]
        Returns:
            List of MaxSim scores for each document.
        """
        scores: List[float] = []
        for doc_vecs in doc_token_vecs_list:
            # 1. Compute all-to-all token similarity matrix S of shape [N_q, N_d]
            sim_matrix = query_token_vecs @ doc_vecs.T

            # 2. For each query token (row), find the maximum similarity across all doc tokens
            max_per_query_token = np.max(sim_matrix, axis=1)  # Shape: [N_q]

            # 3. Sum the maximums across all query tokens
            total_maxsim = float(np.sum(max_per_query_token))
            scores.append(round(total_maxsim, 4))

        return scores


def execute_two_stage_pipeline_with_sla_guard(
    stage1_rrf_candidates: List[Dict[str, Any]],
    query_token_vecs: np.ndarray,
    sla_budget_ms: float = 80.0,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Wraps Stage-2 Reranking in a strict latency SLA guard.
    If Stage 1 already consumed too much time or Stage 2 exceeds budget,
    gracefully falls back to Stage-1 RRF ordering.
    """
    start_time = time.perf_counter()

    try:
        doc_matrices = [cand["token_vecs"] for cand in stage1_rrf_candidates]
        maxsim_scores = ColBERTMaxSimReranker.compute_maxsim_batch(
            query_token_vecs, doc_matrices
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if elapsed_ms > sla_budget_ms:
            raise TimeoutError(f"Stage-2 exceeded SLA budget ({elapsed_ms:.1f}ms > {sla_budget_ms}ms)")

        reranked = []
        for cand, score in zip(stage1_rrf_candidates, maxsim_scores):
            item = {k: v for k, v in cand.items() if k != "token_vecs"}
            item["maxsim_score"] = score
            reranked.append(item)

        reranked.sort(key=lambda x: x["maxsim_score"], reverse=True)
        return {
            "fallback_triggered": False,
            "latency_ms": round(elapsed_ms, 3),
            "results": reranked[:top_k],
        }

    except TimeoutError as err:
        # Graceful degradation: return Stage-1 RRF order so user request never fails!
        fallback_results = [
            {k: v for k, v in cand.items() if k != "token_vecs"}
            for cand in stage1_rrf_candidates[:top_k]
        ]
        return {
            "fallback_triggered": True,
            "reason": str(err),
            "results": fallback_results,
        }


# --- Verification Demo (Matches Our Exact Hand-Calculated Toy Example!) ---
if __name__ == "__main__":
    # Query has 2 tokens: q1="45A", q2="-20C" (3D unit vectors for demonstration)
    q_vecs = np.array([
        [1.0, 0.0, 0.0],  # q1: "45A"
        [0.0, 1.0, 0.0],  # q2: "-20C"
    ], dtype=np.float32)

    stage1_pool = [
        {
            "doc_id": "CHUNK_A_DISTRACTOR",
            "stage1_rrf_rank": 1,  # Ranked #1 by Stage 1 by mistake!
            "text": "At +25C limit is 150A; fan stays off at -20C.",
            "token_vecs": np.array([
                [0.2, 0.1, 0.9],
                [0.1, 0.95, 0.1],  # Has "-20C", but missing "45A"!
            ], dtype=np.float32),
        },
        {
            "doc_id": "CHUNK_B_TRUE_ANSWER",
            "stage1_rrf_rank": 2,  # Ranked #2 in Stage 1, promoted to #1 by Stage 2!
            "text": "At -20C the current limit drops strictly to 45A.",
            "token_vecs": np.array([
                [0.1, 0.1, 0.9],
                [0.96, 0.05, 0.1], # Matches "45A" (0.96)
                [0.05, 0.94, 0.1], # Matches "-20C" (0.94) -> MaxSim = 1.90!
            ], dtype=np.float32),
        },
    ]

    output = execute_two_stage_pipeline_with_sla_guard(stage1_pool, q_vecs, top_k=2)
    print("Two-Stage Pipeline Output:", output)
```
