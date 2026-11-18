# Days 5–6 Quick Reference Notes: Two-Stage Retrieval, Cross-Encoders, ColBERT & <100ms SLAs

*Concise engineering reference sheet covering two-stage architecture, equations, latency budgets, and important questions.*

---

## 1. The 3 Architectures at a Glance

* **Bi-Encoder (Stage 1 — Candidate Generation):**  
  Encodes Query $Q$ and Document $D$ **separately** into 1 vector each ($v_Q, v_D$) via mean-pooling. Document vectors are pre-computed offline. Ultra-fast (**~10 ms** for 10M docs), but loses fine-grained word relationships due to mean-pooling compression.
* **Cross-Encoder (Stage 2 — Precision Reranking):**  
  Feeds `[CLS] Query [SEP] Document [SEP]` **together** through all Transformer layers so every query word attends to every document word. Highest accuracy, but **cannot pre-compute document vectors**—only run on the shortlisted Top $N = 25\text{–}50$ candidates (**~25–35 ms**).
* **ColBERT (Multi-Vector Late Interaction):**  
  Encodes $Q$ and $D$ separately like a Bi-Encoder, but **keeps 1 small vector per token** instead of mean-pooling. Scores pairs in **1–3 ms** using the `MaxSim` operator.

---

## 2. Core Equations to Remember

### A. Cross-Encoder Sigmoid Calibration (Logit $\rightarrow$ Probability)
$$P(\text{relevant} \mid Q, D) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
* Converts unbounded Cross-Encoder raw logits ($z \in (-\infty, +\infty)$) into `[0, 1]` probabilities so you can drop irrelevant chunks (e.g., $\sigma(z) < 0.15$) before calling the LLM.

### B. ColBERT `MaxSim` Late Interaction Formula
$$\text{Score}_{\text{MaxSim}}(Q, D) = \sum_{i=1}^{N_q} \max_{j=1 \dots N_d} \left( q_i \cdot d_j^T \right)$$
* For each query token $i$, find the single highest dot-product match across all document tokens $j$, then sum those best matches across all query tokens.

### C. Quadratic Self-Attention Padding Penalty
$$\text{Attention Compute} \propto B \times L_{\text{max}}^2$$
* Padding a batch of 128-token chunks out to $L_{\text{max}} = 512$ because of one long outlier chunk increases self-attention compute by $(512 / 128)^2 = \mathbf{16\times}$.

---

## 3. The 100ms Latency Budget & 5 Speed Optimizations

| Pipeline Stage / Optimization | Latency / Impact | Engineering Mechanism |
| :--- | :--- | :--- |
| **1. Query Bi-Encoder Pass** | **~8 ms** | Single short sequence ($<32$ tokens) in FP16 on GPU/ONNX. |
| **2. Stage-1 Hybrid Search (Top 50)** | **~12 ms** | Parallel Qdrant HNSW (INT8/Binary) + Sparse BM25 + server-side RRF. |
| **3. Length-Sorted Micro-Batching** | **Cuts Rerank Time ~60%** | Sort the 50 candidates by token length and tokenize with `padding="longest"` per micro-batch to avoid $O(L^2)$ pad waste. |
| **4. Distilled 6/12-Layer + FP16 ONNX** | **~25–35 ms** (for 50 pairs) | Use a 6- or 12-layer distilled Cross-Encoder compiled with TensorRT/ONNX FP16 instead of 24-layer FP32 PyTorch eager mode. |
| **5. SLA Timeout Circuit Breaker** | **Guarantees <100ms P99** | Wrap Stage 2 in an 80ms timeout; if GPU queue spikes, fall back to Stage-1 RRF Top-5 immediately. |

---

## 4. Important Questions (Architecture & Systems Deep-Dive)

**Q1: Why does a Bi-Encoder struggle with negations and multi-entity comparisons compared to a Cross-Encoder?**  
*A Bi-Encoder compresses an entire 300-word chunk into a single 768-dim vector offline without knowing what the user will ask. Mean-pooling averages token representations together, blurring which specific value belongs to which condition (e.g., mixing up specs at `-20C` vs `+25C` in the same paragraph). A Cross-Encoder concatenates `[CLS] Q [SEP] D` and runs full token-to-token cross-attention across every layer, directly binding query conditions to the exact matching words in the document.*

**Q2: Your Cross-Encoder Reranker takes 420ms at p99 for 50 chunks, violating your 100ms SLA. How do you bring it under 35ms without losing accuracy?**  
*Apply four concrete fixes:*  
*1. **Length-Sorted Dynamic Padding:** Sort the 50 candidates by token length into micro-batches of 16 with `padding="longest"` so short 120-token chunks aren't padded to 512 tokens ($16\times$ attention savings).*  
*2. **Model Distillation:** Replace a 24-layer large reranker (560M params) with a distilled 6- or 12-layer cross-encoder (like `MiniLM-L-6-v2` or a custom Margin-MSE distilled student).*  
*3. **Kernel Fusion & FP16:** Export from PyTorch eager mode to **TensorRT / ONNX Runtime in FP16** with FlashAttention.*  
*4. **Adaptive Candidate Pruning:** Only rerank the Top 25–30 candidates when Stage-1 RRF consensus scores show a sharp confidence drop-off.*

**Q3: How does ColBERT achieve near Cross-Encoder accuracy while executing 10x–20x faster at query time?**  
*Cross-Encoders are slow because they must run $N=50$ full document sequences through 12–24 Transformer layers live at query time. ColBERT shifts the heavy document Transformer pass **offline** by pre-computing and storing one compressed vector (e.g., 128d) per document token. At query time, ColBERT only runs the Transformer on the short Query (~10 tokens), and then computes Late Interaction (`MaxSim`) via a tiny matrix multiplication between query token vectors and pre-computed document token vectors in 1–3 ms.*

**Q4: Why should you apply a Sigmoid threshold filter ($\sigma(z) \ge \tau$) after Cross-Encoder reranking before passing chunks to the LLM?**  
*RRF and Top-$K$ sorting are strictly relative—even if the user asks an out-of-domain question where **all 50 retrieved chunks are completely irrelevant**, sorting still hands the "top 5 least-bad" garbage chunks to the LLM, triggering hallucinations or distraction. Passing Cross-Encoder logits through $\sigma(z) = 1 / (1 + e^{-z})$ produces an absolute relevance probability; dropping chunks below $\tau = 0.15\text{–}0.25$ protects the LLM context window and triggers a clean "I don't have enough information" response when appropriate.*
