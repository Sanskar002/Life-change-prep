# Days 9–10 Quick Reference Notes: RAG Triad, Retrieval Metrics, Debugging & FinOps Math

*Concise 5-minute engineering cheat sheet covering evaluation formulas, failure diagnosis, and capacity math.*

---

## 1. The RAG Triad & Failure Diagnosis Matrix

* **Context Relevance ($Q \leftrightarrow C$):** Did Retrieval fetch the right chunks without noise?
* **Groundedness / Faithfulness ($C \leftrightarrow A$):** Ratio of LLM atomic claims strictly backed by Context $C$ (detects hallucinations).
* **Answer Relevance ($Q \leftrightarrow A$):** Does the final response actually solve the user's question?

| Symptom Pattern | Root Cause | Immediate Fix |
| :--- | :--- | :--- |
| **Low Context Rel + High Faithfulness** | **Retrieval Failure** (LLM honestly said "I don't know") | Fix chunking, add BM25 + RRF, tune HNSW/RBAC filters. |
| **High Context Rel + Low Faithfulness** | **Hallucination / Parametric Leakage** | Lower temperature to `0.0`, enforce `[Chunk_ID]` citations, tighten grounding rules. |
| **Low Context Rel + Low Faithfulness + High Answer Rel** | **Silent Parametric Leakage** (LLM answered from pre-training memory!) | Add Stage-2 Sigmoid cutoff threshold (`prob >= 0.20`) to force abstention on bad retrieval. |
| **High `Recall@50` + Low `NDCG@5`** | **Reranker Failure** | Check reranker `max_length` truncation; fine-tune domain cross-encoder. |

---

## 2. Offline Retrieval Metrics Cheat Table

| Metric | Formula | What It Measures | Best Used For |
| :--- | :--- | :--- | :--- |
| **`Recall@K`** | $\frac{\text{Relevant in Top } K}{\text{Total Relevant in Corpus}}$ | Completeness (ignores order) | **Stage-1 Retrieval** (`Recall@50`) |
| **`MRR`** | $\frac{1}{|Q|} \sum \frac{1}{\text{rank}_{\text{first\_hit}}}$ | Position of the **1st** right answer only | Single-fact / exact-code lookups |
| **`NDCG@K`** | $\frac{\text{DCG@K}}{\text{IDCG@K}}$, where $\text{DCG} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}$ | **Graded relevance** (`0–3`) + **position discounting** | **Stage-2 Rerankers** (`NDCG@5`, `NDCG@10`) |

* **Key `NDCG` Discount Values:** $\log_2(2) = \mathbf{1.00}$, $\log_2(3) \approx \mathbf{1.59}$, $\log_2(4) = \mathbf{2.00}$, $\log_2(5) \approx \mathbf{2.32}$.
* **Critical `IDCG` Rule:** Always build `IDCG@K` by sorting the top-$K$ grades from the **entire ground-truth label pool (`qrels`)**, never just the returned list.

---

## 3. 2-Minute Capacity & FinOps Formulas

* **Daily Seconds Anchor:** $1\text{ Day} = 86,400\text{ sec} \approx \mathbf{10^5\text{ sec}}$ (mental shortcut).
* **QPS Formula:** $\text{Avg QPS} = \frac{\text{DAU} \times \text{Queries/User}}{86,400}$; $\text{Peak QPS} = \text{Avg QPS} \times \mathbf{4}$ (enterprise daytime spikes).
  * *Anchor Example:* $100\text{k DAU} \times 10\text{ q/day} = 1\text{M/day} \approx \mathbf{11.6\text{ Avg QPS}} \rightarrow \mathbf{\sim 46\text{–}50\text{ Peak QPS}}$.
* **Vector DB RAM (INT8 + HNSW, $M=16$):**
  $$\text{RAM} \approx N \times (d \times 1\text{ Byte} + 170\text{ Bytes}) \times 1.30$$
  * *Anchor Example ($10\text{M}$ chunks, $1024\text{d}$):* $10\text{M} \times (1024 + 170) \times 1.30 \approx \mathbf{15.5\text{ GB RAM}}$.
* **Cost per 1,000 Queries Formula:**
  $$\text{Cost}_{1\text{k}} = \left(\frac{1000 \times T_{\text{in}}}{10^6} \times P_{\text{in}}\right) + \left(\frac{1000 \times T_{\text{out}}}{10^6} \times P_{\text{out}}\right) + \text{Infra}_{1\text{k}}$$
  * *Anchor Example ($3,000\text{ in} / 500\text{ out}$):* $3\text{M in} + 0.5\text{M out}$ per 1,000 queries $\rightarrow \mathbf{\$16.65}$ (Frontier at $\$3/\$15$) vs. $\mathbf{\$1.15}$ (SLM at $\$0.20/\$0.80$) vs. $\mathbf{\$5.03}$ (75/25 Hybrid Router).

---

## 4. Important Questions (Evaluation, Debugging & Economics)

**Q1: Why is `Recall@50` used for Stage-1 Search while `NDCG@5` is used for Stage-2 Reranking?**  
*Stage-1 Retrieval only needs to ensure the relevant chunks survive into the candidate shortlist (order within the Top 50 does not matter as long as they are present—measured cleanly by `Recall@50`). Stage-2 Reranking decides which 5 chunks actually enter the LLM prompt and in what exact order—where graded relevance (`0–3`) and top-position discounting (`NDCG@5`) directly determine downstream answer quality.*

**Q2: How do you prevent Position Bias and Verbosity Bias when running an automated LLM-as-a-Judge pipeline?**  
*For **Position Bias**, run every pairwise comparison twice with swapped ordering `(A, B)` and `(B, A)` and discard inconsistent wins as ties. For **Verbosity Bias**, avoid open-ended preference prompts; instead, decompose answers into atomic claims (`Yes/No` groundedness checks) and grade against fixed 1-to-5 rubrics calibrated to a human-labeled golden set (targeting Cohen's Kappa $\kappa \ge 0.75$).*

**Q3: Your offline retrieval metrics show `Recall@50 = 0.94`, yet `NDCG@5` dropped to `0.42` and end-to-end Faithfulness is falling. Where is the bug?**  
*High `Recall@50` proves Ingestion and Stage-1 Search are working properly. The bottleneck sits at **Stage-2 Reranking or Context Packing**. Check three things immediately: (1) whether the Cross-Encoder is truncating long chunks (`max_length=256` cutting off answers near the end), (2) whether domain jargon requires a fine-tuned reranker, and (3) whether noisy low-confidence chunks are passing into the LLM without a Sigmoid threshold filter.*

**Q4: What is the single fastest architectural lever to cut RAG API spend by 70% without hurting answer quality?**  
*Deploy a **Hybrid Complexity Router + Prefix/Semantic Caching**. Because 70–80% of enterprise queries are straightforward factual lookups once Stage-2 Reranking hands over the exact 5 chunks, routing those queries to a fast distilled SLM (`~$1.15 / 1k queries`) and escalating only complex multi-document synthesis to a frontier model (`~$16.65 / 1k queries`) drops blended costs by ~70%—and adding a 20% semantic cache cuts another 20% off the top.*
