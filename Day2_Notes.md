# Day 2 Quick Revision Cheat Sheet: Practical Code Design, Late Chunking & Qdrant Internals

*Scan this 5-minute sheet before Low-Level Code Design (Atlassian/Intuit) and RAG System Design rounds.*

---

## 1. Late Chunking Tensor & Offset Checklist

* **Core Tensor Flow:**
  $$\text{Full Doc } (T \le 8192) \xrightarrow{\text{Bi-Encoder}} H \in \mathbb{R}^{T \times d} \xrightarrow{\text{Offset Alignment}} \text{MeanPool}(H[t_{\text{start}}:t_{\text{end}}]) \xrightarrow{\text{L2 Norm}} v_i \in \mathbb{R}^{d}$$
* **3 Bugs Interviewers Watch For:**
  1. **Subword Alignment Bug:** Slicing tokens by word count instead of using `return_offsets_mapping=True` to align character spans `(char_start, char_end)` to token indices `(tok_start, tok_end)`.
  2. **`[CLS]` / `[SEP]` Contamination:** Forgetting to skip special tokens where `offset == (0, 0)`, which leaks the global `[CLS]` token into Chunk 0's mean pool.
  3. **VRAM OOM & Non-Determinism:** Forgetting `@torch.no_grad()` (retains autograd graph across 8,192 tokens) or `model.eval()` (leaves Dropout active) during inference.

---

## 2. Low-Level Design (LLD): Idempotent Record Manager Rules

* **Why Separate `text_hash` from `acl_hash`?**
  * `text_hash = SHA256(breadcrumb + text)` $\rightarrow$ Mutation requires **GPU Embedding + Vector Upsert + Old UUID Delete**.
  * `acl_hash = SHA256(sorted(allowed_groups))` $\rightarrow$ If `text_hash` is unchanged and only `acl_hash` changed, call **`client.set_payload()`** (1 ms metadata update, **0 GPU compute**, **0 HNSW tombstones**!).
* **Deterministic Point IDs:**
  * Generate point IDs via `uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}::chunk_{idx}::{text_hash}")`.
* **Zero-Downtime Mutation Order:**
  1. **`upsert()`** new/modified chunk UUIDs first.
  2. **`set_payload()`** on ACL-only modified chunk UUIDs.
  3. **`delete()`** orphaned old chunk UUIDs last (prevents temporary missing-chunk gaps during concurrent user queries).
  4. **`commit_plan()`** to PostgreSQL Record Manager.

---

## 3. Production Qdrant Configs & Gotchas (Senior Signals)

| Parameter / Feature | Production Setting | Exact Technical Reason |
| :--- | :--- | :--- |
| **Pre-Ingestion Indexing** | Call `create_payload_index()` **before** `upsert()` | Filterable HNSW builds subgraph edges (`payload_m=16`) during graph construction. Indexing after upload forces a full HNSW rebuild. |
| **`full_scan_threshold`** | `6000` (**Kilobytes**, not vector count!) | In Qdrant, this is KB size. For 768-dim FP32 ($3\text{ KB/vec}$), `6000 KB` $\approx 2,000\text{ vectors}$, triggering SIMD flat scan below 2k matches. |
| **`is_tenant=True`** | Set in `KeywordIndexParams` on `allowed_groups` | Physically co-locates vectors sharing the same group/tenant on adjacent disk/RAM pages, eliminating random I/O page faults. |
| **Multi-Filter Intersection** | **ACORN (2-Hop Traversal)** | When combining multiple filters where pre-built subgraph bridges don't exist, ACORN expands HNSW search to 2-hop neighbors ($M^2$) on the fly. |
| **INT8 Scalar Quantization** | `ScalarType.INT8`, `always_ram=True`, `rescore=True` | Compresses vectors $4\times$ (FP32 $\rightarrow$ INT8) in RAM, oversamples $2\times$ during search, and rescores top candidates for $<1\%$ recall loss. |
| **Vacuum Compaction** | `deleted_threshold=0.20`, `vacuum_min_vector_number=1000` | Rebuilds segment HNSW graph in background once soft tombstones exceed 20%, preventing `hnsw_ef` candidate queue starvation. |

---

## 4. 60-Second Back-of-the-Envelope Capacity Math

Use these exact numbers when asked to size a **20-Million Vector Enterprise Index (768-dim)**:

1. **Raw Vector Size (FP32):** $20\text{M} \times 768 \times 4\text{ B} = \mathbf{61.44\text{ GB}}$ (or $\mathbf{30.72\text{ GB}}$ in FP16).
2. **HNSW Graph Link Overhead ($M=16 + \text{payload\_m}=16$):** Each point stores $\sim 32 \times 8\text{ B} \approx 256\text{ B}$ of graph edges $\rightarrow 20\text{M} \times 256\text{ B} \approx \mathbf{5.12\text{ GB}}$.
3. **With INT8 Scalar Quantization (`always_ram=True`):**
   * Quantized Vectors in RAM: $20\text{M} \times 768 \times 1\text{ B} = \mathbf{15.36\text{ GB}}$.
   * Total RAM Required (INT8 Vectors + HNSW Graph + Payload Bitmasks): **$\sim 22\text{–}24\text{ GB RAM}$** (fits comfortably on a single `r6i.xlarge` 32GB node, saving ~65% cloud memory cost while keeping p99 latency under 15 ms via `rescore=True`).
