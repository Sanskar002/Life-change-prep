# Day 2 Complete Self-Contained Study Guide: Hands-On Implementation, Low-Level Code Design & Production Vector Engineering

**Target Role Level:** Senior / Lead AI Engineer (₹55L–₹95L+ TC at Atlassian, Intuit, Microsoft, Adobe, SAP Labs, Target)  
**Study Time:** 50 Minutes (100% self-contained; runnable Python/PyTorch code + production system internals)  
**Focus Shift (Updated Syllabus):** Instead of academic neural-network coding, Day 2 focuses on the exact **Low-Level Code Design (LLD)**, **Token-Span Pooling**, **Idempotent State Diffing**, and **Production Qdrant RBAC Engineering** tested in 60-minute practical coding and system design rounds.

---

## Block 1: Hands-On Late Chunking Span Pooling in Python/PyTorch (Minutes 0–15)

### 1.1 Why Interviewers Ask About Span Pooling
Because your resume highlights **1 year of encoder/reranker distillation at Samsung** and **4 years of enterprise RAG at Volkswagen**, interviewers will test if you know how to bridge long-context encoders (like `jina-embeddings-v3` or `bge-m3`, supporting 8,192 tokens) with chunk-level vector indexing.

In **Naive Chunking**, you slice text *before* passing it to the tokenizer and model:
$$\text{Text} \xrightarrow{\text{split}} [\text{Chunk}_1, \text{Chunk}_2] \xrightarrow{\text{Encoder}} [v_1, v_2]$$
In **Late Chunking**, you run the **entire document** through the Transformer layers first so every token attends to the full document context via bidirectional self-attention, and then mean-pool the output token embeddings across character/token span boundaries:
$$\text{Full Document} \xrightarrow{\text{Transformer}} H \in \mathbb{R}^{T \times d} \xrightarrow{\text{Span Mean Pool}} [v_1, v_2, \dots, v_k]$$

---

### 1.2 The Production Implementation Trap: Character Offsets vs. Subword Tokens
A common interview bug is assuming 1 word = 1 token when mapping text chunks to tensor indices. Modern tokenizers (BPE / WordPiece) split words into subwords and insert special tokens (`[CLS]`, `[SEP]`, `<s>`).

* **How to solve it cleanly in code:** Use Hugging Face's fast Rust tokenizer option `return_offsets_mapping=True`. This returns a list of `(char_start, char_end)` tuples for every single token $t \in [0, T-1]$, allowing you to map character-level chunk boundaries to exact token slice indices $[t_{\text{start}} : t_{\text{end}}]$.

---

### 1.3 Complete, Runnable Code: Late Chunking with Offset Alignment & Mask-Safe Pooling

```python
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class LateChunkingEmbedder:
    """
    Production-grade Late Chunking implementation.
    Passes the full document through a long-context bi-encoder once, aligns character
    chunk boundaries to subword token offsets, and applies L2-normalized span mean pooling.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        self.device = device
        # Fast Rust tokenizer is required for return_offsets_mapping=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def get_char_chunk_spans(
        document: str, chunk_char_size: int = 300, overlap_chars: int = 40
    ) -> List[Tuple[int, int, str]]:
        """
        Splits document into character spans (start_char, end_char, chunk_text)
        respecting sentence/paragraph boundaries where possible.
        """
        spans: List[Tuple[int, int, str]] = []
        doc_len = len(document)
        start = 0

        while start < doc_len:
            end = min(start + chunk_char_size, doc_len)
            # Snap to nearest period or newline if not at end of document
            if end < doc_len:
                snap_Boundary = max(
                    document.rfind(". ", start, end),
                    document.rfind("\n", start, end),
                )
                if snap_Boundary != -1 and snap_Boundary > start + (chunk_char_size // 2):
                    end = snap_Boundary + 1

            chunk_text = document[start:end].strip()
            if chunk_text:
                spans.append((start, end, chunk_text))

            if end >= doc_len:
                break
            start = max(start + 1, end - overlap_chars)

        return spans

    @staticmethod
    def map_char_spans_to_token_spans(
        char_spans: List[Tuple[int, int, str]],
        offset_mapping: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Maps character-level (char_start, char_end) boundaries to token indices (tok_start, tok_end).
        Ignores special tokens like [CLS] and [SEP] which have offset (0, 0).
        """
        token_spans: List[Tuple[int, int]] = []

        for char_start, char_end, _ in char_spans:
            tok_start = None
            tok_end = None

            for tok_idx, (t_char_start, t_char_end) in enumerate(offset_mapping):
                # Special tokens ([CLS], [SEP], [PAD]) have (0, 0) offsets in HF fast tokenizers
                if t_char_start == 0 and t_char_end == 0:
                    continue

                # First token that overlaps with the chunk's character start
                if tok_start is None and t_char_end > char_start:
                    tok_start = tok_idx

                # Last token that overlaps with the chunk's character end
                if t_char_start < char_end:
                    tok_end = tok_idx + 1  # Exclusive upper bound for slicing

            # Fallback guard if a span only hit whitespace
            if tok_start is None or tok_end is None or tok_start >= tok_end:
                tok_start, tok_end = 1, 2
            token_spans.append((tok_start, tok_end))

        return token_spans

    @torch.no_grad()
    def encode_document_chunks(
        self, document: str, chunk_char_size: int = 250, overlap_chars: int = 30
    ) -> List[Dict[str, object]]:
        """
        Executes a single forward pass over the full document and pools token embeddings per chunk.
        """
        # 1. Generate character-level chunk boundaries
        char_spans = self.get_char_chunk_spans(document, chunk_char_size, overlap_chars)

        # 2. Tokenize full document with character offset mapping
        encoded = self.tokenizer(
            document,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=8192,
        )
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        # 3. Single forward pass through the bi-encoder -> H shape: [1, seq_len, hidden_dim]
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # Shape: [seq_len, hidden_dim]
        attention_mask = inputs["attention_mask"][0]     # Shape: [seq_len]

        # 4. Align character spans to token spans
        token_spans = self.map_char_spans_to_token_spans(char_spans, offset_mapping)

        # 5. Late Mean Pooling per span + L2 Normalization
        results = []
        for (char_start, char_end, chunk_text), (tok_start, tok_end) in zip(char_spans, token_spans):
            span_vectors = token_embeddings[tok_start:tok_end]          # [span_len, d]
            span_mask = attention_mask[tok_start:tok_end].unsqueeze(-1) # [span_len, 1]

            # Mask-safe mean pooling (prevents division by zero on empty/padded slices)
            valid_token_count = torch.clamp(span_mask.sum(dim=0), min=1e-9)
            pooled_vec = (span_vectors * span_mask).sum(dim=0) / valid_token_count

            # L2 normalize so dot product == cosine similarity in Vector DB
            normalized_vec = F.normalize(pooled_vec, p=2, dim=0)

            results.append({
                "chunk_text": chunk_text,
                "char_span": (char_start, char_end),
                "token_span": (tok_start, tok_end),
                "embedding": normalized_vec.cpu().numpy(),
            })

        return results


# --- Verification Demo ---
if __name__ == "__main__":
    sample_doc = (
        "The Volkswagen ID.4 BMS firmware v3.2 regulates thermal throttling during DC fast charging. "
        "The telemetry controller samples cell voltages every 10 milliseconds across all 12 modules. "
        "Under extreme cold (-20C), it limits current draw to 45A for the first 8 minutes to prevent lithium plating."
    )
    # Using mock tensor validation so you can trace shapes without downloading weights:
    char_spans = LateChunkingEmbedder.get_char_chunk_spans(sample_doc, chunk_char_size=95, overlap_chars=0)
    for idx, (cs, ce, txt) in enumerate(char_spans):
        print(f"Chunk {idx} [{cs}:{ce}]: {txt}")
```

#### Key Interview Code Signals to Point Out:
1. **`@torch.no_grad()` & `self.model.eval()`:** Forgetting `torch.no_grad()` retains computation graphs in VRAM, causing instant Out-Of-Memory (OOM) crashes on 8,192-token sequences. Forgetting `.eval()` leaves Dropout active, making embeddings non-deterministic.
2. **Skipping `(0, 0)` Special Token Offsets:** `[CLS]` and `[SEP]` tokens have `(0, 0)` character offsets. If you accidentally include `[CLS]` inside Chunk 0's span pool, Chunk 0's vector becomes skewed toward the global `[CLS]` representation.
3. **`torch.clamp(span_mask.sum(), min=1e-9)`:** Prevents `NaN` vectors if a corrupted span contains 0 unmasked tokens.

---

## Block 2: Low-Level Code Design (LLD) — Two-Level SHA-256 Record Manager (Minutes 15–35)

### 2.1 How Atlassian & Intuit Test This in "Code Design" Rounds
In Atlassian (P40/P50) and Intuit Senior AI loops, you are given a 60-minute **Code Design / Machine Coding round**. They ask you to write clean, modular, object-oriented Python (using dataclasses, type hints, and clear state interfaces) for an infrastructure component—such as an **Idempotent RAG Ingestion & Diffing Engine**.

#### Evolving Requirements Pattern in Interviews:
* **Part 1 (Base Requirement):** Given an incoming document and its chunks, compute hashes so unchanged documents and unchanged paragraphs are skipped without calling the GPU embedder.
* **Part 2 (Scale/Edge Case Twist):** What if two identical paragraphs appear in the same document (e.g., standard legal disclaimers)? What if a user only changes the document's RBAC `allowed_groups` without editing the text?
* **How Our Code Solves Both Twists:**
  1. **Composite Chunk Hash vs. Payload Hash:** If `allowed_groups` changes, we should **not** re-run the GPU embedding model (the text didn't change!), we should only issue a lightweight `set_payload` metadata update to the Vector DB. Therefore, we track `text_hash` (controls GPU embedding) separately from `acl_hash` (controls metadata payload update).
  2. **Deterministic UUIDv5 Generation:** Vector DBs like Qdrant require UUIDs or 64-bit unsigned integers as point IDs. Using `uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}#{chunk_index}#{text_hash}")` generates a deterministic, collision-free UUID without needing a centralized auto-increment counter.

---

### 2.2 Complete, Runnable OOP Code: Production Record Manager & Diff Engine

```python
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


@dataclass(frozen=True)
class IncomingChunk:
    """Represents a parsed chunk produced by the layout-aware splitter."""
    chunk_index: int
    text: str
    section_breadcrumb: str


@dataclass(frozen=True)
class StoredChunkRecord:
    """State row persisted in PostgreSQL / DynamoDB Record Manager."""
    point_id: str          # Deterministic UUIDv5 stored in Vector DB
    doc_id: str
    chunk_index: int
    text_hash: str         # SHA-256 of (section_breadcrumb + text) -> triggers GPU re-embed
    acl_hash: str          # SHA-256 of sorted allowed_groups -> triggers metadata-only update


@dataclass
class IngestionDiffPlan:
    """Execution plan emitted by the Record Manager for a single document update."""
    doc_id: str
    short_circuited: bool = False
    chunks_to_embed_and_upsert: List[Tuple[str, IncomingChunk]] = field(default_factory=list)
    point_ids_for_acl_only_update: List[str] = field(default_factory=list)
    orphaned_point_ids_to_delete: List[str] = field(default_factory=list)
    skipped_unchanged_count: int = 0


class IdempotentRecordManager:
    """
    Two-level SHA-256 state diffing engine for incremental RAG indexing.
    Separates text mutations (GPU re-embedding required) from ACL mutations (payload-only update).
    """

    def __init__(self):
        # In production, these two dictionaries are backed by PostgreSQL tables with B-tree indexes on doc_id
        self._doc_hash_store: Dict[str, str] = {}
        self._chunk_records_by_doc: Dict[str, Dict[int, StoredChunkRecord]] = {}

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def compute_acl_hash(cls, allowed_groups: List[str]) -> str:
        # Sort groups so ["eng", "legal"] and ["legal", "eng"] produce the exact same hash
        normalized = ",".join(sorted(set(allowed_groups)))
        return cls._sha256(normalized)

    @classmethod
    def compute_chunk_text_hash(cls, chunk: IncomingChunk) -> str:
        return cls._sha256(f"{chunk.section_breadcrumb}||{chunk.text}")

    @staticmethod
    def make_deterministic_point_id(doc_id: str, chunk_index: int, text_hash: str) -> str:
        """Generates a deterministic UUIDv5 compatible with Qdrant / Milvus."""
        seed = f"{doc_id}::chunk_{chunk_index}::{text_hash}"
        return str(uuid.uuid5(uuid.NAMESPACE_OID, seed))

    def compute_diff_plan(
        self,
        doc_id: str,
        raw_full_text: str,
        incoming_chunks: List[IncomingChunk],
        allowed_groups: List[str],
    ) -> IngestionDiffPlan:
        """
        Compares incoming document + chunks against stored state and returns a minimal mutation plan.
        Does NOT mutate state until commit_plan() is called after Vector DB confirmation.
        """
        acl_hash = self.compute_acl_hash(allowed_groups)
        full_doc_composite_hash = self._sha256(f"{raw_full_text}||ACL:{acl_hash}")

        # Level 1 Short-Circuit: Neither text nor ACLs changed across the entire document
        if self._doc_hash_store.get(doc_id) == full_doc_composite_hash:
            return IngestionDiffPlan(
                doc_id=doc_id,
                short_circuited=True,
                skipped_unchanged_count=len(incoming_chunks),
            )

        plan = IngestionDiffPlan(doc_id=doc_id)
        existing_chunks: Dict[int, StoredChunkRecord] = self._chunk_records_by_doc.get(doc_id, {})
        seen_indices: Set[int] = set()

        # Level 2 Chunk-by-Chunk Diffing
        for chunk in incoming_chunks:
            idx = chunk.chunk_index
            seen_indices.add(idx)
            new_text_hash = self.compute_chunk_text_hash(chunk)
            new_point_id = self.make_deterministic_point_id(doc_id, idx, new_text_hash)

            old_record: Optional[StoredChunkRecord] = existing_chunks.get(idx)

            if old_record is None:
                # Case A: Brand new chunk index added to document
                plan.chunks_to_embed_and_upsert.append((new_point_id, chunk))
            elif old_record.text_hash != new_text_hash:
                # Case B: Chunk text was edited -> Queue new embedding AND delete old point_id
                plan.chunks_to_embed_and_upsert.append((new_point_id, chunk))
                plan.orphaned_point_ids_to_delete.append(old_record.point_id)
            elif old_record.acl_hash != acl_hash:
                # Case C: Text is 100% identical, ONLY RBAC groups changed!
                # Zero GPU compute needed -> issue lightweight metadata-only update on existing point_id
                plan.point_ids_for_acl_only_update.append(old_record.point_id)
            else:
                # Case D: Both text and ACLs are identical -> Complete no-op
                plan.skipped_unchanged_count += 1

        # Identify deleted trailing chunks (e.g., doc shrank from 10 chunks to 7 chunks)
        for old_idx, old_record in existing_chunks.items():
            if old_idx not in seen_indices:
                plan.orphaned_point_ids_to_delete.append(old_record.point_id)

        return plan

    def commit_plan(
        self,
        doc_id: str,
        raw_full_text: str,
        incoming_chunks: List[IncomingChunk],
        allowed_groups: List[str],
    ) -> None:
        """Atomically commits the new state after Vector DB upsert/delete succeeds."""
        acl_hash = self.compute_acl_hash(allowed_groups)
        self._doc_hash_store[doc_id] = self._sha256(f"{raw_full_text}||ACL:{acl_hash}")

        new_records: Dict[int, StoredChunkRecord] = {}
        for chunk in incoming_chunks:
            t_hash = self.compute_chunk_text_hash(chunk)
            p_id = self.make_deterministic_point_id(doc_id, chunk.chunk_index, t_hash)
            new_records[chunk.chunk_index] = StoredChunkRecord(
                point_id=p_id,
                doc_id=doc_id,
                chunk_index=chunk.chunk_index,
                text_hash=t_hash,
                acl_hash=acl_hash,
            )
        self._chunk_records_by_doc[doc_id] = new_records


# --- Verification Demo ---
if __name__ == "__main__":
    rm = IdempotentRecordManager()
    doc_id = "CONFLUENCE_PAGE_402"

    # Initial Ingestion (3 chunks)
    v1_chunks = [
        IncomingChunk(0, "BMS thermal spec v3.2 overview.", "BMS > Overview"),
        IncomingChunk(1, "Sampling rate is 10ms across 12 modules.", "BMS > Telemetry"),
        IncomingChunk(2, "Cold limit is 45A at -20C.", "BMS > Thermal Limits"),
    ]
    v1_raw = "\n".join(c.text for c in v1_chunks)
    plan1 = rm.compute_diff_plan(doc_id, v1_raw, v1_chunks, ["grp_bms_eng"])
    rm.commit_plan(doc_id, v1_raw, v1_chunks, ["grp_bms_eng"])
    print(f"Run 1 (Initial): Embed={len(plan1.chunks_to_embed_and_upsert)}, Skip={plan1.skipped_unchanged_count}")

    # Edit 1: Paragraph 1 edited; Paragraphs 0 and 2 untouched
    v2_chunks = [
        IncomingChunk(0, "BMS thermal spec v3.2 overview.", "BMS > Overview"),
        IncomingChunk(1, "Sampling rate upgraded to 5ms across 12 modules.", "BMS > Telemetry"),
        IncomingChunk(2, "Cold limit is 45A at -20C.", "BMS > Thermal Limits"),
    ]
    v2_raw = "\n".join(c.text for c in v2_chunks)
    plan2 = rm.compute_diff_plan(doc_id, v2_raw, v2_chunks, ["grp_bms_eng"])
    rm.commit_plan(doc_id, v2_raw, v2_chunks, ["grp_bms_eng"])
    print(
        f"Run 2 (1 Paragraph Edited): Embed={len(plan2.chunks_to_embed_and_upsert)}, "
        f"Delete={len(plan2.orphaned_point_ids_to_delete)}, Skip={plan2.skipped_unchanged_count}"
    )

    # Edit 2: Text untouched, but Legal group granted read access
    plan3 = rm.compute_diff_plan(doc_id, v2_raw, v2_chunks, ["grp_bms_eng", "grp_legal"])
    print(
        f"Run 3 (ACL-Only Change): Embed={len(plan3.chunks_to_embed_and_upsert)}, "
        f"Metadata-Only Update={len(plan3.point_ids_for_acl_only_update)}"
    )
```

#### Why This Impresses Senior Interviewers:
1. **Zero GPU Waste on Permission Changes (`point_ids_for_acl_only_update`):** In enterprise Confluence/SharePoint, permissions change constantly. Separating `text_hash` from `acl_hash` means permission updates trigger a 1ms Qdrant `set_payload()` call instead of re-running GPU inference.
2. **Insert-Before-Delete Consistency:** Notice how `compute_diff_plan` outputs both `chunks_to_embed_and_upsert` (with a **new** deterministic UUIDv5) and `orphaned_point_ids_to_delete` (the **old** UUIDv5). In your worker, you **upsert the new points first**, and **delete the old points second**. If a user queries the database during the 5ms window while the update is in flight, they never see a missing paragraph!

---

## Block 3: Production Qdrant RBAC Engineering & Advanced Internals (Minutes 35–50)

### 3.1 Four Critical Production Details Most Candidates Get Wrong
When discussing Qdrant or Milvus in a Senior System Design round, drop these four exact engineering realities:

1. **Create Payload Indexes *Before* Bulk Ingestion:**  
   Filterable HNSW builds extra subgraph edges per metadata value (`allowed_groups`) **when the HNSW graph is constructed**. If you upload 20 million vectors first and call `create_payload_index()` afterward, the vector DB has to rebuild the entire HNSW graph from scratch to wire up the subgraph bridges! Always create your `KEYWORD` payload index on an empty collection before ingesting vectors.
2. **`full_scan_threshold` is Measured in Kilobytes (KB), Not Vector Count:**  
   In Qdrant's `HnswConfigDiff`, `full_scan_threshold` (default `10000` = **10,000 KB = 10 MB**) specifies the **estimated size in Kilobytes** of the filtered candidate vectors below which the query planner skips HNSW and runs a SIMD flat scan.
   * **Example Math:** For 768-dim FP32 vectors ($768 \times 4\text{ B} = 3\text{ KB/vector}$), setting `full_scan_threshold = 6000` (6,000 KB) tells Qdrant: *"Whenever a filter matches fewer than $\frac{6000\text{ KB}}{3\text{ KB}} = 2,000\text{ vectors}$, bypass HNSW and run an exact SIMD brute-force scan."*
3. **Co-Locating Multi-Tenant Data on Disk (`is_tenant=True`):**  
   In a multi-tenant cloud architecture (e.g., Atlassian Jira organizations or Volkswagen divisions), set `is_tenant=True` inside `KeywordIndexParams`. This instructs Qdrant to physically **co-locate vectors belonging to the same tenant in adjacent disk/memory pages**, eliminating random SSD page faults during filtered reads.
4. **What Happens When a Query Combines *Multiple* Filters? (The ACORN Algorithm):**  
   Filterable HNSW subgraph bridges work seamlessly for a single filter (`allowed_groups ANY user_groups`). What if a user queries with **three simultaneous filters**: `allowed_groups ANY user_groups AND created_at >= 2026-01-01 AND doc_type == "spec"`? You cannot pre-build subgraph bridges for every possible combinatorial intersection!
   * **The Solution (ACORN / Two-Hop Predicate Traversal, Qdrant 1.16+):** When multiple filters intersect and the cardinality is still too large for a flat scan, the engine switches to **ACORN traversal**. Instead of only inspecting immediate 1-hop neighbors ($M$) in the HNSW graph and getting stuck at blocked nodes, ACORN dynamically expands the search frontier to **2-hop neighbors ($M^2$)** whenever immediate neighbors fail the filter predicate, maintaining graph connectivity on the fly without pre-built combinatorial edges.

---

### 3.2 Complete Python Code: Production Qdrant Collection Setup, Upsert & RBAC Search

```python
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models


class EnterpriseVectorStore:
    """
    Production Qdrant wrapper demonstrating:
    1. HNSW Vacuum Optimizer tuning (tombstone compaction)
    2. Pre-ingestion Filterable HNSW payload indexing with disk co-location (is_tenant=True)
    3. Zero-trust group-level RBAC search with payload-only ACL updates
    """

    def __init__(self, collection_name: str = "enterprise_rag_v1", vector_dim: int = 768):
        # Uses in-memory instance for local execution; replace with url="http://qdrant:6333" in prod
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self._init_production_collection()

    def _init_production_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        # Step 1: Create collection with tuned HNSW & Vacuum Compaction parameters
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE,
                on_disk=False,  # Keep quantized vectors in RAM for <15ms p99 latency
            ),
            # INT8 Scalar Quantization cuts RAM by 4x (FP32 -> INT8) with <1% recall drop
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,                       # Max bidirectional edges per node in base layer L0
                payload_m=16,               # Extra subgraph edges per payload value (Filterable HNSW)
                ef_construct=128,           # Search width during graph build
                full_scan_threshold=6000,   # 6,000 KB (~2,000 vectors) cutoff for SIMD flat-scan fallback
            ),
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=0.20,          # Compact segment when tombstones > 20%
                vacuum_min_vector_number=1000,   # Min segment size to run vacuum compaction
                default_segment_number=4,        # Parallel search segments
            ),
        )

        # Step 2: ALWAYS create Payload Indexes BEFORE bulk ingestion so subgraph edges build once!
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="allowed_groups",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,  # Co-locates vectors sharing the same group on adjacent pages
                on_disk=False,
            ),
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

    def apply_diff_plan(
        self,
        doc_id: str,
        allowed_groups: List[str],
        embeddings_and_chunks: List[Dict[str, Any]],
        acl_only_point_ids: List[str],
        orphaned_point_ids: List[str],
    ) -> None:
        """
        Executes the Record Manager diff plan in strict order:
        1. Upsert new/modified vectors first (zero downtime)
        2. Update metadata in-place for ACL-only changes (zero GPU cost)
        3. Soft-delete orphaned point IDs last
        """
        # 1. Upsert newly embedded chunks
        if embeddings_and_chunks:
            points = [
                models.PointStruct(
                    id=item["point_id"],
                    vector=item["vector"],
                    payload={
                        "doc_id": doc_id,
                        "chunk_index": item["chunk_index"],
                        "text": item["text"],
                        "section_breadcrumb": item["section_breadcrumb"],
                        "allowed_groups": allowed_groups,
                    },
                )
                for item in embeddings_and_chunks
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

        # 2. Fast metadata-only update for chunks where only permissions changed
        if acl_only_point_ids:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"allowed_groups": allowed_groups},
                points=acl_only_point_ids,
            )

        # 3. Delete orphaned chunk versions (marks soft tombstones until Vacuum compacts)
        if orphaned_point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=orphaned_point_ids),
            )

    def search_with_rbac(
        self,
        query_vector: List[float],
        user_active_groups: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Executes single-stage Filterable HNSW search enforced by the user's IAM/Redis security groups.
        """
        if not user_active_groups:
            raise PermissionError("Zero-trust policy: User has no active security groups.")

        rbac_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="allowed_groups",
                    match=models.MatchAny(any=user_active_groups),
                )
            ]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=rbac_filter,
            limit=top_k,
            search_params=models.SearchParams(
                hnsw_ef=64,          # Priority queue width during L0 traversal
                exact=False,         # Allow Adaptive Query Planner to choose HNSW vs SIMD Flat Scan
                quantization=models.QuantizationSearchParams(
                    rescore=True,    # Search in INT8, then rescore top candidates with original vectors
                    oversampling=2.0,
                ),
            ),
        )

        return [
            {
                "point_id": hit.id,
                "score": round(hit.score, 4),
                "doc_id": hit.payload.get("doc_id"),
                "text": hit.payload.get("text"),
                "breadcrumb": hit.payload.get("section_breadcrumb"),
            }
            for hit in response.points
        ]


# --- Verification Demo ---
if __name__ == "__main__":
    store = EnterpriseVectorStore(vector_dim=4)
    p1, p2 = "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"

    store.apply_diff_plan(
        doc_id="DOC_BMS_402",
        allowed_groups=["grp_bms_eng"],
        embeddings_and_chunks=[
            {
                "point_id": p1,
                "vector": [0.9, 0.1, 0.0, 0.0],
                "chunk_index": 0,
                "text": "Cold limit is 45A at -20C.",
                "section_breadcrumb": "BMS > Thermal",
            }
        ],
        acl_only_point_ids=[],
        orphaned_point_ids=[],
    )
    store.apply_diff_plan(
        doc_id="DOC_HR_101",
        allowed_groups=["grp_hr_all"],
        embeddings_and_chunks=[
            {
                "point_id": p2,
                "vector": [0.85, 0.15, 0.0, 0.0],
                "chunk_index": 0,
                "text": "Annual leave policy for 2026.",
                "section_breadcrumb": "HR > Leave",
            }
        ],
        acl_only_point_ids=[],
        orphaned_point_ids=[],
    )

    # Engineer queries -> only sees BMS chunk despite HR chunk having high cosine similarity
    hits = store.search_with_rbac([0.88, 0.12, 0.0, 0.0], user_active_groups=["grp_bms_eng"])
    print("RBAC Filtered Hits for Engineer:", hits)
```

---

## Block 4: Rehearsed 90-Second Verbal Scripts (Speak Out Loud)

In your daily 45-minute study routine, spend the final 5 minutes reading these two scripts **out loud** so your phrasing is effortless in live interviews:

### Script 1: Defending Late Chunking vs. Contextual Retrieval (Bridging Samsung + Volkswagen)
> *"In our enterprise RAG architectures, standard chunking fails on technical specs because pronouns like 'it' or 'the module' lose their entity definitions across paragraph splits. We evaluate two solutions depending on whether the bottleneck is ingestion budget or lexical hybrid search.*  
> *When running self-hosted long-context encoders—similar to the bi-encoders I worked on at Samsung—we use **Late Chunking**: we pass the full 8,192-token document through the Transformer layers first, align character chunks to subword token spans via `return_offsets_mapping`, and mean-pool each span. That gives every chunk full bidirectional document context with zero generative LLM cost.*  
> *However, Late Chunking doesn't modify the raw text, meaning **BM25 lexical search still misses entity keywords** in pronoun-heavy paragraphs. Therefore, for hybrid BM25 + Dense pipelines, we either pair Late Chunking with automated DOM header/breadcrumb injection, or use Anthropic-style **Contextual Retrieval with Prefix KV Caching** when exact keyword expansion is critical."*

### Script 2: Defending Incremental Indexing & Zero-Trust RBAC
> *"At enterprise scale with tens of thousands of edits per hour, re-embedding full documents on every webhook event wastes over 90% of GPU capacity and causes HNSW tombstone bloat. I design a **Two-Level SHA-256 Record Manager** that hashes both the text and the RBAC security groups separately. Unchanged paragraphs are skipped completely; edited paragraphs get a new deterministic UUIDv5 upserted *before* the old UUID is soft-deleted; and permission-only changes trigger a 1-millisecond `set_payload` update without touching the GPU.*  
> *On the retrieval side, we never store raw `user_ids` in vector metadata to avoid ACL update storms. We store immutable `allowed_group_ids`, create the `KEYWORD` payload index with `is_tenant=True` **before** bulk ingestion so Filterable HNSW wires subgraph bridges at build time, and configure the Vacuum Optimizer at a 20% `deleted_threshold` so tombstones don't pollute our `efSearch` candidate queue."*
