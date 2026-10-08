"""CPU FAISS + E5 retrieval server for the search-tool RL recipe.

Serves the prebuilt Search-R1 wiki-18 dense index (``intfloat/e5-base-v2``,
768-dim, a FAISS ``IndexFlatIP``) directly -- no embedding build is required.
The server encodes incoming queries with E5 (mean pooling + L2 normalization
and the ``"query: "`` prefix), searches the flat index, and returns the top-k
wiki-18 passages. Query and document vectors live in the same E5 space because
the document side was embedded with the same model when the index was built
upstream.

Everything runs on CPU: ``faiss-cpu`` for the brute-force inner-product search
and the E5 encoder on the CPU device. The flat index holds ~21M float32
vectors (~64 GB), so the host needs enough RAM to load it (the bench pins this
to a large-memory compute).

Run::

    python -m interactive_training.recipes.search_tool.retrieval_server \\
        --index-path /data/e5_Flat.index \\
        --corpus-path /data/wiki-18.jsonl \\
        --model intfloat/e5-base-v2 \\
        --host 0.0.0.0 --port 8000

The recipe's retrieval tool (see ``tools.py``) is a thin HTTP client that POSTs
``{"queries": [...], "topk": k}`` to ``/retrieve`` and receives
``{"result": [[passage, ...], ...]}``.
"""

from __future__ import annotations

import argparse
import json
import threading
from logging import INFO, basicConfig, getLogger

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

logger = getLogger(__name__)

DEFAULT_MODEL = "intfloat/e5-base-v2"
DEFAULT_TOPK = 3
# E5 queries are short; 256 tokens matches the Search-R1 retrieval server.
QUERY_MAX_LENGTH = 256
# FAISS parallelizes the flat-index search across OpenMP threads, and each
# thread's BLAS call would itself spawn threads. On many-core hosts that nested
# oversubscription blows past OpenBLAS's per-thread buffer limit and aborts the
# process ("BLAS : Program is Terminated. Because you tried to allocate too many
# memory regions."). Cap FAISS/torch threads to a modest default and force
# single-threaded inner BLAS (the bench also exports OPENBLAS_NUM_THREADS=1).
DEFAULT_NUM_THREADS = 8


def load_corpus(corpus_path: str) -> list[str]:
    """Load passage ``contents`` strings indexed by FAISS row id.

    The FAISS index rows align 1:1 with the corpus JSONL line order, so the
    returned list can be indexed directly by the ids FAISS returns. Each line
    is a JSON object with a ``contents`` field (or ``title``/``text``).
    """
    contents: list[str] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("contents")
            if text is None:
                title = obj.get("title", "")
                body = obj.get("text", "")
                text = f"{title}\n{body}".strip() if title else body
            contents.append(str(text))
    return contents


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding (the E5 pooling method)."""
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Encoder:
    """CPU E5 query encoder (mean pooling + L2 normalize, ``"query: "`` prefix)."""

    def __init__(self, model_path: str, max_length: int = QUERY_MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        texts = [f"query: {q}" for q in queries]
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        output = self.model(**inputs, return_dict=True)
        emb = _mean_pool(output.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        # FAISS expects C-contiguous float32.
        return emb.cpu().numpy().astype(np.float32, order="C")


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: int | None = None


def build_app(
    index: faiss.Index,
    corpus: list[str],
    encoder: E5Encoder,
    default_topk: int,
) -> FastAPI:
    app = FastAPI(title="search-tool E5 retrieval")
    # FastAPI runs sync endpoints in a threadpool, so concurrent requests would
    # otherwise launch overlapping FAISS searches (each already multi-threaded
    # across the capped CPU cores) and oversubscribe BLAS. Serialize the heavy
    # encode+search so one request fully uses the allotted threads at a time.
    search_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest) -> dict[str, list[list[str]]]:
        topk = request.topk or default_topk
        if not request.queries:
            return {"result": []}
        with search_lock:
            embeddings = encoder.encode_queries(request.queries)
            _scores, idxs = index.search(embeddings, topk)
        result: list[list[str]] = []
        for row in idxs:
            # FAISS returns -1 for empty slots when fewer than topk hits exist.
            result.append([corpus[int(i)] for i in row if i >= 0])
        return {"result": result}

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to the prebuilt FAISS flat index (e.g. e5_Flat.index).",
    )
    parser.add_argument(
        "--corpus-path",
        required=True,
        help="Path to the wiki-18 corpus JSONL whose line order matches the index.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model id / path of the E5 query encoder.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="Default number of passages returned when a request omits topk.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=(
            "Max FAISS/torch CPU threads. Kept modest to avoid OpenBLAS "
            "nested-thread oversubscription on many-core hosts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    # Cap CPU threading before any heavy compute to avoid OpenBLAS nested-thread
    # oversubscription (FAISS OMP threads x BLAS threads) aborting the server.
    faiss.omp_set_num_threads(args.num_threads)
    torch.set_num_threads(args.num_threads)
    logger.info("Capped FAISS/torch CPU threads to %d", args.num_threads)

    logger.info("Loading FAISS index from %s ...", args.index_path)
    index = faiss.read_index(args.index_path)
    logger.info("Index loaded: ntotal=%d, d=%d", index.ntotal, index.d)

    logger.info("Loading corpus from %s ...", args.corpus_path)
    corpus = load_corpus(args.corpus_path)
    logger.info("Corpus loaded: %d passages", len(corpus))
    if len(corpus) != index.ntotal:
        logger.warning(
            "Corpus size (%d) != index ntotal (%d); ids may be misaligned.",
            len(corpus),
            index.ntotal,
        )

    logger.info("Loading E5 encoder %s (CPU) ...", args.model)
    encoder = E5Encoder(args.model)

    app = build_app(index, corpus, encoder, args.topk)
    logger.info("Serving retrieval on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
