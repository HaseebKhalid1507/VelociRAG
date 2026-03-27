"""
Cross-encoder reranking for Velocirag.

Uses ONNX Runtime for cross-encoder inference — no PyTorch required.
Downloads the TinyBERT cross-encoder model on first use and caches it locally.
Provides smart document excerpting and graceful degradation on load failure.
"""

import logging
import os
import threading
import numpy as np
from typing import Dict, List, Any

# Constants
DEFAULT_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
DEFAULT_MODEL_CACHE = os.path.expanduser("~/.cache/velocirag/models/cross-encoder-tinybert")
ONNX_FILENAME = "onnx/model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
MAX_SEQ_LENGTH = 512
MAX_EXCERPT_LENGTH = 2000
EXCERPT_HEAD = 1000
EXCERPT_TAIL = 1000

logger = logging.getLogger("velocirag.reranker")

_load_lock = threading.Lock()


class Reranker:
    """
    Cross-encoder reranker for search result relevance scoring.

    Uses a TinyBERT cross-encoder (ONNX Runtime) to score query-document pairs
    and rerank search results by relevance. No PyTorch dependency — uses the
    same onnxruntime backend as the embedder.

    Features:
    - Lazy model loading (model loaded only on first rerank() call)
    - Thread-safe initialization
    - Smart document excerpting for long content
    - Graceful degradation if model unavailable
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize reranker with lazy model loading.

        Args:
            model_name: HuggingFace cross-encoder model identifier
        """
        self.model_name = model_name
        self._model_session = None
        self._tokenizer = None
        self._loaded = False
        self._load_error = None

    def rerank(self, query: str, results: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder relevance scoring.

        Scores query-document pairs and returns results sorted by relevance.
        Adds rerank_score to each result's metadata.

        Args:
            query: Original search query
            results: List of results with 'content' key
            limit: Max results to return

        Returns:
            Reranked results with rerank_score in metadata, limited to specified count
        """
        if not results:
            return results

        # Ensure model is loaded
        if not self._loaded and self._load_error is None:
            self._load_model()

        # Graceful degradation if model unavailable
        if self._load_error:
            logger.warning(f"Reranker unavailable ({self._load_error}), returning unranked results")
            return results[:limit]

        try:
            pairs = []
            for result in results:
                content = result.get('content', '')
                pairs.append((query, self._excerpt_content(content)))

            scores = self._predict(pairs)

            scored_results = []
            for i, result in enumerate(results):
                result_copy = result.copy()
                if 'metadata' not in result_copy:
                    result_copy['metadata'] = {}
                result_copy['metadata']['rerank_score'] = round(float(scores[i]), 4)
                scored_results.append((float(scores[i]), result_copy))

            scored_results.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored_results[:limit]]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning unranked results")
            return results[:limit]

    def __call__(self, query: str, results: list[dict], limit: int = 5) -> list[dict]:
        """Make Reranker callable — enables passing directly to Searcher."""
        return self.rerank(query, results, limit)

    def get_status(self) -> Dict[str, Any]:
        """Get reranker status and configuration."""
        return {
            'model_name': self.model_name,
            'loaded': self._loaded,
            'error': self._load_error
        }

    def _load_model(self) -> None:
        """Load ONNX cross-encoder model and tokenizer. Thread-safe."""
        with _load_lock:
            if self._loaded or self._load_error:
                return

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
                from huggingface_hub import hf_hub_download

                cache_dir = DEFAULT_MODEL_CACHE
                os.makedirs(cache_dir, exist_ok=True)

                model_path = os.path.join(cache_dir, "onnx", "model.onnx")
                tokenizer_path = os.path.join(cache_dir, "tokenizer.json")

                if not os.path.exists(model_path):
                    logger.info("Downloading cross-encoder ONNX model (first run)...")
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=ONNX_FILENAME,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False
                    )

                if not os.path.exists(tokenizer_path):
                    logger.info("Downloading cross-encoder tokenizer...")
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=TOKENIZER_FILENAME,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False
                    )

                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = min(os.cpu_count() or 1, 4)
                session_options.log_severity_level = 3  # Suppress ONNX warnings

                self._model_session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options
                )

                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
                self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

                # Inspect output names for inference
                self._output_names = [o.name for o in self._model_session.get_outputs()]
                self._input_names = [i.name for i in self._model_session.get_inputs()]

                self._loaded = True
                logger.info(f"Cross-encoder ONNX model loaded: {self.model_name}")

            except Exception as e:
                self._load_error = str(e)
                logger.error(f"Failed to load cross-encoder model: {e}")

    def _predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """
        Run cross-encoder inference on query-document pairs.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            Array of relevance scores (sigmoid-normalized), shape [n_pairs]
        """
        # Encode all pairs — tokenizer handles [CLS] query [SEP] doc [SEP]
        encodings = [self._tokenizer.encode(query, doc) for query, doc in pairs]

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        feed = {}
        for name in self._input_names:
            if "input_ids" in name:
                feed[name] = input_ids
            elif "attention_mask" in name:
                feed[name] = attention_mask
            elif "token_type_ids" in name:
                feed[name] = token_type_ids

        outputs = self._model_session.run(self._output_names, feed)
        logits = outputs[0]  # shape: [batch_size, 1] or [batch_size, 2]

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim == 2 and logits.shape[1] == 2:
            # Binary classification — take positive class logit
            logits = logits[:, 1]

        # Sigmoid to get relevance score in [0, 1]
        return 1.0 / (1.0 + np.exp(-logits))

    def _excerpt_content(self, content: str) -> str:
        """
        Create smart excerpt for long documents.

        For documents longer than MAX_EXCERPT_LENGTH, takes first EXCERPT_HEAD
        characters and last EXCERPT_TAIL characters to capture both intro
        and conclusion.

        Args:
            content: Full document content

        Returns:
            Excerpted content suitable for cross-encoder input
        """
        if len(content) <= MAX_EXCERPT_LENGTH:
            return content
        head = content[:EXCERPT_HEAD].rstrip()
        tail = content[-EXCERPT_TAIL:].lstrip()
        return f"{head}\n...\n{tail}"
