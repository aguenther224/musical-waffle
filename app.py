import os
import re
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from flask import Flask, request, jsonify
import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


"""
Innovative approach: Reflexive Retrieval-Augmented Decisioning (RRAD)

This service blends three light-weight AI behaviors without external APIs or keys:
1) Neuro-symbolic routing:
   - Detects and executes numeric/logic intents (e.g., sum, difference, average).
   - Falls back to retrieval or classification when computation isn’t applicable.

2) Reflexive retrieval:
   - Creates several low-cost paraphrases of the query.
   - Performs consensus-based retrieval over a hashed vector space to reduce brittleness.

3) Online self-improvement:
   - Accepts incremental labeled examples via /teach.
   - Updates an online classifier (PassiveAggressive) with hashing features.
   - Uses a simple confidence calibration from decision margins.

All state is in-memory for simplicity. No API keys required.
"""


@dataclass
class Doc:
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflexiveAIService:
    def __init__(self):
        # HashingVectorizer avoids fitting, enables fast updates and low memory
        self.vectorizer = HashingVectorizer(
            n_features=2**16,
            alternate_sign=False,
            norm="l2",
            analyzer="char",
            ngram_range=(3, 5),
            stop_words=None
        )

        # Retrieval memory
        self.docs: List[Doc] = []
        self.X_docs: Optional[csr_matrix] = None

        # Online classifier
        self.clf = PassiveAggressiveClassifier(max_iter=5, random_state=42)
        self.known_labels: List[str] = []
        self.clf_initialized = False

        # Simple synonyms/paraphrase mapping for reflexive retrieval
        self.syn_map = {
            r"\bhow many\b": "count",
            r"\bprice\b": "cost",
            r"\bbuy\b": "purchase",
            r"\bfaq\b": "frequently asked questions",
            r"\binfo\b": "information",
            r"\bhelp\b": "assist",
            r"\bguide\b": "manual",
            r"\bissue\b": "problem",
            r"\bbug\b": "defect",
        }

        # Arithmetic keyword sets
        self.ops = {
            "sum": {"sum", "add", "plus", "total"},
            "sub": {"subtract", "minus", "difference", "less"},
            "mul": {"multiply", "times", "product"},
            "div": {"divide", "over", "quotient"},
            "avg": {"average", "mean"},
            "max": {"max", "maximum", "largest", "greatest"},
            "min": {"min", "minimum", "smallest", "least"},
            "count_words": {"how many words", "word count"},
        }

    # ---------- Utilities ----------
    def _to_vec(self, texts: List[str]) -> csr_matrix:
        return self.vectorizer.transform(texts)

    def _append_doc(self, text: str, metadata: Dict[str, Any]):
        doc_id = len(self.docs)
        self.docs.append(Doc(id=doc_id, text=text, metadata=metadata or {}))
        vec = self._to_vec([text])
        if self.X_docs is None:
            self.X_docs = vec
        else:
            self.X_docs = vstack([self.X_docs, vec])

    def upsert_docs(self, docs: List[Dict[str, Any]]):
        # docs: [{"text": "...", "metadata": {...}}, ...]
        for d in docs:
            text = (d.get("text") or "").strip()
            if not text:
                continue
            metadata = d.get("metadata") or {}
            self._append_doc(text, metadata)

    def teach(self, text: str, label: str):
        # Train online classifier and add as retrievable doc
        x = self._to_vec([text])
        if not self.clf_initialized:
            # initialize with current label set
            if label not in self.known_labels:
                self.known_labels.append(label)
            self.clf.partial_fit(x, [label], classes=np.array(self.known_labels))
            self.clf_initialized = True
        else:
            if label not in self.known_labels:
                # expand known labels
                self.known_labels.append(label)
                # Re-initialize minimally by calling partial_fit with classes
                self.clf.partial_fit(x, [label], classes=np.array(self.known_labels))
            else:
                self.clf.partial_fit(x, [label])

        # Also add to KB for retrieval with label metadata
        self._append_doc(text, {"label": label, "source": "taught"})

    # ---------- Reflexive Retrieval ----------
    def _paraphrases(self, text: str) -> List[str]:
        text_norm = " ".join(text.split()).strip()
        if not text_norm:
            return [text_norm]

        v1 = text_norm

        # Replace digits with # to smooth numeric mismatch
        v2 = re.sub(r"\d", "#", text_norm.lower())

        # Apply simple synonym mapping
        v3 = text_norm.lower()
        for pat, rep in self.syn_map.items():
            v3 = re.sub(pat, rep, v3)

        # Merge, deduplicate while keeping order
        seen = set()
        outs = []
        for v in [v1, v2, v3]:
            if v not in seen:
                outs.append(v)
                seen.add(v)
        return outs

    def _cosine_topk(self, q_vec: csr_matrix, k: int = 3) -> List[Dict[str, Any]]:
        if self.X_docs is None or self.X_docs.shape[0] == 0:
            return []
        # Because vectorizer norm='l2', cosine sim == dot product
        sims = (self.X_docs @ q_vec.T).toarray().ravel()
        top_idx = np.argsort(-sims)[:k]
        results = []
        for idx in top_idx:
            results.append({
                "doc_id": int(idx),
                "score": float(sims[idx]),
                "text": self.docs[idx].text,
                "metadata": self.docs[idx].metadata
            })
        return results

    def reflexive_retrieve(self, query: str, k: int = 3):
        variants = self._paraphrases(query)
        vote_counter: Dict[int, float] = {}
        best_per_variant = []
        for v in variants:
            qv = self._to_vec([v])
            top = self._cosine_topk(qv, k)
            best_per_variant.append(top)
            for i, hit in enumerate(top):
                # weight higher ranks more
                weight = 1.0 / (1 + i)
                vote_counter[hit["doc_id"]] = vote_counter.get(hit["doc_id"], 0.0) + weight

        # Aggregate consensus
        if not vote_counter:
            return {
                "results": [],
                "consensus_doc": None,
                "consensus_score": 0.0
            }

        # Choose doc with maximum votes (consensus), tie-break by highest base similarity
        consensus_doc = max(vote_counter.items(), key=lambda x: x[1])[0]
        base_sim = 0.0
        for top in best_per_variant:
            for hit in top:
                if hit["doc_id"] == consensus_doc:
                    base_sim = max(base_sim, hit["score"])

        return {
            "results": best_per_variant[0] if best_per_variant else [],
            "consensus_doc": consensus_doc,
            "consensus_score": float(base_sim)
        }

    # ---------- Arithmetic / Logic ----------
    def _contains_any(self, text: str, words: set) -> bool:
        t = text.lower()
        return any(w in t for w in words)

    def try_arithmetic(self, text: str) -> Optional[Dict[str, Any]]:
        t = text.lower()
        nums = re.findall(r"[-+]?\d*\.?\d+", t)
        numbers = [float(x) for x in nums] if nums else []

        # count words intent
        if self._contains_any(t, self.ops["count_words"]):
            wc = len(re.findall(r"\w+", text))
            return {
                "answer": f"{wc}",
                "detail": {"operation": "count_words", "words": wc},
                "confidence": 0.85
            }

        if not numbers:
            return None

        def seq_reduce(op: str, values: List[float]) -> float:
            if op == "sum":
                return float(np.sum(values))
            if op == "sub":
                return float(np.subtract.reduce(values)) if len(values) > 1 else float(values[0])
            if op == "mul":
                return float(np.prod(values))
            if op == "div":
                if len(values) == 1:
                    return float(values[0])
                denom = float(np.prod(values[1:]))
                return float(values[0] / denom) if denom != 0 else float("inf")
            if op == "avg":
                return float(np.mean(values))
            if op == "max":
                return float(np.max(values))
            if op == "min":
                return float(np.min(values))
            return float("nan")

        op_detected = None
        for op_name, lex in self.ops.items():
            if op_name == "count_words":
                continue
            if self._contains_any(t, lex):
                op_detected = op_name
                break

        if op_detected:
            result = seq_reduce(op_detected, numbers)
            conf = 0.9 if len(numbers) >= 2 else 0.7
            return {
                "answer": f"{result}",
                "detail": {
                    "operation": op_detected,
                    "numbers": numbers
                },
                "confidence": conf
            }

        # If numbers exist but no op keyword, return None (not confident)
        return None

    # ---------- Classification ----------
    def classify(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.clf_initialized or not self.known_labels:
            return None
        x = self._to_vec([text])
        pred = self.clf.predict(x)[0]
        # Use decision_function margin for confidence -> logistic transform
        margin = self.clf.decision_function(x)
        if isinstance(margin, np.ndarray):
            if margin.ndim == 1:
                m = float(abs(margin[0]))
            else:
                # multiclass: take top-vs-second difference
                margins = margin.ravel()
                top2 = np.sort(margins)[-2:] if margins.size >= 2 else margins
                m = float(top2[-1] - (top2[-2] if len(top2) > 1 else 0.0))
        else:
            m = float(abs(margin))
        conf = 1.0 / (1.0 + math.exp(-m))
        return {
            "label": pred,
            "confidence": float(conf)
        }

    # ---------- Router ----------
    def answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {
                "strategy": "error",
                "answer": "Empty query.",
                "confidence": 0.0
            }

        # 1) Arithmetic branch
        ar = self.try_arithmetic(query)
        best = None
        if ar:
            best = {"strategy": "arithmetic", **ar}

        # 2) Reflexive retrieval
        rr = self.reflexive_retrieve(query, k=top_k)
        if rr["consensus_doc"] is not None:
            doc = self.docs[rr["consensus_doc"]]
            ret_ans = doc.text
            ret_conf = max(0.0, min(1.0, rr["consensus_score"]))
            cand = {
                "strategy": "retrieval",
                "answer": ret_ans,
                "confidence": ret_conf,
                "sources": [{"doc_id": rr["consensus_doc"], "metadata": doc.metadata}]
            }
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand

        # 3) Classification
        cl = self.classify(query)
        if cl:
            cand = {
                "strategy": "classification",
                "answer": f"Predicted label: {cl['label']}",
                "confidence": cl["confidence"]
            }
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand

        # 4) Fallback
        if best is None or best["confidence"] < 0.35:
            return {
                "strategy": "clarify",
                "answer": "I’m not confident yet. Please clarify your intent or provide examples via /teach or knowledge via /upsert_docs.",
                "confidence": float(best["confidence"] if best else 0.0)
            }

        return best


# -------------------- Web App --------------------
app = Flask(__name__)
service = ReflexiveAIService()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "RRAD Service",
        "version": "0.1.0",
        "description": "Reflexive Retrieval-Augmented Decisioning: neuro-symbolic routing + reflexive retrieval + online learning.",
        "endpoints": {
            "POST /ask": {"query": "text", "top_k": "optional int"},
            "POST /teach": {"text": "text", "label": "string"},
            "POST /upsert_docs": {"docs": [{"text": "text", "metadata": {...}}...]},
            "POST /reset": {},
            "GET /stats": {},
            "GET /healthz": {}
        }
    })


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "doc_count": len(service.docs),
        "labels": service.known_labels,
        "classifier_initialized": service.clf_initialized
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))
    try:
        result = service.answer(query, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/teach", methods=["POST"])
def teach():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    label = (data.get("label") or "").strip()
    if not text or not label:
        return jsonify({"error": "Both 'text' and 'label' are required."}), 400
    try:
        service.teach(text, label)
        return jsonify({"status": "ok", "taught": {"text": text, "label": label}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upsert_docs", methods=["POST"])
def upsert_docs():
    data = request.get_json(force=True, silent=True) or {}
    docs = data.get("docs", [])
    if not isinstance(docs, list) or not docs:
        return jsonify({"error": "'docs' must be a non-empty list."}), 400
    try:
        service.upsert_docs(docs)
        return jsonify({"status": "ok", "added": len(docs), "total_docs": len(service.docs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    # Reinitialize the service in-place
    try:
        global service
        service = ReflexiveAIService()
        return jsonify({"status": "ok", "message": "Service reset."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local testing; on Render use Gunicorn via Procfile
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)