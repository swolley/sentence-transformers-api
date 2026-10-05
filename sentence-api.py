import math
import os

from flask import Flask, jsonify, request
from sentence_transformers import CrossEncoder, SentenceTransformer

app = Flask(__name__)

# Default model, and how many models to keep resident. The Laraplate client
# sends the active profile's service_model in each request, so this default is
# only used when a request omits "model".
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
MODEL_CACHE = max(1, int(os.environ.get("EMBEDDING_MODEL_CACHE", "2")))
PORT = int(os.environ.get("EMBEDDING_PORT", "8000"))

# Cross-encoder reranking (POST /score). As with /embed, the client may name the
# model in the request and this is the default. A cross-encoder is another kind
# of model than the embedding one and must be multilingual for multilingual
# content, hence a multilingual MiniLM. Setting CROSS_ENCODER_MODEL to an empty
# string turns the default off: the host then does not rerank and answers 503
# unless the request names a model.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_MODEL = os.environ.get("CROSS_ENCODER_MODEL", DEFAULT_RERANK_MODEL) or None
RERANK_MODEL_CACHE = max(1, int(os.environ.get("CROSS_ENCODER_MODEL_CACHE", "1")))
# Pairs are truncated to this many tokens: a long-context reranker would
# otherwise default to its whole window, which is slow on CPU.
RERANK_MAX_LENGTH = max(1, int(os.environ.get("CROSS_ENCODER_MAX_LENGTH", "512")))
# The Laraplate client sends at most 64 pairs per request.
RERANK_MAX_PAIRS = max(1, int(os.environ.get("CROSS_ENCODER_MAX_PAIRS", "64")))

# Per-family input prefixes. Some embedding models are trained to distinguish a
# search query from an indexed passage and require a textual prefix to embed
# them into the right space; others (MiniLM, mpnet, gte, bge-m3, ...) want none.
#
# The client stays model-agnostic: it always sends the *semantic* intent via
# "input_type" ("query" or "passage"), and the server translates it into the
# right prefix for the resolved model. To support a new family, add an entry
# here — nothing changes client-side.
#
# Matching is by case-insensitive substring on the model name; the first match
# wins, so order more specific keys before more generic ones.
MODEL_PREFIXES = {
    "e5": {"query": "query: ", "passage": "passage: "},
    "nomic": {"query": "search_query: ", "passage": "search_document: "},
    # BGE is intentionally omitted: v1.5 wants an instruction only on queries
    # while bge-m3 wants nothing, so a blanket rule would be wrong. Add a
    # specific entry (e.g. "bge-large") if/when you actually use one.
}

# name -> SentenceTransformer, ordered oldest..newest (simple LRU by re-insert).
_models = {}


def load_model(name):
    name = name or DEFAULT_MODEL
    model = _models.pop(name, None)
    if model is None:
        model = SentenceTransformer(name)
    _models[name] = model  # mark as most-recently used
    while len(_models) > MODEL_CACHE:
        _models.pop(next(iter(_models)))
    return name, model


def prefix_scheme(model_name):
    """Return the {query, passage} prefix dict for a model, or None."""
    lower = model_name.lower()
    for key, scheme in MODEL_PREFIXES.items():
        if key in lower:
            return scheme
    return None


def apply_prefix(texts, model_name, input_type):
    """Ensure each text carries the family prefix for input_type (idempotent).

    The client tells the server the *semantic* intent via input_type; the server
    knows how each model family expects it. If a text already starts with the
    expected prefix (e.g. the client pre-prefixed it, as the Laraplate config
    does), it is kept as-is; otherwise the prefix is prepended. This makes the
    endpoint safe for both "dumb" clients and clients that already prefix, with
    no risk of double prefixes ("query: query: ...").

    No-op (returns texts unchanged) when input_type is omitted or the model has
    no prefix convention.

    Returns (texts, added) where added is True if a prefix was added to at least
    one text.

    Note: detection is a startswith() check, so a passage that literally begins
    with "query: " would be treated as already-prefixed. Acceptable in practice.
    """
    if not input_type:
        return texts, False
    scheme = prefix_scheme(model_name)
    if not scheme or input_type not in scheme:
        return texts, False
    prefix = scheme[input_type]
    added = False
    out = []
    for t in texts:
        if t.startswith(prefix):
            out.append(t)
        else:
            out.append(prefix + t)
            added = True
    return out, added


# name -> CrossEncoder, same LRU scheme as the embedding models.
_rerankers = {}


def load_reranker(name):
    """Return (name, CrossEncoder), or (None, None) when no model is configured."""
    name = name or RERANK_MODEL
    if not name:
        return None, None
    model = _rerankers.pop(name, None)
    if model is None:
        model = CrossEncoder(name, max_length=RERANK_MAX_LENGTH)
    _rerankers[name] = model  # mark as most-recently used
    while len(_rerankers) > RERANK_MODEL_CACHE:
        _rerankers.pop(next(iter(_rerankers)))
    return name, model


def reranker_info():
    return {
        "default_model": RERANK_MODEL,
        "loaded_models": list(_rerankers.keys()),
        "model_cache": RERANK_MODEL_CACHE,
        "max_length": RERANK_MAX_LENGTH,
        "max_pairs": RERANK_MAX_PAIRS,
    }


def validate_pairs(pairs):
    """Return an error message for a malformed `pairs` value, or None."""
    if not isinstance(pairs, list):
        return 'pairs must be a list of {"query": str, "text": str}'
    if len(pairs) > RERANK_MAX_PAIRS:
        return f"too many pairs: {len(pairs)} (max {RERANK_MAX_PAIRS})"
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, dict)
            or not isinstance(pair.get("query"), str)
            or not isinstance(pair.get("text"), str)
        ):
            return f'pairs[{index}] must be {{"query": str, "text": str}}'
    return None


def score_pairs(model, pairs):
    """One relevance score in [0, 1] per pair, in the order received.

    The sigmoid is passed explicitly. With activation_fn=None the library falls
    back to the model's own default, which is a sigmoid for some checkpoints and
    raw logits for others (the MS MARCO ones). The Laraplate client clamps
    scores to [0, 1], so raw logits would collapse into ties and lose the order.
    """
    from torch import nn

    raw = model.predict(
        [(pair["query"], pair["text"]) for pair in pairs],
        activation_fn=nn.Sigmoid(),
        convert_to_numpy=True,
    )
    if getattr(raw, "ndim", 1) != 1 or len(raw) != len(pairs):
        raise ValueError(
            "the model does not return one score per pair: it is not a single-score cross-encoder"
        )
    scores = [float(score) for score in raw]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("the model returned a non-finite score")
    return scores


# Warm the defaults so the first request is not a cold load. A model that cannot
# be loaded stops the process here instead of answering 500 to every request.
load_model(DEFAULT_MODEL)
if RERANK_MODEL:
    load_reranker(RERANK_MODEL)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": DEFAULT_MODEL,
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys()),
        "model_cache": MODEL_CACHE,
        "reranker": reranker_info(),
    })


@app.route("/models", methods=["GET"])
def models():
    """Discovery endpoint: what the client can ask for and how input_type maps.

    The service loads any model name on demand, so this does not enumerate a
    fixed allow-list; it exposes the default, what is currently resident, and
    the input_type prefix families so the client knows when input_type matters.
    """
    return jsonify({
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys()),
        "model_cache": MODEL_CACHE,
        "input_types": ["query", "passage"],
        "prefix_families": MODEL_PREFIXES,
        "reranker": reranker_info(),
    })


@app.route("/embed", methods=["POST"])
def embed():
    try:
        data = request.get_json(silent=True) or {}

        if "texts" in data:
            texts = data["texts"]
        elif "text" in data:
            texts = [data["text"]]
        else:
            return jsonify({"error": "No text or texts provided"}), 400

        input_type = data.get("input_type")
        if input_type is not None and input_type not in ("query", "passage"):
            return jsonify({
                "error": "input_type must be 'query' or 'passage'",
            }), 400

        name, model = load_model(data.get("model"))
        texts, prefix_applied = apply_prefix(texts, name, input_type)
        normalize = bool(data.get("normalize_embeddings", True))
        embeddings = model.encode(texts, normalize_embeddings=normalize)

        return jsonify({
            "model": name,
            "input_type": input_type,
            "prefix_applied": prefix_applied,
            "embeddings": [embedding.tolist() for embedding in embeddings],
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/score", methods=["POST"])
def score():
    """Relevance score in [0, 1] for each {query, text} pair, from a cross-encoder."""
    try:
        data = request.get_json(silent=True) or {}
        pairs = data.get("pairs")

        error = validate_pairs(pairs)
        if error:
            return jsonify({"error": error}), 400

        requested = data.get("model")
        if requested is not None and (not isinstance(requested, str) or requested == ""):
            return jsonify({"error": "model must be a non-empty string"}), 400

        name, model = load_reranker(requested)
        if model is None:
            return jsonify({
                "error": 'No cross-encoder model configured: set CROSS_ENCODER_MODEL or send "model"',
            }), 503

        return jsonify({
            "model": name,
            "scores": score_pairs(model, pairs) if pairs else [],
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
