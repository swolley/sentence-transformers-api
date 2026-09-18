import os

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Default model, and how many models to keep resident. The Laraplate client
# sends the active profile's service_model in each request, so this default is
# only used when a request omits "model".
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
MODEL_CACHE = max(1, int(os.environ.get("EMBEDDING_MODEL_CACHE", "2")))

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


# Warm the default so the first embed request is not a cold load.
load_model(DEFAULT_MODEL)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": DEFAULT_MODEL,
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys()),
        "model_cache": MODEL_CACHE,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
