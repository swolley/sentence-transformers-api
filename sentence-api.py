import os

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Default model, and how many models to keep resident. The Laraplate client
# sends the active profile's service_model in each request, so this default is
# only used when a request omits "model".
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
MODEL_CACHE = max(1, int(os.environ.get("EMBEDDING_MODEL_CACHE", "2")))

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

        name, model = load_model(data.get("model"))
        normalize = bool(data.get("normalize_embeddings", True))
        embeddings = model.encode(texts, normalize_embeddings=normalize)

        return jsonify({
            "model": name,
            "embeddings": [embedding.tolist() for embedding in embeddings],
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
