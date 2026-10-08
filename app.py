from flask import Flask, jsonify, render_template, request
from src.generate import generateResponse
import configparser
import os

app = Flask(__name__)

# Load RAG config once at startup
_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
RAG_NAME   = _config.get('rag', 'name',   fallback='OpenRAG')
RAG_SOURCE = _config.get('rag', 'source', fallback='Unknown Source')


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    Request body (JSON): { "query": "your question here" }
    Response (JSON):     { "answer": "...", "rag": "...", "source": "..." }
    """
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Missing or empty 'query' field."}), 400

    try:
        answer = generateResponse(query)
        return jsonify({
            "answer": answer,
            "rag":    RAG_NAME,
            "source": RAG_SOURCE,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Liveness check — useful for Render health check config."""
    return jsonify({"status": "ok", "rag": RAG_NAME, "source": RAG_SOURCE})


# ── UI ────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        rag_name=RAG_NAME,
        rag_source=RAG_SOURCE,
    )


if __name__ == "__main__":
    app.run(debug=False)
