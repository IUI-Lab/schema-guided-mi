# -*- coding: utf-8 -*-

"""
Dialogue-State Similarity Search Server

Flask server that, given a free-text query, returns the top-N most similar
dialogue-state instances (looked up via cosine similarity over a precomputed
embedding DB) together with their annotated Src-Tgt policy instances.

Command: uv run python -m src.find_near_state.find_near_DS_flask_server \
    --jsonl_dir /app/dataset/instance_jsonl/AnnoMI/policy/DCFrame_support_dataset/ \
    --embeddings_pkl /app/src/find_near_state/embedding_DB/AnnoMI_inst919_DSembeddings.pkl \
    --tokenizer_path /app/models/tokenizers/tokenizer_xlmrLong \
    --model_path /app/models/origin_pretrained/xlmrLong \
    --device cuda:0 \
    --host 127.0.0.1 --port 10800
"""

import json
import logging
import logging.config
from pathlib import Path

from flask import Flask, request, jsonify  # pip install Flask

from .find_near_dialogue_state_controller import FindNearDialogueStateController

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

FindNearDSCtr = FindNearDialogueStateController()


# --------------------API-------------------------
@app.route('/')
def index():
    return "<p>Hello world.</p>"


@app.route('/get_similar', methods=['POST'])
def get_similar_json():
    """Receives a JSON query, returns the top-N similar dialogue states as JSON.

    Returns:
        `List` : [{'uid':str, 'similarity_score':float, 'DS_instance':dict}, ...]
    """
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400

    params_dict = json.loads(request.json)
    logging.debug('[SV] params_dict: {}'.format(params_dict))
    if params_dict.get('query_str', None) is None:
        return jsonify({'error': True, 'msg': 'getSimilar_No_query_str'}), 400
    if params_dict.get('top_n', None) is None:
        return jsonify({'error': True, 'msg': 'getSimilar_No_top_n'}), 400

    try:
        similar_result_list = FindNearDSCtr.get_similarity_dialogue_state(params_dict['query_str'], params_dict['top_n'])
        return jsonify({'similar_ret': similar_result_list})
    except Exception as e:
        logging.error('Exception: {}'.format(e))
        return jsonify({'error': True, 'msg': str(e)}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Dialogue-state similarity search server.")
    parser.add_argument("--jsonl_dir", required=True, help="Directory of Src-Tgt policy instance jsonl files")
    parser.add_argument("--embeddings_pkl", required=True, help="Path to the precomputed DS embeddings pickle")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer directory")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model directory")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 / cpu (default: cuda:0). Use cuda:1, cuda:2, ... to pick among multiple GPUs")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, i.e. not reachable from outside the container)")
    parser.add_argument("--port", type=int, default=10800, help="Server port (default: 10800)")
    args = parser.parse_args()

    logging.config.fileConfig(PROJECT_ROOT / "logging.conf")

    FindNearDSCtr.set_DS_DB_using_SrcTgtPolicyInstances(args.jsonl_dir)
    FindNearDSCtr.set_DSembeddingMatrix_from_pickle(args.embeddings_pkl)
    FindNearDSCtr.set_embedding_model(args.tokenizer_path, args.model_path, args.device)

    app.run(host=args.host, port=args.port, threaded=True, debug=True)
