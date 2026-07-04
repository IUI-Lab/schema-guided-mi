# -*- coding: utf-8 -*-

"""
Dialogue State Embedding Builder

Builds a BERT (XLM-R) [CLS] embedding for each Client dialogue-state instance
(history + DCFrame) found in a directory of Src-Tgt policy instance jsonl files,
keyed by ``<session_num>__<last_csv_idx>``, and saves the resulting embedding
dict to a pickle file.

Command: uv run python -m src.find_near_state.doc_embedding.create_embedding_controller \
    --src_tgt_policy_instances_dir /app/dataset/instance_jsonl/AnnoMI/policy/DCFrame_support_dataset/ \
    --tokenizer_path /app/models/tokenizers/tokenizer_xlmrLong \
    --model_path /app/models/origin_pretrained/xlmrLong \
    --device cuda:0 \
    --out_prefix AnnoMI
"""

import os
import json
import pickle
import time
import logging
import logging.config
from pathlib import Path

from .embedding_xlm_r_multilingual import EmbeddingXLMRMultilingual
from ..format_dialogue_state_utils import DialogueStateInformationFormatUtils

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CreateEmbeddingController(object):
    """Builds and stores BERT [CLS] embeddings for dialogue-state instances."""

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

        self.DS_DB = {}  # {uid `str` <session_num>__<last_csv_idx> : instance_dict}
        self.DS_embeddings = {}  # {uid `str` <session_num>__<last_csv_idx> : embedding}

        self.embeddingModelCtr = None
        self.FormatUtils = DialogueStateInformationFormatUtils()

    # ------- Set Model --------------------------------
    def set_model(self, tokenizer_path, model_path, device):
        ctr = EmbeddingXLMRMultilingual()
        ctr.set_model(tokenizer_path, model_path, device)
        self.embeddingModelCtr = ctr

    # ------- Set Dialogue State DB --------------------------------
    def set_DS_DB_using_SrcTgtPolicyInstances(self, src_tgt_policy_instances_dir: str):
        """Build ``DS_DB``: ``{uid `str` <session_num>__<last_csv_idx> : instance_dict}``"""
        jsonl_fnames = self.get_jsonl_files(src_tgt_policy_instances_dir)

        for jsonl_fname in jsonl_fnames:
            self.log.debug(f"reading jsonl: {jsonl_fname}")
            uid_inst_dict = {}  # {uid `str` <session_num>__<last_csv_idx> : instance_dict}

            filename_without_ext = Path(jsonl_fname).stem  # assumes ~~~_session_<SESSION_NUM>.jsonl
            session_num = filename_without_ext.split("_")[-1]
            jsonl_path = os.path.join(src_tgt_policy_instances_dir, jsonl_fname)
            src_tgt_policy_instances = self.get_instances_from_jsonl(jsonl_path)
            for inst_d in src_tgt_policy_instances:
                csv_idx_list = inst_d["target"].get("csv_idx_list", None)
                if csv_idx_list is None:
                    self.log.error(f"csv_idx_list is None. {inst_d['target']}")
                    raise ValueError(f"csv_idx_list missing in target: {inst_d['target']}")

                uid = f"{session_num}__{csv_idx_list[-1]}"
                uid_inst_dict[uid] = inst_d

            self.DS_DB.update(uid_inst_dict)  # NOTE: assumes uids never collide across files

        self.log.debug(f"DS_DB instance num: {len(self.DS_DB)}")
        self.log.debug(f"DS_DB keys: {list(self.DS_DB.keys())[:5]}")

    def get_jsonl_files(self, jsonl_dir: str):
        """Returns:
            list: jsonl filenames found in the directory
        """
        return [f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]

    def get_instances_from_jsonl(self, jsonl_path: str):
        """Returns:
            list: instances loaded from the jsonl file
        """
        with open(jsonl_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # ------- Build Embeddings --------------------------------
    def build_all_embeddings(self) -> None:
        """Compute a [CLS] embedding for every instance in ``DS_DB`` and store it in ``DS_embeddings``."""
        s_time = time.time()
        for idx, uid in enumerate(self.DS_DB.keys()):
            self.log.debug(f"{idx + 1}/{len(self.DS_DB)}")
            inst_d = self.DS_DB[uid]
            embed_target_str = self.FormatUtils.format_history_DCFrame(inst_d)
            embedding = self.embeddingModelCtr.get_cls_embedding_1sentence(embed_target_str)  # np.array (768,)
            self.DS_embeddings[uid] = embedding

        elapsed_time = time.time() - s_time
        self.log.info(f"built {len(self.DS_embeddings)} embeddings in {elapsed_time:.2f}sec")

    def save_embeddings(self, out_pkl_path) -> None:
        with open(out_pkl_path, "wb") as f:
            pickle.dump(self.DS_embeddings, f)
        self.log.info(f"saved! {out_pkl_path}")


def create_history_DCFrame_embeddings(
    src_tgt_policy_instances_dir: str,
    tokenizer_path: str,
    model_path: str,
    device: str,
    out_prefix: str,
    out_dir: str,
) -> None:
    """Build history+DCFrame [CLS] embeddings for all Src-Tgt policy instances and save them to a pickle file."""
    ctr = CreateEmbeddingController()
    ctr.set_DS_DB_using_SrcTgtPolicyInstances(src_tgt_policy_instances_dir)
    ctr.set_model(tokenizer_path, model_path, device)

    ctr.build_all_embeddings()

    inst_num = len(ctr.DS_embeddings)
    out_pkl_path = Path(out_dir) / f"{out_prefix}_inst{inst_num}_DSembeddings.pkl"
    ctr.save_embeddings(out_pkl_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Text Encoder [CLS] embeddings for dialogue-state instances.")
    parser.add_argument("--src_tgt_policy_instances_dir", required=True, help="Directory containing Src-Tgt policy instance jsonl files")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer directory")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model directory")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--out_prefix", required=True, help="Prefix for the output pickle filename")
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "src" / "find_near_state" / "embedding_DB"), help="Output directory for the embeddings pickle")
    args = parser.parse_args()

    logging.config.fileConfig(PROJECT_ROOT / "logging.conf")
    create_history_DCFrame_embeddings(
        src_tgt_policy_instances_dir=args.src_tgt_policy_instances_dir,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
        device=args.device,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir,
    )
