# -*- coding: utf-8 -*-

# Build pre-datasets for policy learning from MI dialogue JSON files.

import os, sys
from pathlib import Path
import logging, logging.config
import json
import functools
import copy


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MakeInstanceDataset:
    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Utterance-list helpers
    # ------------------------------------------------------------------
    def concatenete_utterance(self, combined_list: list, join_uttr_str: str = "") -> dict:
        """Concatenate consecutive utterances from the same speaker.

        Args:
            combined_list: Consecutive utterances by one speaker.
                           Each element: {"speaker": ..., "utterance": ..., ...}
            join_uttr_str: String inserted between utterances when joining.

        Returns:
            {"speaker": <str>, "concat_utterance": <str>}
        """
        concat_utterance = join_uttr_str.join([u["utterance"] for u in combined_list])
        return {"speaker": combined_list[0]["speaker"], "concat_utterance": concat_utterance}

    def get_n_context_source_target_pair_for_prediction_1dialogue(
        self,
        uttr_msg_list: list,
        is_target_key: str,
        is_target_key_value: str,
        n_context_window: int = 1,
    ) -> list:
        """Build (source, target) pairs for prediction from a single dialogue.

        For each utterance that matches the target condition, collect the
        preceding n utterances as context (source).

        Args:
            uttr_msg_list: Utterance list for one dialogue session.
            is_target_key: Dict key used to identify the target utterance.
            is_target_key_value: Expected value of is_target_key for targets.
            n_context_window: Number of preceding utterances to use as context.

        Returns:
            List of dicts:
                [{"target": <uttr_dict>,
                  "source": {"n_contexts": [<oldest>, ..., <newest>]}}, ...]
            Context entries are None when no utterance exists at that position.
        """
        src_tgt_pair_list = []
        for idx, uttr_dict in enumerate(uttr_msg_list):
            if uttr_dict[is_target_key] != is_target_key_value:
                continue

            _pair = {"target": uttr_dict}
            _pair["source"] = {
                "n_contexts": [
                    uttr_msg_list[i] if i >= 0 else None
                    for i in range(idx - n_context_window, idx)
                ]
            }
            src_tgt_pair_list.append(copy.deepcopy(_pair))

        return src_tgt_pair_list


class MakePolicyLearnDataset:
    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)
        self.MakeInstanceDataset = MakeInstanceDataset()

    def read_utterance_list_from_json(self, json_path: str) -> list:
        """Load a session JSON file and return its utterance list.

        Args:
            json_path: Path to the JSON file.

        Returns:
            List of utterance dicts: [{"speaker": ..., "utterance": ..., ...}, ...]
        """
        with open(json_path, "r") as f:
            return json.load(f)

    def mapfunc_concatenete_same_speaker_info(
        self, same_speaker_uttr_list: list, join_uttr_str: str = ""
    ) -> dict:
        """Aggregate a run of same-speaker utterances into a single record.

        Args:
            same_speaker_uttr_list: Consecutive utterances by one speaker.
                                    Each element must have keys:
                                    "speaker", "utterance", "csv_idx", "category".
            join_uttr_str: String inserted between utterances when joining.

        Returns:
            {"speaker": ...,
             "concat_utterance": ...,
             "csv_idx_list": [old -> new],
             "category_list": [old -> new]}
        """
        concat_utterance = join_uttr_str.join(
            [u["utterance"] for u in same_speaker_uttr_list]
        )
        csv_idx_list = [u["csv_idx"] for u in same_speaker_uttr_list]
        categories = [int(u["category"]) for u in same_speaker_uttr_list]
        return {
            "speaker": same_speaker_uttr_list[0]["speaker"],
            "concat_utterance": concat_utterance,
            "csv_idx_list": csv_idx_list,
            "category_list": categories,
        }

    def write2jsonl(self, obj_list: list, out_jsonl_path: str) -> None:
        """Write a list of objects to a JSON-Lines file (one object per line)."""
        with open(out_jsonl_path, "w") as f:
            for obj in obj_list:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")

    def make_policy_learn_instance_1session(
        self,
        session_json_path: str,
        out_json_path: str,
        n_context_num: int,
        concat_uttr_str: str,
    ) -> None:
        """Build policy-learning instances for a single session (target: Counselor).

        Args:
            session_json_path: Input session JSON file.
            out_json_path: Output JSONL file path.
            n_context_num: Number of context utterances to include.
            concat_uttr_str: String used to join consecutive same-speaker utterances.
        """
        session_uttr_list = self.read_utterance_list_from_json(session_json_path)

        concat_list = list(
            map(
                functools.partial(
                    self.mapfunc_concatenete_same_speaker_info,
                    join_uttr_str=concat_uttr_str,
                ),
                session_uttr_list,
            )
        )
        print(f"concat_uttr_list len: {len(concat_list)}")

        src_tgt_pair_list = (
            self.MakeInstanceDataset.get_n_context_source_target_pair_for_prediction_1dialogue(
                concat_list,
                is_target_key="speaker",
                is_target_key_value="Counselor",
                n_context_window=n_context_num,
            )
        )
        print(f"src_tgt_pair_list len: {len(src_tgt_pair_list)}")

        self.write2jsonl(src_tgt_pair_list, out_json_path)
        print(f"wrote: {out_json_path}")

    def make_extractDCFrame_PreDataset_supporting_policy_learn(
        self,
        session_json_path: str,
        out_json_path: str,
        n_context_num: int,
        concat_uttr_str: str,
    ) -> None:
        """Build DCFrame extraction pre-dataset for a single session (target: Client).

        Policy learning requires DCFrame labels. This function creates the
        pre-dataset used to extract DCFrame annotations for Client utterances.

        Args:
            session_json_path: Input session JSON file.
            out_json_path: Output JSONL file path.
            n_context_num: Number of context utterances to include.
            concat_uttr_str: String used to join consecutive same-speaker utterances.
        """
        session_uttr_list = self.read_utterance_list_from_json(session_json_path)

        concat_list = list(
            map(
                functools.partial(
                    self.mapfunc_concatenete_same_speaker_info,
                    join_uttr_str=concat_uttr_str,
                ),
                session_uttr_list,
            )
        )
        print(f"concat_uttr_list len: {len(concat_list)}")

        src_tgt_pair_list = (
            self.MakeInstanceDataset.get_n_context_source_target_pair_for_prediction_1dialogue(
                concat_list,
                is_target_key="speaker",
                is_target_key_value="Client",
                n_context_window=n_context_num,
            )
        )
        print(f"src_tgt_pair_list len: {len(src_tgt_pair_list)}")

        self.write2jsonl(src_tgt_pair_list, out_json_path)
        print(f"wrote: {out_json_path}")


def get_json_files(jsonl_dir: str) -> list:
    """Return a list of *.json file names found in jsonl_dir."""
    return [f for f in os.listdir(jsonl_dir) if f.endswith(".json")]


# ==============================================================================
# Data 1 — Policy-learning dataset  (target: Counselor)
# ==============================================================================
def make_policy_dataset_co_tgt_multiSession() -> None:
    """Process all sessions in a directory; write Counselor-target policy instances."""

    # --- USER CONFIG ----------------------------------------------------------
    N_CONTEXT_NUM = 5       # number of preceding turns used as context
    CONCAT_UTTR_STR = " "  # [En] separator for joining same-speaker consecutive turns

    SESSION_JSON_DIR = PROJECT_ROOT / "dataset/json/AnnoMI"
    SESSION_NUM_IDX  = 0        # index in filename.split("_") for the session number
    OUT_JSONL_DIR = PROJECT_ROOT / "dataset/instance_jsonl/AnnoMI/policy/TgtCo"
    # --------------------------------------------------------------------------

    jsonl_files = [
        os.path.join(SESSION_JSON_DIR, fname)
        for fname in get_json_files(SESSION_JSON_DIR)
    ]

    for session_json_path in jsonl_files:
        print(f"processing: {session_json_path}")
        basename = os.path.splitext(os.path.basename(session_json_path))[0]
        session_num = basename.split("_")[SESSION_NUM_IDX]  # e.g. "s01" from "ja_MI_Nakano22_s01_..."

        out_jsonl_path = (
            f"{OUT_JSONL_DIR}/policyInst_TgtCoCtx{N_CONTEXT_NUM}_session_{session_num}.jsonl"
        )
        print(f"{session_num=}, {out_jsonl_path=}")

        dataset = MakePolicyLearnDataset()
        dataset.make_policy_learn_instance_1session(
            session_json_path, out_jsonl_path, N_CONTEXT_NUM, CONCAT_UTTR_STR
        )


# ==============================================================================
# Data 2 — DCFrame extraction pre-dataset  (target: Client)
# ==============================================================================
def make_extractDCFrame_PreDataset_supporting_policy_learn_multiSession() -> None:
    """Process all sessions; write Client-target DCFrame extraction pre-datasets."""

    # --- USER CONFIG ----------------------------------------------------------
    N_CONTEXT_NUM = 4       # number of preceding turns used as context
    CONCAT_UTTR_STR = " "  # [En] separator for joining same-speaker consecutive turns

    SESSION_JSON_DIR = PROJECT_ROOT / "dataset/json/AnnoMI"
    SESSION_NUM_IDX  = 0        # index in filename.split("_") for the session number
    OUT_JSONL_DIR = PROJECT_ROOT / "dataset/instance_jsonl/AnnoMI/policy/DCFrame_extracted_predata"
    # --------------------------------------------------------------------------

    jsonl_files = [
        os.path.join(SESSION_JSON_DIR, fname)
        for fname in get_json_files(SESSION_JSON_DIR)
    ]

    for session_json_path in jsonl_files:
        print(f"processing: {session_json_path}")
        basename = os.path.splitext(os.path.basename(session_json_path))[0]
        session_num = basename.split("_")[SESSION_NUM_IDX]

        out_jsonl_path = (
            f"{OUT_JSONL_DIR}/policySuportPreDCFrame_TgtClCtx{N_CONTEXT_NUM}_session_{session_num}.jsonl"
        )
        print(f"{session_num=}, {out_jsonl_path=}")

        dataset = MakePolicyLearnDataset()
        dataset.make_extractDCFrame_PreDataset_supporting_policy_learn(
            session_json_path, out_jsonl_path, N_CONTEXT_NUM, CONCAT_UTTR_STR
        )


if __name__ == "__main__":
    logging.config.fileConfig(PROJECT_ROOT / "logging.conf")

    # Task 1: Counselor-target policy-learning instances
    make_policy_dataset_co_tgt_multiSession()

    # Task 2: Client-target DCFrame extraction pre-dataset
    make_extractDCFrame_PreDataset_supporting_policy_learn_multiSession()
