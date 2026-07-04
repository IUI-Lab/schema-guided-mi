# -*- coding: utf-8 -*-

# Generates pseudo-policies for policy learning.
# Ground-truth intent labels are NOT used during pseudo-policy generation.

import logging
import logging.config
import json
import time
import copy
from pathlib import Path

from ..GPTs.openAI_utils import OpenAIUtils

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GeneratePseudoPolicyController(object):
    def __init__(self, model_name) -> None:
        self.log = logging.getLogger(__name__)
        self.model_name = model_name
        self.openai_utils = OpenAIUtils(model_name)
        self.log.info(f"Use OpenAI model: {model_name}")

        self.policy_schema = None
        self.sys_prompt = None

    def set_policy_schema(self, policy_schema):
        self.policy_schema = policy_schema
        self.log.debug(f"set policy_schema: {self.policy_schema.model_json_schema()}")

    def set_sys_prompt(self, sys_prompt):
        self.sys_prompt = sys_prompt
        self.log.info(f"set sys_prompt: {self.sys_prompt}")

    def _get_all_dict_in_jsonl(self, jsonl_path: str):
        loaded_list = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for _line_dict in f:
                loaded_list.append(json.loads(_line_dict.strip()))
        return loaded_list

    # Build Client DCFrame DB
    # --------------------------------------------------------------------------
    def make_Cl_DCFrameDB(self, cl_policySupportDCFrame_jsonl_path: str):
        """Build a DCFrame DB indexed by the last Client utterance CSV index for one session.

        Each instance is a Target:Client+DCFrame pair.

        Returns:
            dict: ``{last_client_csv_idx: DCFrame_dict}``
        """
        Cl_DCFrameDB = {}
        with open(cl_policySupportDCFrame_jsonl_path, "r", encoding="utf-8") as f:
            for idx, _line_dict in enumerate(f):
                cl_DCFrame_inst = json.loads(_line_dict.strip())
                last_Client_csv_idx = cl_DCFrame_inst["target"]["csv_idx_list"][-1]
                DCFrame = cl_DCFrame_inst["DCFrame_out"]["DCFrame_result"]
                Cl_DCFrameDB[last_Client_csv_idx] = copy.deepcopy(DCFrame)
        return Cl_DCFrameDB

    # Single instance
    # --------------------------------------------------------------------------
    def _make_history_str(self, src_tgt_dict: dict):
        """Format dialogue history as ``Role1: Utterance1\\nRole2: Utterance2\\n...``"""
        src_uttr_list = []
        for src_uttr_dict in src_tgt_dict["source"]["n_contexts"]:
            if src_uttr_dict is None:
                continue
            src_uttr_list.append(f"{src_uttr_dict['speaker']}: {src_uttr_dict['concat_utterance']}")
        return "\n".join(src_uttr_list)

    def _make_user_prompt(self, history: str, DCFrame: str, tgt_co_uttr: str):
        return (
            "### History\n"
            f"{history}\n\n"
            "### Dialogue_content\n"
            f"{DCFrame}\n\n"
            "### Counselor_response\n"
            f"{tgt_co_uttr}\n"
        )

    def get_paired_Cl_DCFrame(self, Cl_DCFrameDB, tgt_instance):
        """Retrieve the DCFrame that preceded the target Counselor utterance.

        Looks up the last Client utterance in the source context to find the
        matching DCFrame from ``Cl_DCFrameDB``.

        Returns:
            dict: DCFrame dict, or empty dict if the session starts with a Counselor turn.
        """
        last_Cl_uttr_dict = tgt_instance["source"]["n_contexts"][-1]
        if last_Cl_uttr_dict is None:
            # Session begins with a Counselor turn — no preceding Client DCFrame exists
            last_Cl_csv_idx = "begin"
        else:
            last_Cl_csv_idx = last_Cl_uttr_dict["csv_idx_list"][-1]

        DCFrame = {} if last_Cl_csv_idx == "begin" else Cl_DCFrameDB[last_Cl_csv_idx]
        self.log.debug(f"last_Cl_csv_idx: {last_Cl_csv_idx}, DCFrame: {DCFrame}")
        return DCFrame

    def get_policy_1instance(self, Cl_DCFrame, todo_policyInstance):
        """Generate a pseudo-policy for one Counselor utterance using OpenAI Structured Outputs.

        Args:
            Cl_DCFrame (dict): DCFrame extracted from the immediately preceding Client utterance.
            todo_policyInstance (dict): Source-target pair instance to process.

        Returns:
            dict: ``{'response': {'focuses': [...], 'intent': str,
                                  'seek_attribute': str|None, 'seek_frame_type': str|None},
                     'usage': dict, 'elapse_time': float}``
        """
        tgt_uttr = f"{todo_policyInstance['target']['speaker']}: {todo_policyInstance['target']['concat_utterance']}"
        history = self._make_history_str(todo_policyInstance)
        user_ctx = self._make_user_prompt(history, str(Cl_DCFrame), tgt_uttr)
        self.log.debug(f"user_ctx: {user_ctx}")

        policy_d = self.openai_utils.get_1time_ChatCompletion_StructuredOutput_dict(
            self.sys_prompt, user_ctx, out_schema=self.policy_schema, temperature=0.0
        )
        return policy_d

    # Single session
    # --------------------------------------------------------------------------
    def get_policy_1session(self, todo_policyInstance_jsonl_path, ExDCFrameSupport_jsonl_path, out_jsonl_path):
        Cl_DCFrameDB = self.make_Cl_DCFrameDB(ExDCFrameSupport_jsonl_path)
        todo_policyInstances = self._get_all_dict_in_jsonl(todo_policyInstance_jsonl_path)
        of = open(out_jsonl_path, "w", encoding="utf-8")

        s_time = time.time()
        for idx, src_tgt_d in enumerate(todo_policyInstances):
            previous_cl_DCFrame = self.get_paired_Cl_DCFrame(Cl_DCFrameDB, src_tgt_d)
            policy_d = self.get_policy_1instance(previous_cl_DCFrame, src_tgt_d)
            self.log.debug(f"policy_d: {policy_d}")

            updated_src_tgt_d = copy.deepcopy(src_tgt_d)
            updated_src_tgt_d["DCFrame"] = copy.deepcopy(previous_cl_DCFrame)
            updated_src_tgt_d["out_policy"] = policy_d

            of.write(f"{json.dumps(updated_src_tgt_d, ensure_ascii=False)}\n")

        elapsed_time = round((time.time() - s_time), 3)
        self.log.debug(f"out_jsonl_path: {out_jsonl_path}, elapsed_time: {elapsed_time}")
        of.close()

    # Inspection helpers
    # --------------------------------------------------------------------------
    def map_co_categoryID2Label(self, co_label: int):
        """ [Diet-MI] """
        co_categoryID2Label = {
            1: "Question", 2: "Question",
            3: "Affirmation", 4: "Affirmation", 5: "Affirmation", 6: "Affirmation",
            7: "Reflection", 8: "Reflection", 9: "Reflection", 10: "Reflection",
            11: "Reflection", 12: "Reflection", 13: "Reflection", 14: "Reflection",
            15: "Summarization", 16: "Summarization",
            17: "Other", 18: "Other", 19: "Other", 20: "Other", 21: "Other",
            22: "Other", 23: "Other", 24: "Other", 25: "Other",
        }
        label = co_categoryID2Label.get(co_label, None)
        if label is None:
            raise ValueError(f"Invalid co_label: {co_label}")
        return label

    def _print_1psedu_policy_instance(self, src_tgt_d, use_GT_intent):
        """Print a single pseudo-policy instance. Null values appear as None."""
        history = self._make_history_str(src_tgt_d)
        DCFrame = src_tgt_d["DCFrame"]
        tgt_co_category = src_tgt_d["target"]["category_list"]
        last_co_category = tgt_co_category[-1]

        policy_d = src_tgt_d["out_policy"]["response"]
        if use_GT_intent is False:
            intent = policy_d.get("intent")
        else:
            intent = self.map_co_categoryID2Label(last_co_category) # [Diet-MI] case

        focuses = policy_d.get("focuses", [])
        seek_frame_type = policy_d.get("seek_frame_type")
        seek_attribute = policy_d.get("seek_attribute")

        print(
            f"### History\n{history}\n\n"
            f"### Dialogue_content\n{DCFrame}\n\n"
            f"### Intent\n{intent}\n\n"
            f"### Focuses\n{focuses}\n\n"
            f"### Seek_frame_type\n{seek_frame_type}\n\n"
            f"### Seek_attribute\n{seek_attribute}\n"
        )
        print("-----------------------------------")
        print(
            f"{src_tgt_d['target']['speaker']}: "
            f"cat: {src_tgt_d['target']['category_list']} "
            f"{src_tgt_d['target']['concat_utterance']}"
        )

    def show_pseudo_policy_result(self, pseudo_policy_jsonl_path, use_GT_intent):
        print(f"\n====== {use_GT_intent=} ======\n")
        src_tgt_list = self._get_all_dict_in_jsonl(pseudo_policy_jsonl_path)
        while True:
            jsonl_line_num = input("input jsonl line num >")
            if jsonl_line_num == "q":
                break
            tgt_instance = src_tgt_list[int(jsonl_line_num) - 1]
            self._print_1psedu_policy_instance(tgt_instance, use_GT_intent)


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_multisession_pseudo_policy():
    """Generate pseudo-policies for multiple sessions."""
    model_name = "gpt-4o-2024-08-06"

    session_nums = [36]
    # session_nums = [36, 64, 100, 121, 133]
    todo_policyInstance_jsonl_dir = PROJECT_ROOT / "dataset/instance_jsonl/AnnoMI/policy/TgtCo"
    supportDCFrame_jsonl_dir = PROJECT_ROOT / "dataset/instance_jsonl/AnnoMI/policy/DCFrame_support_dataset"
    pseudo_policy_out_jsonl_dir = PROJECT_ROOT / "dataset/instance_jsonl/AnnoMI/policy/pseudo_policy_outdata"

    out_fname_temp = "pseudoPolicy_woIntent_{todo_fname}"
    from ..gen_SemanticStrategy.counselor_policy_schema import CounselorPolicyWithoutIntent as CounselorPolicy
    gen_policy_sys_prompt_path = PROJECT_ROOT / "src/decide_policy/sys_prompt_gen_pseudo_policy_woIntent_en.txt"

    GenPseudoPolicyCtr = GeneratePseudoPolicyController(model_name)
    GenPseudoPolicyCtr.set_policy_schema(CounselorPolicy)
    GenPseudoPolicyCtr.set_sys_prompt(read_txt(gen_policy_sys_prompt_path))

    for todo_session_num in session_nums:
        todo_fname = f"policyInst_TgtCoCtx5_session_{todo_session_num}.jsonl"
        todo_policyInstance_jsonl_path = todo_policyInstance_jsonl_dir / todo_fname

        support_fname = f"policySupportDCFrame_TgtClCtx4_session_{todo_session_num}.jsonl"
        supportDCFrame_jsonl_path = supportDCFrame_jsonl_dir / support_fname

        out_fname = out_fname_temp.format(todo_fname=todo_fname)
        pseudo_policy_out_jsonl_path = pseudo_policy_out_jsonl_dir / out_fname
        print(
            f"Processing session {todo_session_num}: "
            f"{todo_policyInstance_jsonl_path=}, "
            f"{supportDCFrame_jsonl_path=}, "
            f"{pseudo_policy_out_jsonl_path=}"
        )
        GenPseudoPolicyCtr.get_policy_1session(
            todo_policyInstance_jsonl_path, supportDCFrame_jsonl_path, pseudo_policy_out_jsonl_path
        )

    print("done")


def see_result():
    model_name = "gpt4o-2024-08-06"
    pseudo_policy_jsonl_path = PROJECT_ROOT / ".jsonl"
    use_GT_intent = False

    GenPseudoPolicyCtr = GeneratePseudoPolicyController(model_name)
    GenPseudoPolicyCtr.show_pseudo_policy_result(pseudo_policy_jsonl_path, use_GT_intent)


if __name__ == "__main__":
    logging.config.fileConfig(PROJECT_ROOT / "logging.conf")
    get_multisession_pseudo_policy()

    # see_result()