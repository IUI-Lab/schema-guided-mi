# -*- coding: utf-8 -*-


import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import json
import time
from datetime import datetime as dt
import traceback
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from ..extract_DCFrame.extract_frames_use_openAI_utils import ExtractFramesOpenAIUtils
from ..AddingDCFrame.DCFrame_add_update_controller import AddingDCFrameUpdateController


class MakeDCFrameSupportDatasetController(object):
    def __init__(self, model_name: str) -> None:
        self.log = logging.getLogger(__name__)
        if model_name == "":
            self.ExtractFramesOpenAIUtils = None
        else:
            self.ExtractFramesOpenAIUtils = ExtractFramesOpenAIUtils(model_name)

        self.AddingBaseDCFrameUpdate = AddingDCFrameUpdateController()

    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_all_dict_in_jsonl(self, jsonl_path: str):
        loaded_list = []
        with open(jsonl_path, 'r', encoding="utf-8") as f:
            for _line_dict in f:
                loaded_list.append(json.loads(_line_dict.strip()))
        return loaded_list

    def prepare_user_context1(self, src_tgt_dict, previous_DCFrame):
        """
        Args:
            src_tgt_dict: dict with keys:
                'target': utterance dict
                'source': {'n_contexts': [utterance dict with 'concat_utterance', ...]}
                          n_contexts is ordered from oldest to newest; entries may be None
            previous_DCFrame: dict or None; previous DCFrame result (empty dict if None)

        Returns:
            str: formatted user context string
        """
        tgt_uttr = f"{src_tgt_dict['target']['speaker']}: {src_tgt_dict['target']['concat_utterance']}"

        src_uttr_list = []
        for src_uttr_dict in src_tgt_dict['source']['n_contexts']:
            if src_uttr_dict is None:
                continue
            src_uttr_list.append(f"{src_uttr_dict['speaker']}: {src_uttr_dict['concat_utterance']}")
        src_uttr = '\n'.join(src_uttr_list)

        user_ctx = f"### Current_structure\n{previous_DCFrame}\n\n### Utterance\n{src_uttr}\n{tgt_uttr}\n\n### Updated_structure\n"
        return user_ctx

    def extractDCFrame_1jsonl_addBase(
        self,
        input_extract_DCFrame_jsonl_path: str,
        out_extract_DCFrame_jsonl_path: str,
        save_result_key: str,
        LLM_out_key: str,
        sys_prompt,
        examples: str,
        step1_temperature=0.0,
    ):
        """[Direct + AddBase]
        Extracts DCFrame for each src-tgt pair in a single-session jsonl and saves results.

        Args:
            input_extract_DCFrame_jsonl_path: str - input jsonl file path
            out_extract_DCFrame_jsonl_path: str - output jsonl file path;
                                                  AddBase error log is saved to the same directory
            save_result_key: str - key for storing the final DCFrame result
            LLM_out_key: str - key for storing raw LLM output
            sys_prompt: str
            examples: str
            step1_temperature: float - LLM sampling temperature
        """
        save_result_key_inside_key = 'DCFrame_result'
        sys_prompt = f"{sys_prompt}\n{examples}"

        loaded_extracted_DCFrame_list = self._get_all_dict_in_jsonl(input_extract_DCFrame_jsonl_path)

        of = open(out_extract_DCFrame_jsonl_path, 'w', encoding='utf-8')

        dir_path = os.path.dirname(out_extract_DCFrame_jsonl_path)
        file_name_without_ext = os.path.splitext(os.path.basename(out_extract_DCFrame_jsonl_path))[0]
        except_fname = f'exception_ExDCFrame_{file_name_without_ext}.err'
        except_of = open(os.path.join(dir_path, except_fname), 'w', encoding='utf-8')
        except_of.write(f'Exception log: deal with {input_extract_DCFrame_jsonl_path}\n')

        previous_out_DCFrame = None
        for line_num, src_tgt_dict in enumerate(loaded_extracted_DCFrame_list):
            self.log.info(f'processing... line_num: {line_num+1} / {len(loaded_extracted_DCFrame_list)}')

            if previous_out_DCFrame is None:
                previous_out_DCFrame = {}
            self.log.debug(f'--- previous_out_DCFrame ---\n{previous_out_DCFrame}')

            user_ctx = self.prepare_user_context1(src_tgt_dict, previous_out_DCFrame)

            self.log.debug(f'[query-LLM] --- sys_prompt\n{sys_prompt}\n--- user_ctx\n{user_ctx}')
            out_LLM = self.ExtractFramesOpenAIUtils.get_DCFrame(sys_prompt, user_ctx, temperature=step1_temperature)
            src_tgt_dict[LLM_out_key] = out_LLM  # {'function_res', 'usage', 'elapsed_time'}

            out_LLM_function_res = out_LLM['function_res']['arguments']
            if out_LLM_function_res == -1 or out_LLM_function_res == -2:
                # Failed to extract DCFrame; keep previous DCFrame
                self.log.warning(f'Failed to extract DCFrame line_num: {line_num+1} {out_LLM=} keep using old previous_out_DCFrame')
                src_tgt_dict[save_result_key] = {save_result_key_inside_key: copy.deepcopy(previous_out_DCFrame), 'err': 'extractDCFrame'}
                of.write(f'{json.dumps(copy.deepcopy(src_tgt_dict), ensure_ascii=False)}\n')
                continue

            # Successfully extracted DCFrame; apply AddBase
            try:
                _addBase_out_DCFrame = self.AddingBaseDCFrameUpdate.adding_base_update_DCFrame(previous_out_DCFrame, out_LLM_function_res)
            except Exception as e:
                self.log.warning(f'Failed to addBase DCFrame line_num: {line_num+1} {out_LLM_function_res=} keep using old previous_out_DCFrame')
                except_of.write(f'------------------------------- Error: {line_num=}\n')
                except_of.write(f'Error: {e}\n')
                except_of.write(f'Error: {traceback.format_exc()}\n')
                src_tgt_dict[save_result_key] = {save_result_key_inside_key: copy.deepcopy(previous_out_DCFrame), 'err': 'addBase'}
                of.write(f'{json.dumps(copy.deepcopy(src_tgt_dict), ensure_ascii=False)}\n')
                continue

            src_tgt_dict[save_result_key] = {save_result_key_inside_key: copy.deepcopy(_addBase_out_DCFrame), 'err': ''}
            previous_out_DCFrame = _addBase_out_DCFrame

            of.write(f'{json.dumps(copy.deepcopy(src_tgt_dict), ensure_ascii=False)}\n')

        of.close()
        self.log.info(f'Finish extracting DCFrame: {input_extract_DCFrame_jsonl_path}')


# ----------------------------------------------------------------------------------
def make_extracted_DCFrame_dataset_1session():
    model_name = "gpt-4o-2024-08-06"
    todo_extractDC_jsonl_path = PROJECT_ROOT / ''
    output_after_extractDC_jsonl_dir = PROJECT_ROOT / ''
    sys_prompt_path = PROJECT_ROOT / 'src/extract_DCFrame/samples/sys_prompt_plink_uttrNoCategory_en.txt'
    examples_path = PROJECT_ROOT / 'src/extract_DCFrame/samples/AnnoMI/AnnoMI_s98_fs10link_NoCategory.txt'

    basename_without_ext = os.path.splitext(os.path.basename(todo_extractDC_jsonl_path))[0]
    session_num = basename_without_ext.split('_')[-1]
    tgt_ctx_num = basename_without_ext.split('_')[1]
    output_after_extractDC_jsonl_path = output_after_extractDC_jsonl_dir / f"policySupportDCFrame_{tgt_ctx_num}_session_{session_num}.jsonl"

    MakeDCFrameDataset = MakeDCFrameSupportDatasetController(model_name)
    sys_prompt = MakeDCFrameDataset.read_txt(sys_prompt_path)
    examples = MakeDCFrameDataset.read_txt(examples_path)
    MakeDCFrameDataset.extractDCFrame_1jsonl_addBase(
        todo_extractDC_jsonl_path,
        output_after_extractDC_jsonl_path,
        save_result_key='DCFrame_out',
        LLM_out_key='LLM_out',
        sys_prompt=sys_prompt,
        examples=examples,
        step1_temperature=0.0,
    )


def make_extracted_DCFrame_dataset_multisession():
    # Config ----------------------------------------------------------------------------------
    model_name = "gpt-4o-2024-08-06"
    todo_extractDC_jsonl_dir = PROJECT_ROOT / 'dataset/instance_jsonl/AnnoMI/policy/DCFrame_extracted_predata'
    todo_extractDC_jsonl_filename_template = 'policySuportPreDCFrame_TgtClCtx4_session_{session_num}.jsonl'
    todo_session_nums = [36]

    output_after_extractDC_jsonl_dir = PROJECT_ROOT / 'dataset/instance_jsonl/AnnoMI/policy/DCFrame_support_dataset/'
    output_jsonl_filename_template = 'policySupportDCFrame_TgtClCtx4_session_{session_num}.jsonl'

    sys_prompt_path = PROJECT_ROOT / 'src/extract_DCFrame/samples/sys_prompt_plink_uttrNoCategory_en.txt'
    examples_path = PROJECT_ROOT / 'src/extract_DCFrame/samples/AnnoMI/AnnoMI_s98_fs10link_NoCategory.txt'
    # -----------------------------------------------------------------------------

    MakeDCFrameDataset = MakeDCFrameSupportDatasetController(model_name)
    sys_prompt = MakeDCFrameDataset.read_txt(sys_prompt_path)
    examples = MakeDCFrameDataset.read_txt(examples_path)

    for todo_session_num in todo_session_nums:
        in_fname = todo_extractDC_jsonl_filename_template.format(session_num=todo_session_num)
        todo_extractDC_jsonl_path = todo_extractDC_jsonl_dir / in_fname

        out_fname = output_jsonl_filename_template.format(session_num=todo_session_num)
        output_after_extractDC_jsonl_path = output_after_extractDC_jsonl_dir / out_fname
        print(f'process {todo_session_num=}, {todo_extractDC_jsonl_path=}, {output_after_extractDC_jsonl_path=}')

        MakeDCFrameDataset.extractDCFrame_1jsonl_addBase(
            todo_extractDC_jsonl_path,
            output_after_extractDC_jsonl_path,
            save_result_key='DCFrame_out',
            LLM_out_key='LLM_out',
            sys_prompt=sys_prompt,
            examples=examples,
            step1_temperature=0.0,
        )


if __name__ == "__main__":
    logging.config.fileConfig(PROJECT_ROOT / "logging.conf")

    # make_extracted_DCFrame_dataset_1session()
    make_extracted_DCFrame_dataset_multisession()
