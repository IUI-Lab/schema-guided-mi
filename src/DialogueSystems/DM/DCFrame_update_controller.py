# -*- coding: utf-8 -*-

# IS
#   - history_uttrs (list)

import os, sys
from pathlib import Path
import copy
import pprint
import logging
import traceback

from ..common.config import Config
from ..common.information_state_utils import InformationStateUtils
from ...AddingDCFrame.DCFrame_add_update_controller import AddingDCFrameUpdateController  # Add base DCFrame controller
from ...extract_DCFrame.extract_frames_use_openAI_utils import ExtractFramesOpenAIUtils


class DCFrameUpdateController(object):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.Config = Config()
        self.ISUtils = InformationStateUtils()
        model_name = self.Config.config.model_name
        if model_name == "":
            self.ExtractFramesOpenAIUtils = None
        else:
            self.ExtractFramesOpenAIUtils = ExtractFramesOpenAIUtils(model_name)
        self.sys_prompt = ""

        self.AddingDCFrameCtr = AddingDCFrameUpdateController()

    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            txt = f.read()
            return txt
        
    def set_sys_prompt_by_path(self, sys_prompt_path, example_path=None):
        """NOTE: deal example version"""
        sys_prompt = self.read_txt(sys_prompt_path)
        example = self.read_txt(example_path) if example_path is not None else ""
        self.sys_prompt = sys_prompt + '\n'  + example
        
        
    def show_sys_prompt(self):
        self.log.info(f'[DCFrame update] set ---------------- sys_prompt\n{self.sys_prompt}')

    def getContext(self, IS:dict, history_size):
        """
        Get the past history_uttrs utterances up to history_size. order: old->new
        NOTE: Each utterance of Co, Cl is counted as one
        """
        history_uttrs = IS.get('history_uttrs')
        _ctx = history_uttrs[-history_size:]    # if fewer than size, get as many as available
        # print(f"_ctx len: {len(_ctx)}, _ctx: {_ctx}")
        return _ctx
    

    def getUpdatedDCFrame(self, IS:dict, log_info={}):
        """update DCFrame based on history (include new user input)
        flow:
            1. get previous_DCFrame (current_DCFrame)
            2. Extract DCFrame
            3. Adding base update DCFrame
            4. set DCFrame (IS)

        history_format ():
            Client: Hello
            Counselor: Hi
            Client: I want to talk about my stress
            Counselor: Sure, I can help you with that
        updated IS params:
            - current_DCFrame: dict     # current DCFrame of the dialogue
            - DCFrame_LLMOut_history: list  # append DCFrame_LLMOut
            - DCFrame_unexpectedErrors: list     # append DCFrame parse errors
            - DCFrame_history: list     # keep appending current_DCFrame

        """
        ### 1. prepare Context using history_uttrs
        _IS = copy.deepcopy(IS)
        n_ctx_uttr_list = self.getContext(_IS, self.Config.config.use_history_size)
        formatted_ctx = [f"{uttr_dict['speaker']}: {uttr_dict['utterance']}" for uttr_dict in n_ctx_uttr_list]
        str_ctx = '\n'.join(formatted_ctx)  # History
        
        previous_DCFrame = _IS.get("current_DCFrame", None) # the last DCFrame from the previous step, to be updated
        if previous_DCFrame is None:
            previous_DCFrame = {}
        # self.log.debug(f'previous_DCFrame (current_DCFrame): {previous_DCFrame}')
        

        ### 2. get DCFrame from LLM & update src_tgt_dict
        self.log.debug(f'[DCFrame] [logInfo:{log_info}] Query previous_DCFrame(current_DCFrame)\n{previous_DCFrame}\n--- str_ctx\n{str_ctx}')
        user_ctx = f"### Current_structure\n{str(previous_DCFrame)}\n\n### Utterance\n{str_ctx}\n\n### Updated_structure\n"   # string combining sys_prompt & utterance context
        # print(f'debug user_ctx:\n{user_ctx}') # debug

        try:
            out_LLM = self.ExtractFramesOpenAIUtils.get_DCFrame(self.sys_prompt, user_ctx)   # {'function_res', 'usage', 'elapsed_time'}
            self.log.debug(f'[DCFrame] [logInfo:{log_info}] out_LLM: {out_LLM}')
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_LLMOut_history', out_LLM, log_info)     # NOTE: IS-DCFrame_LLMOut_history (result of LLM extraction success/failure)
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_unexpectedErrors', {}, log_info)       # NO unexpected error -> {}

        # API call Error NOTE: unintended error
        except Exception as e:
            self.log.error(f'[DCFrame] [logInfo:{log_info}] Error: Step1 in Extract DCFrame. Do not update `current_DCFrame` \nError: {e}')
            _result_fail_DCFrame = {'error': 'query_get_DCFrame_fail'}
            # NOTE: update [DCFrame_LLMOut_history, DCFrame_history, DCFrame_unexpectedErrors] NO-UPDATE: [current_DCFrame]
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_LLMOut_history', _result_fail_DCFrame, log_info)
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_unexpectedErrors', {'phase':'get_DCFrame', 'error': str(e)}, log_info)
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_history', copy.deepcopy(previous_DCFrame), log_info)

            return _IS
        
        # After extract DCFrame
        out_LLM_function_res = out_LLM['function_res']['arguments']  # DCFrame or Error code
        if out_LLM_function_res == -1 or out_LLM_function_res == -2:
            # NOTE: update [DCFrame_LLMOut_history, DCFrame_history, ] NO-UPDATE: [DCFrame_unexpectedErrors, current_DCFrame]
            self.log.warning(f'[DCFrame] [logInfo:{log_info}] Failed to extract DCFrame. Do not update `current_DCFrame` out_LLM_function_res:[{out_LLM_function_res}]')
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_history',  copy.deepcopy(previous_DCFrame), log_info)
            
            return _IS
        

        ### 3. Adding base update DCFrame
        try:
            _addBase_out_DCFrame = self.AddingDCFrameCtr.adding_base_update_DCFrame(previous_DCFrame, out_LLM_function_res)    # addBase
            self.log.debug(f'[DCFrame] [logInfo:{log_info}] _addBase_out_DCFrame: {_addBase_out_DCFrame}')
        except Exception as e:
            self.log.warning(f'[DCFrame] [logInfo:{log_info}] Failed to addBase DCFrame. Keep using old DCFrame \nError: {e}')
            self.log.warning(f'[DCFrame] [logInfo:{log_info}] Error: {traceback.format_exc()}\n')
            # NOTE: update [DCFrame_LLMOut_history, DCFrame_history, DCFrame_unexpectedErrors] NO-UPDATE: [ current_DCFrame]
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_unexpectedErrors', {'phase':'addBase_DCFrame', 'error': str(e)}, log_info)
            _IS = self.ISUtils.appendISList(_IS, 'DCFrame_history',  copy.deepcopy(previous_DCFrame), log_info)
            return _IS

        ### 4. update DCFrame (IS)
        # NOTE: update [DCFrame_LLMOut_history, DCFrame_history, current_DCFrame]
        _IS = self.ISUtils.setValue(_IS, 'current_DCFrame', _addBase_out_DCFrame)
        _IS = self.ISUtils.appendISList(_IS, 'DCFrame_history', _addBase_out_DCFrame, log_info)
        
        return _IS


        


