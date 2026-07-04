# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import copy

from ...GPTs.openAI_utils import OpenAIUtils


from ..common.config import Config
from ..common.ds_utils import DialogueSystemUtils
from ..common.information_state_utils import InformationStateUtils

class RGMethodOpenAIBasedIntentFocusSeekAttribute():
    def __init__(self, model_name:str, rg_error_response:str) -> None:
        self.log = logging.getLogger(__name__)
        self.Config = Config()
        self.DSUtils = DialogueSystemUtils()
        self.ISUtils = InformationStateUtils()
        self.GPTUtils = OpenAIUtils(model_name)
        self.rg_error_response = rg_error_response
        
        self.sys_prompt = ""

    def set_sys_prompt(self, sys_prompt:str):
        self.sys_prompt = sys_prompt
        self.log.debug(f'[RG] set sys_prompt: {self.sys_prompt}')

    def _make_user_prompt(self, history:str, DCFrame:str, policy:dict):
        """ Deal DC. intent, focuses, seek_frame_type, seek_attribute"""
        self.log.debug(f"[RG Method] {policy=}")
        intent = policy['intent']
        focuses = policy.get('focuses', [])
        seek_frame_type = policy.get('seek_frame_type', "")
        seek_attr = policy.get('seek_attribute', "")

        user_prompt = """### History
{history}

### Dialogue_content
{DCFrame}

### Intent
{Intent}

### Focuses
{focuses}

### Seek_frame_type
{seek_frame_type}

### Seek_attribute
{seek_attribute}

### Response
""".format(history=history, DCFrame=str(DCFrame), Intent=intent, focuses=str(focuses), seek_frame_type=seek_frame_type, seek_attribute=str(seek_attr))
        
        return user_prompt


    #  Use DA only
    # ----------------------------------------------------------------------------------------------
    def get_response(self, IS:dict, policy:dict, part_DCFrame=False, log_info={}):
        """Use Intent, Focuses, Seek_attribute in user prompt
        Returns:
            _IS: dict: updated IS( appended RG_history)
            str: sys response
        """
        _IS = copy.deepcopy(IS)
        # get user prompt
        history_lst = self.DSUtils.getContext(_IS, self.Config.config.use_history_size)
        history = '\n'.join([f"{m_dict['speaker']}: {m_dict['utterance']}" for m_dict in history_lst])
        current_DCFrame = _IS["current_DCFrame"]

        # make user prompt
        user_prompt = self._make_user_prompt(history, current_DCFrame, policy)
        self.log.debug(f"[RG] [logInfo:{log_info}] query to RG {user_prompt=}")
        res_d = self.GPTUtils.get_top1_ChatCompletion_dict(self.sys_prompt, user_prompt, self.Config.config.rg_temperature)  # -> dict: {'response': str, 'usage': dict, 'elapse_time': float}
        self.log.debug(f"[RG] [logInfo:{log_info}] RGCtr-res_d: {res_d}")
        
        # Log to IS 
        _IS = self.ISUtils.appendISList(_IS, 'RG_history', res_d, log_info=log_info, verbose=self.Config.config.IS_console_print_verbose)
        

        # fail to gen response. then return error_response
        if res_d['response'] == -1:
            self.log.error(f"[RG] [logInfo:{log_info}] res_d['response'] == -1")
            return _IS, self.rg_error_response
        else:
            return _IS, res_d['response']
