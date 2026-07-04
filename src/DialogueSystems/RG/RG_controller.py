# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import copy

### DS_TOP
DS_PATH = str(Path(__file__).resolve().parents[3])
sys.path.append(DS_PATH)

from ..common.config import Config
from ..common.ds_utils import DialogueSystemUtils
from ..common.information_state_utils import InformationStateUtils

# RG Methods
from .rg_method_intent_focus_seekAttr_openAI import RGMethodOpenAIBasedIntentFocusSeekAttribute    # OpenAI Intent, Focus, SeekAttr


class ResponseGenerationController(object):
    def __init__(self, model_name:str, rg_error_response:str) -> None:
        self.log = logging.getLogger(__name__)
        self.Config = Config()
        self.ISUtils = InformationStateUtils()
        self.DSUtils = DialogueSystemUtils()

        # RG Method
        self.RGMethod = RGMethodOpenAIBasedIntentFocusSeekAttribute(model_name, rg_error_response)  # Policy:[DA, Focus, SeekAttr],  opneAI


    def appendSysUttr2IShistory(self, IS, out_uttr, log_info={}, verbose=False):
        _IS = copy.deepcopy(IS)
        sys_d = {"speaker": "Counselor", "utterance": out_uttr}
        _IS = self.ISUtils.appendISList(_IS, "history_uttrs", sys_d, log_info=log_info, verbose=verbose)
        return _IS

    def set_RG_sys_prompt(self, sys_prompt:str):
        """Set RG sys prompt"""
        self.RGMethod.set_sys_prompt(sys_prompt)
    
    def get_response(self, IS:dict, policy:dict, log_info={}):
        """Get response
        Returns:
            _IS `dict`: Updated IS (append RG result on RG_history)
        """
        _IS = copy.deepcopy(IS)
        _IS, response_str = self.RGMethod.get_response(IS, policy, part_DCFrame=False, log_info=log_info)
        return _IS, response_str
    
    def replace_unnnecessary_char(self, text:str):
        text = text.strip()
        text = text.lstrip('「')
        text = text.rstrip('」')

        return text
