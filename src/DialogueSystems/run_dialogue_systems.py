# -*- coding: utf-8 -*-

### command
# uv run python -m src.DialogueSystems.run_dialogue_systems --config src/DialogueSystems/sys_config_en_AnnoMI.yaml
# ...
# Current History Uttr size: 0
# User>       // <-- input user utterance (e.g., "Hello, I'm worried about overeating.")
#
# NOTE: entering "exit" (or "Exit") during the dialogue ends the session and saves the log to Dialogue_log_<timestamp>.json

import os, sys
import argparse
from pathlib import Path
import pprint
import logging, logging.config
import datetime
import copy
import time
import traceback

# common
from .common.config import Config
from .common.ds_utils import DialogueSystemUtils
from .common.information_state_utils import InformationStateUtils
# DM
from .DM.LU_controller import LUController
from .DM.DCFrame_update_controller import DCFrameUpdateController
# DM - policy
from .DM.decide_policy_controller import DecidePolicyController # policy main controller
from ..gen_SemanticStrategy.gen_policy import GeneratePolicyController       # policy method
# RG
from .RG.RG_controller import ResponseGenerationController


PROJECT_ROOT = Path(__file__).resolve().parents[2]

class DSController():
    def __init__(self):
        self.log = logging.getLogger(__name__)
        # Common
        self.Config = Config() # TODO then don't forget set config file
        self.DSUtils = DialogueSystemUtils()
        self.ISUtils = InformationStateUtils()

        # LU
        self.LUCtr = LUController()
        # DCFrame
        self.ExtractFrameCtr = None
        # Policy
        self.PolicyCtr = None
        self.FindNearDSClientCtr = None
        # RG
        self.RGCtr =None


    # ------------------ Main ------------------
    def _initialize_IS(self, uinfo:dict):
        _IS = {}
        _IS = self.ISUtils.setValue(_IS, 'current_DCFrame', {}, log_info=uinfo)
        _IS = self.ISUtils.setValue(_IS, 'DCFrame_LLMOut_history', [], log_info=uinfo)
        _IS = self.ISUtils.setValue(_IS, 'DCFrame_unexpectedErrors', [], log_info=uinfo)    # [{}, {'phase', 'error'}, ...]
        _IS = self.ISUtils.setValue(_IS, 'DCFrame_history', [], log_info=uinfo)
        _IS = self.ISUtils.setValue(_IS, 'history_uttrs', [], log_info=uinfo)   # changed unnecessary char, final response
        _IS = self.ISUtils.setValue(_IS, 'DM_policy_nearDS', [], log_info=uinfo)   # [[topNnearDS: {'uid', '(GT)intent', 'GT_response'}], []]
        _IS = self.ISUtils.setValue(_IS, 'DM_act_history', [], log_info=uinfo)  # [{'intent': , 'focuses':, 'seek_attribute': }, ...]
        _IS = self.ISUtils.setValue(_IS, 'DM_act_info_history', [], log_info=uinfo) # `DM_act_history` with info (completion usage etc)
        _IS = self.ISUtils.setValue(_IS, 'RG_history', [], log_info=uinfo)  # raw generated response
        _IS = self.ISUtils.setValue(_IS, 'total_generation_time', [], log_info=uinfo)  # generation time log
        _IS = self.ISUtils.setValue(_IS, 'sys_config', self.Config.get_config_dict(), log_info=uinfo)
                                               
        return _IS
    
    def gen_process(self, user_info:dict, IS:dict, input_uttr:str):
        """
        Returns:
            IS: dict, updated IS
            out_uttr: str
        """
        s_time = time.time()
        _IS = copy.deepcopy(IS) # _IS: intermidiate IS result
        if _IS == {}:
            # Initialize IS (first turn)
            self.log.info(f"[DS] [logInfo:{user_info}] Initialize IS")
            _IS = self._initialize_IS(user_info)

        # LU
        _IS = self.LUCtr.appendUserUttr2IShistory(_IS, input_uttr)

        # Extract DCFrame & Update DCFrame
        _IS = self.ExtractFrameCtr.getUpdatedDCFrame(_IS, log_info=user_info)
        self.log.debug(f'[DS] after ExtracDCFrame _IS: {_IS}')
        
        # get DS-policy
        if self.Config.config.use_near_DS is False:
            _IS, policy = self.PolicyCtr.get_policy(_IS, log_info=user_info)
        else:
            _IS, policy = self.PolicyCtr.get_policy_with_near_DS(_IS, log_info=user_info)
        self.log.info(f"[DS] [logInfo:{user_info}] final {policy=}")
        
        # RG (use policy)
        _IS, sys_uttr = self.RGCtr.get_response(_IS, policy, log_info=user_info)
        sys_uttr = self.RGCtr.replace_unnnecessary_char(sys_uttr)   # replace unnecessary characters
        _IS = self.RGCtr.appendSysUttr2IShistory(_IS, sys_uttr, log_info=user_info, verbose=self.Config.config.IS_console_print_verbose)    # add sys_uttr to IS-history: {'speaker': 'Counselor', 'utterance': sys_uttr}

        self.log.debug(f'[DS] [logInfo:{user_info}] final IS[{_IS}]')        # debug TODO: remove
        gen_elapsed_time = time.time() - s_time
        _IS = self.ISUtils.appendISList(_IS, 'total_generation_time', gen_elapsed_time, log_info=user_info)

        return _IS, sys_uttr

    def gen_Echo_process(self, user_info:dict, IS:dict, input_uttr:str):
        """ for debbug """
        _IS = copy.deepcopy(IS) # _IS: intermidiate IS result
        if _IS == {}:
            # Initialize IS (first turn)
            self.log.info(f"[DS] [logInfo:{user_info}] Initialize IS")
            _IS = self._initialize_IS(user_info)
        # LU
        _IS = self.LUCtr.appendUserUttr2IShistory(_IS, input_uttr)
        # Echo Ver for test
        sys_uttr = "Echo-" + input_uttr   # Echo ver
        _IS = self.RGCtr.appendSysUttr2IShistory(_IS, sys_uttr, log_info=user_info, verbose=self.Config.config.IS_console_print_verbose)    # add sys_uttr to IS-history: {'speaker': 'Counselor', 'utterance': sys_uttr}
        self.log.debug(f'[DS] [logInfo:{user_info}] final IS[{_IS}]')        # debug TODO: remove
        return _IS, sys_uttr

    def IOloop(self, uinfo:dict, init_IS:dict):
        _IS = init_IS   # _IS: intermidiate IS result
        while True:
            # LU
            print(f"Current History Uttr size: {len(_IS['history_uttrs'])}")
            in_uttr = input("User> ") # NOTE: do not press DELETE in the terminal; it gets entered as an unparsable character and breaks JSON output too

            if in_uttr in ["exit", "Exit"]:
                print('Session Ended')
                return _IS
            s_time = time.time()
            _updated_IS, out_uttr = self.gen_process(uinfo, _IS, in_uttr)
            turn_time = time.time() - s_time
    
            print('------------------------------------------------------')
            print(f'Response Time: {turn_time:.4f} [sec]\n')
            print(f"Sys> {out_uttr}")
            print('------------------------------------------------------')


            _IS = _updated_IS

    def init_cond_proposed(self, user_info=None):
        """ dynamic-policy
        """
        # Setup DS (Condition) -----------------------------------------------------------------------------------------------------
        
        # ExtractDCFrame
        self.ExtractFrameCtr = DCFrameUpdateController()
        # sys_prompt_path = PROJ_PATH + '/LLMs/extract_DCFrame/samples/sys_prompt_plink_uttrNoCategory_20241009.txt'
        _extractDC_sys_prompt_path = PROJECT_ROOT / self.Config.config.extractDC_sys_prompt_path if self.Config.config.extractDC_sys_prompt_path is not None else None
        _extractDC_examples_path = PROJECT_ROOT / self.Config.config.extractDC_examples_path if self.Config.config.extractDC_examples_path is not None else None
        self.ExtractFrameCtr.set_sys_prompt_by_path(_extractDC_sys_prompt_path, _extractDC_examples_path)


        ### Policy TODO: select the policy generation method and set it on PolicyController
        USE_NEAR_DS_policy = self.Config.config.use_near_DS
        self.PolicyCtr = DecidePolicyController(USE_NEAR_DS_policy)
        _GenPolicyMethod = GeneratePolicyController(self.Config.config.model_name)  # policy generation method

        # -------------- dynamic fewshot policy prompt -------------- 
        # Proposed FindNearDS->policy
        _policy_sys_prompt_path = PROJECT_ROOT / self.Config.config.policy_sys_prompt_path if self.Config.config.policy_sys_prompt_path is not None else None
        # _policy_sys_fewshot_path = PROJECT_ROOT / self.Config.config.policy_prompt_fewshot_path if self.Config.config.policy_prompt_fewshot_path is not None else None
        _GenPolicyMethod.set_sys_prompt(f"{self.DSUtils.read_txt(_policy_sys_prompt_path)}")
        self.PolicyCtr.set_gen_policy_method(_GenPolicyMethod)  # set policy generate method to PolicyController
        # ---------------------------------------------------------------------------------------------------------------
        
        ### RG
        self.RGCtr = ResponseGenerationController(self.Config.config.model_name, self.Config.config.rg_error_response)    # TODO select RG method in ResponseGenerationController
        # [Method] DA, focueses, seek_attribute (with DCFrame full)
        _fmt_rg_sys_prompt_path = PROJECT_ROOT / self.Config.config.fmt_rg_sys_prompt_path if self.Config.config.fmt_rg_sys_prompt_path is not None else None
        _rg_sys_prompt = self.DSUtils.read_txt(_fmt_rg_sys_prompt_path)
        
        self.RGCtr.set_RG_sys_prompt(_rg_sys_prompt)


    def run_DS(self):
        _user_info = {'username': 'test_user', 'roomId': 'test_room'}
        # Init IS        
        init_IS = self._initialize_IS(_user_info)
        _IS = self.IOloop(uinfo=_user_info, init_IS=init_IS)   # for debug

        now = datetime.datetime.now()
        now_str = now.strftime('%Y%m%d%H%M%S')
        out_log_json = f"./Dialogue_log_{now_str}.json"
        self.ISUtils.saveISasJson(_IS, out_log_json, self.Config.config.save_json_indent)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the dialogue system")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "src" / "DialogueSystems" / "sys_config_en_AnnoMI.yaml",
        help="Path to the sys_config yaml file (default: %(default)s)",
    )
    return parser.parse_args()

def test_ds(config_path: Path):
    DSCtr = DSController()
    DSCtr.Config.set_config(config_path)  # access config params by DSCtr.Config.config.<param_key>
    DSCtr.init_cond_proposed()  # initialize DS with config and prompt
    pprint.pprint(DSCtr.Config.config)
    DSCtr.run_DS()

if __name__ == "__main__":
    config_f = PROJECT_ROOT / "logging.conf"
    logging.config.fileConfig(config_f)
    args = parse_args()
    test_ds(args.config)