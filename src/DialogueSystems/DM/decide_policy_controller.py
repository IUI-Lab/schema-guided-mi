# -*- coding: utf-8 -*-
# Decide and manage Policy TODO: set the policy generation method

import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import datetime
import copy
import time
import traceback


# common
from ..common.config import Config
from ..common.information_state_utils import InformationStateUtils
from ..common.ds_utils import DialogueSystemUtils
# Find Near DS Client
from .find_near_DS_controller import FindNearDSController

class DecidePolicyController():
    def __init__(self, use_near_DS_policy:bool):
        """
        Args:
            use_near_DS_policy (bool): True: use near DS policy, False: not use near DS policy
        """
        self.log = logging.getLogger(__name__)
        self.Config = Config()
        self.ISUtils = InformationStateUtils()
        self.DSUtils = DialogueSystemUtils()

        self.GenPolicyMethod = None # TODO: set policy generate method
        self.FindNearDSClientCtr = None
        if use_near_DS_policy:
            self.log.info(f"[Policy] {use_near_DS_policy=} server_URL: {self.Config.config.near_DS_server_url}, CoIntentMapType: {self.Config.config.near_DS_CoIntentMapType}")
            self.FindNearDSClientCtr = FindNearDSController(self.Config.config.near_DS_server_url, self.Config.config.near_DS_CoIntentMapType)

    def set_gen_policy_method(self, gen_policy_method):
        self.GenPolicyMethod = gen_policy_method

    # ------------------ Policy ------------------
    def _getUserPolicyInput(self, history, DCFrame):
        user_prompt_fmt = """### History
{history}

### Dialogue_content
{DCFrame}

### Counselor_policy
"""
        return user_prompt_fmt.format(history=history, DCFrame=DCFrame)

    def deal_unexpected_polisy(self, policy):
        """
        Args:
            policy (dict): {'intent': , 'focuses':, 'seek_attribute': }
        Return:
            policy (dict)
        """
        co_intent = policy['intent']
        if co_intent not in ['Question', 'Affirmation', 'Reflection', 'Summarization', 'Other']:
            self.log.warning(f"[Policy] intent is special case: {co_intent=}. then Intent: Other")
            new_policy = copy.deepcopy(policy)
            new_policy['intent'] = 'Other'
            self.log.info(f"[Policy] fix policy {new_policy=}")
            return new_policy
        else:
            return policy
        

    def get_policy(self, IS:dict, log_info={}):
        """ 
        Return:
            Dict: {'intent': , 'focuses':, 'seek_attribute': }
        """
        _IS = copy.deepcopy(IS)
        history_lst = self.DSUtils.getContext(_IS, self.Config.config.use_history_size)
        history = '\n'.join([f"{m_dict['speaker']}: {m_dict['utterance']}" for m_dict in history_lst])
        _current_DCFrame = _IS["current_DCFrame"]

        user_prompt = self._getUserPolicyInput(history, _current_DCFrame)
        self.log.debug(f"[Policy] [logInfo:{log_info}] user_prompt: {user_prompt}")
        policy_res_d = self.GenPolicyMethod.get_counselor_policy(user_prompt, temperature=self.Config.config.policy_temperature)  # {'response': , 'usage'}
        _IS = self.ISUtils.appendISList(_IS, "DM_act_info_history", policy_res_d, log_info)
        self.log.debug(f'[Policy] [logInfo:{log_info}] policy_res_d: {policy_res_d}')

        # Clean up policy
        policy = policy_res_d['response']

        if policy_res_d['response'] == -1:
            # NOTE: fail to get policy. then reply Other
            self.log.error(f"[Policy] [logInfo:{log_info}] policy_res_d['response'] == -1")
            policy = {'intent': 'Other'}   # NOTE: 
            _IS = self.ISUtils.appendISList(_IS, "DM_act_history", policy, log_info)
            return _IS, policy
        
        # Fix unexcepted case in policy-intent
        policy = self.deal_unexpected_polisy(policy)
        # add IS
        self.ISUtils.appendISList(_IS, "DM_act_history", policy, log_info)

        return _IS, policy
    
    def get_policy_with_near_DS(self, IS:dict, log_info={}):
        """ 
        Return:
            Dict: {'intent': , 'focuses':, 'seek_attribute': }
            Update IS:
                - DM_policy_nearDS
                - DM_act_history
                - DM_act_info_history : Added 2025-12-10. Stores info related to DM_act_history generation.
        """
        _IS = copy.deepcopy(IS)
        history_lst = self.DSUtils.getContext(_IS, self.Config.config.near_DS_context_num)  # TODO: check with server side ctx_num in embedding
        history = '\n'.join([f"{m_dict['speaker']}: {m_dict['utterance']}" for m_dict in history_lst])
        _current_DCFrame = _IS["current_DCFrame"]

        nearDS_samples_str, nearDS_infos = self.FindNearDSClientCtr.get_similar_DS_samples_prompt_GTIntent(history_lst, _current_DCFrame, self.Config.config.near_DS_num)
        # self.log.debug(f"[Policy] [logInfo:{log_info}] nearDS_samples_str: {nearDS_samples_str}")       # NOTE: debug check the few-shot string from near DS
        self.log.info(f"[Policy] [logInfo:{log_info}] nearDS_infos: {nearDS_infos}")

        main_query_user_prompt = self._getUserPolicyInput(history, _current_DCFrame)
        self.log.debug(f"[Policy] [logInfo:{log_info}] (near DS samples + ) main_query_user_prompt: {main_query_user_prompt}")
        concat_user_prompt = f"--- Examples ---\n\n{nearDS_samples_str}\n--- Examples End ---\n\n{main_query_user_prompt}"

        policy_res_d = self.GenPolicyMethod.get_counselor_policy(concat_user_prompt, temperature=self.Config.config.policy_temperature)  # {'response': , 'usage'}
        _IS = self.ISUtils.appendISList(_IS, "DM_act_info_history", policy_res_d, log_info)
        self.log.debug(f'[Policy] [logInfo:{log_info}] policy_res_d: {policy_res_d}')
        
        # Clean up policy
        policy = policy_res_d['response']        
    
        if policy_res_d['response'] == -1:
            # NOTE: fail to get policy. then reply Other
            self.log.error(f"[Policy] [logInfo:{log_info}] policy_res_d['response'] == -1")
            policy = {'intent': 'Other'}   # NOTE:
            _IS = self.ISUtils.appendISList(_IS, "DM_policy_nearDS", nearDS_infos, log_info)
            _IS = self.ISUtils.appendISList(_IS, "DM_act_history", policy, log_info)
            return _IS, policy
        
        # Fix unexcepted case in policy-intent
        policy = self.deal_unexpected_polisy(policy)
        # add IS
        _IS = self.ISUtils.appendISList(_IS, "DM_policy_nearDS", nearDS_infos, log_info)
        _IS = self.ISUtils.appendISList(_IS, "DM_act_history", policy, log_info)

        return _IS, policy