# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import datetime
import copy
import time


from ...find_near_state.find_near_DS_json_client import FindSimilarDSClient
from ...find_near_state.format_dialogue_state_utils import DialogueStateInformationFormatUtils

class FindNearDSController(object):
    def __init__(self, FindNearDSServerURL: str, CoIntentMapType: str) -> None:
        self.log = logging.getLogger(__name__)
        self.base_url = FindNearDSServerURL
        self.CoIntentMapType = CoIntentMapType
        self.FindNearDSClient = FindSimilarDSClient(self.base_url)
        self.log.debug(f"Set FindNearDSClient: URL: {self.base_url}")
        
        self.FormatDSUtils = DialogueStateInformationFormatUtils()

    def map_co_categoryID2Label(self, co_label:int):

        if self.CoIntentMapType == "Diet_MI":
            co_categoryID2Label = {
                1: "Question", 2: "Question",
                3: "Affirmation", 4: "Affirmation", 5: "Affirmation", 6: "Affirmation",
                7: "Reflection", 8: "Reflection", 9: "Reflection", 10: "Reflection", 11: "Reflection", 12: "Reflection", 13: "Reflection", 14: "Reflection",
                15: "Summarization", 16: "Summarization",
                17: "Other", 18: "Other", 19: "Other", 20: "Other", 21: "Other", 22: "Other", 23: "Other", 24: "Other", 25: "Other",
            }
        elif self.CoIntentMapType == "en_AnnoMI":
            # ref: make_dataset/AnnoMI/convert_AnnoMI.py
            co_categoryID2Label = {0: "Question", 1: "Affirmation", 2: "Reflection", 3: "Summarization", 4: "Other"}

        label = co_categoryID2Label.get(co_label, None)
        if label is None:
            raise ValueError(f"Invalid co_label: {co_label}")
        return label
    
    def get_similar_DS_samples_prompt_GTIntent(self, history_list:list, hicurrDCFrane:dict, top_n_DS:int):
        """ Create a few-shot prompt of History+DCFrame+Policy
            NOTE: intent use GT of counselor (-> use pseudo policy wo intent in NearDS_DB)
        Returns:
            concat_samples_str: str
                (sample1)### History\n<history>### Dialogue_content\n<>### Counselor_policy\n<>

                ---

                (sample2)
                ...
                
            sample_debug_info_list: list
                [{'uid': int, 'intent': str}, ...]

            Error: -1, -1
        """
        # query
        histroy_str = '\n'.join([f"{m_dict['speaker']}: {m_dict['utterance']}" for m_dict in history_list])
        query_srt = self.FormatDSUtils.template_hisory_DCFrame(histroy_str, str(hicurrDCFrane))

        # get similar DS
        nearDS_list_ret = self.FindNearDSClient.getSimilar(query_srt, top_n=top_n_DS)
        if nearDS_list_ret.get('error') is True:
            self.log.error(f"FindNearDSClient error: {nearDS_list_ret}")
            return -1, -1

        # make few-shot prompt str
        sample_str_list = []
        sample_debug_info_list = []
        for nearDS in nearDS_list_ret['similar_ret']:
            DS_instance = nearDS['DS_instance'] # {'DCFrame', out_policy({'response': {(intent), focuses, seek_attribute, }})}
            history_str = self.FormatDSUtils.format_history(DS_instance)
            DCFrame_str = str(DS_instance['DCFrame'])
            # policy + GT intent
            co_policy = DS_instance['out_policy']['response']
            GT_tgt_co_intent = DS_instance['target']['category_list'][-1]
            GT_tgt_co_uttr = DS_instance['target']['concat_utterance']
            co_policy['intent'] = self.map_co_categoryID2Label(GT_tgt_co_intent)   # update intent use GT of counselor's intent
            sample_str = self.FormatDSUtils.sample_format_history_DCFrame_policy(history_str, DCFrame_str, co_policy)
            sample_str_list.append(sample_str)

            # debug info
            _info_d = {'uid': nearDS['uid'], 'intent': co_policy['intent'], 'GT_response': GT_tgt_co_uttr}
            sample_debug_info_list.append(_info_d)

        concat_samples_str = '\n---\n'.join(sample_str_list)

        return concat_samples_str, sample_debug_info_list
            