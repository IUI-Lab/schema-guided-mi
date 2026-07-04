# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pprint
import logging, logging.config
import json
import time
import copy

import openai
from openai import OpenAI  # pip install --upgrade openai

from .counselor_policy_schema import CounselorPolicy
from ..GPTs.openAI_utils import OpenAIUtils

class GeneratePolicyController(object):
    def __init__(self, model_name:str) -> None:
        self.log = logging.getLogger(__name__)
        self.model_name = model_name
        self.client = OpenAI()
        self.openai_utils = OpenAIUtils(model_name)
        self.log.info(f"Use OpenAI-model_name: {model_name}")

        self.CounselorPolicySchema = CounselorPolicy    # generate: intent, focuses, seek_frame_type, seek_attribute

        self.sys_prompt = None

    def set_sys_prompt(self, sys_prompt:str):
        self.sys_prompt = sys_prompt
        self.log.info(f'set generate policy sys_prompt: {self.sys_prompt}')

    def get_1time_ChatCompletion_StructuredOutput(self, user_cnt, temperature=0.0):
        """
        NOTE: use opneAI's Structured Outputs  model: gpt-4o-mini, gpt-4o-2024-08-06 and later
        """
        res = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_cnt}
            ],
            temperature=temperature,
            response_format=self.CounselorPolicySchema,  # specify structured output
        )

        return res

    def get_counselor_policy(self, user_cnt, temperature=0.0):
        s_time = time.time()
        try:
            res = self.get_1time_ChatCompletion_StructuredOutput(user_cnt, temperature)
        except Exception as e:
            # Handle edge cases
            self.log.error(f"Error in query policy: {e}")
            elapsed_time = round((time.time() - s_time), 3)
            return {'response': -1, 'usage': None, 'elapse_time': elapsed_time}
        elapsed_time = round((time.time() - s_time), 3)

        struct_obj_str = res.choices[0].message.content # str
        try:
            struct_obj = json.loads(struct_obj_str)
        except Exception as e:
            usage = self.openai_utils.convert_usage_dict(res)
            self.log.error(f"[Policy] Error in json.loads: {e}")
            return {'response': -2, 'usage': usage, 'elapse_time': elapsed_time}
        self.log.debug(f"raw openAI response: {res}")
        usage = self.openai_utils.convert_usage_dict(res)

        return {'response': struct_obj, 'usage': usage, 'elapse_time': elapsed_time}
    