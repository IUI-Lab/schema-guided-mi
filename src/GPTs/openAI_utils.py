# -*- coding: utf-8 -*-

# Utility functions for the OpenAI API.

import logging
import json
import time

from openai import OpenAI


class OpenAIUtils(object):
    def __init__(self, model_name: str) -> None:
        self.log = logging.getLogger(__name__)
        self.model_name = model_name
        self.client = OpenAI()
        self.log.info(f"Use OpenAI model: {model_name}")

    def get_1time_Chat(self, sys_cnt="", user_cnt="", temperature=0.0):
        """Send a single chat completion request (no tool use).

        Returns:
            ChatCompletion object. Access fields via attributes,
            e.g. ``res.choices[0].message.content``.
        """
        res = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": sys_cnt},
                {"role": "user", "content": user_cnt},
            ],
        )
        return res

    def get_top1_ChatCompletion_dict(self, sys_cnt, user_cnt, temperature=0.0):
        """Call the Chat API and return the top-1 generation result as a dict.

        Returns:
            dict: ``{'response': str, 'usage': dict, 'elapse_time': float}``

            On API error: ``{'response': -1, 'usage': -1, 'elapse_time': float}``
        """
        s_time = time.time()
        try:
            res = self.get_1time_Chat(sys_cnt, user_cnt, temperature)
        except Exception as e:
            elapsed_time = time.time() - s_time
            res_d = {"response": -1, "usage": -1, "elapse_time": elapsed_time}
            self.log.error(f"API error: {e}. Returning {res_d}")
            return res_d
        elapsed_time = round((time.time() - s_time), 3)

        self.log.debug(f"raw OpenAI response: {res}")
        top_res = res.choices[0]
        res_uttr = top_res.message.content
        usage = self.convert_usage_dict(res)

        return {"response": res_uttr, "usage": usage, "elapse_time": elapsed_time}

    def get_1time_Chat_with_tool(self, sys_cnt="", user_cnt="", tools=[], tool_choice="auto", temperature=0.0):
        """Send a single chat completion request with tool/function calling enabled.

        Args:
            sys_cnt: System prompt content.
            user_cnt: User message content.
            tools: List of tool definitions.
            tool_choice: Tool selection mode or specific tool spec.
            temperature: Sampling temperature.

        Returns:
            ChatCompletion response object.
        """
        res = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": sys_cnt},
                {"role": "user", "content": user_cnt},
            ],
            tools=tools,
            tool_choice=tool_choice,
        )
        return res

    def get_1time_ChatCompletion_StructuredOutput(self, sys_prompt, user_cnt, out_schema, temperature=0.0):
        """Send a chat completion request using OpenAI Structured Outputs.

        Note:
            Requires models that support Structured Outputs
            (e.g. ``gpt-4o-mini``, ``gpt-4o-2024-08-06`` or later).
        """
        res = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_cnt},
            ],
            temperature=temperature,
            response_format=out_schema,
        )
        return res

    def get_1time_ChatCompletion_StructuredOutput_dict(self, sys_prompt, user_cnt, out_schema, temperature=0.0):
        """Call the Structured Outputs API and return the result as a dict.

        Args:
            sys_prompt: System prompt.
            user_cnt: User message content.
            out_schema: Pydantic model defining the output schema.
            temperature: Sampling temperature.

        Returns:
            dict: ``{'response': <parsed dict>, 'usage': dict, 'elapse_time': float}``

            On API error: ``{'response': -1, 'usage': None, 'elapse_time': float}``

            On JSON parse error: ``{'response': -2, 'usage': dict, 'elapse_time': float}``
        """
        s_time = time.time()
        try:
            res = self.get_1time_ChatCompletion_StructuredOutput(sys_prompt, user_cnt, out_schema, temperature)
        except Exception as e:
            self.log.error(f"Structured output API error: {e}")
            elapsed_time = round((time.time() - s_time), 3)
            return {"response": -1, "usage": None, "elapse_time": elapsed_time}
        elapsed_time = round((time.time() - s_time), 3)

        struct_obj_str = res.choices[0].message.content
        try:
            struct_obj = json.loads(struct_obj_str)
        except Exception as e:
            usage = self.convert_usage_dict(res)
            self.log.error(f"JSON parse error: {e}")
            return {"response": -2, "usage": usage, "elapse_time": elapsed_time}

        self.log.debug(f"raw OpenAI response: {res}")
        usage = self.convert_usage_dict(res)

        return {"response": struct_obj, "usage": usage, "elapse_time": elapsed_time}

    def convert_usage_dict(self, openai_res) -> dict:
        """Convert an OpenAI response's usage object to a plain dict.

        Args:
            openai_res: Raw ChatCompletion response object.

        Returns:
            dict with token counts and detail breakdowns.
        """
        completion_tokens_details = {
            "accepted_prediction_tokens": openai_res.usage.completion_tokens_details.accepted_prediction_tokens,
            "audio_tokens": openai_res.usage.completion_tokens_details.audio_tokens,
            "reasoning_tokens": openai_res.usage.completion_tokens_details.reasoning_tokens,
            "rejected_prediction_tokens": openai_res.usage.completion_tokens_details.rejected_prediction_tokens,
        }
        prompt_tokens_details = {
            "audio_tokens": openai_res.usage.prompt_tokens_details.audio_tokens,
            "cached_tokens": openai_res.usage.prompt_tokens_details.cached_tokens,
        }
        return {
            "completion_tokens": openai_res.usage.completion_tokens,
            "prompt_tokens": openai_res.usage.prompt_tokens,
            "total_tokens": openai_res.usage.total_tokens,
            "completion_tokens_details": completion_tokens_details,
            "prompt_tokens_details": prompt_tokens_details,
        }
