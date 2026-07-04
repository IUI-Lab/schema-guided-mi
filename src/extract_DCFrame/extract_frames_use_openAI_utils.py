# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import logging
import json
import time
import copy

from openai import OpenAI

from .mi_frames_schema_link import MI_Structures, GoalAndIdealFrame


class ExtractFramesOpenAIUtils(object):
    def __init__(self, model_name: str) -> None:
        self.log = logging.getLogger(__name__)
        _goal = GoalAndIdealFrame()
        self.MI_Structures = MI_Structures(goal_frame=_goal)
        self.FrameSchema = self.MI_Structures.model_json_schema()
        self.model_name = model_name
        self.client = OpenAI()
        self.log.info(f"Use OpenAI model: {model_name}")
        self.log.debug(f"FrameSchema: {self.FrameSchema}")

    def get_1time_Chat(self, sys_cnt="", user_cnt="", tools=[], tool_choice="auto", temperature=0.0):
        """Send a single chat completion request to the OpenAI API.

        Args:
            sys_cnt: System prompt content.
            user_cnt: User message content.
            tools: List of tool definitions for function calling.
            tool_choice: Tool selection mode or specific tool spec.
            temperature: Sampling temperature.

        Returns:
            OpenAI ChatCompletion response object.
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

    def get_DCFrame(self, sys_prompt, user_ctx, temperature=0.0):
        """Extract a dialogue frame (DCFrame) using OpenAI function calling with the MI structure schema.

        Args:
            sys_prompt: System prompt defining the extraction task.
            user_ctx: User-side dialogue context to analyze.
            temperature: Sampling temperature.

        Returns:
            dict: On success —
                ``{'function_res': {'name': ..., 'arguments': <DCFrame dict>},
                   'usage': <token usage dict>, 'elapse_time': <float seconds>}``

            dict: On error —
                ``{'function_res': {'arguments': -1}, ...}`` if the API call fails,
                ``{'function_res': {'arguments': -2}, ...}`` if JSON parsing fails.
        """
        function_schema = {
            "type": "function",
            "function": {
                "name": self.FrameSchema["title"],        # -> 'MI_Structures'
                "description": self.FrameSchema["description"],
                "parameters": self.FrameSchema,
            },
        }

        s_time = time.time()
        try:
            res = self.get_1time_Chat(
                sys_prompt,
                user_ctx,
                tools=[function_schema],
                tool_choice={"type": "function", "function": {"name": "MI_Structures"}},
                temperature=temperature,
            )
        except Exception as e:
            elapsed_time = round((time.time() - s_time), 3)
            self.log.error(f"OpenAI API error: {e}")
            return {"function_res": {"arguments": -1}, "usage": -1, "elapse_time": elapsed_time}

        elapsed_time = round((time.time() - s_time), 3)
        self.log.info(f"[res] type: {type(res)} {res}")
        self.log.info(f"elapsed_time: {elapsed_time:.3f} [sec]")

        res_d = json.loads(res.model_dump_json())

        top_res = res_d["choices"][0]
        usage = res_d["usage"]
        function_res = top_res["message"]["tool_calls"][0]["function"]

        try:
            function_args_obj = json.loads(function_res["arguments"])
        except Exception as e:
            self.log.error(f"JSON parse error: {e}")
            function_args_obj = -2

        # Replace the raw JSON string with the parsed dict
        function_res["arguments"] = function_args_obj

        return {
            "function_res": copy.deepcopy(function_res),
            "usage": copy.deepcopy(usage),
            "elapse_time": elapsed_time,
        }
