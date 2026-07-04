# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
import pprint
import logging, logging.config


class DialogueSystemUtils(object):
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            txt = f.read()
            return txt

    def getContext(self, IS:dict, history_size:int):
        """
        Get the past history_uttrs of the given history_size. order: old->new
        NOTE: Each utterance of Co, Cl is counted as one
        """
        _ctx = IS.get("history_uttrs")[-history_size:]    # if fewer than size, get all available
        # print(f"_ctx len: {len(_ctx)}, _ctx: {_ctx}")
        return _ctx
