# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
import pprint
import logging, logging.config

from ..common.information_state_utils import InformationStateUtils

class LUController(object):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.ISUtils = InformationStateUtils()

    def appendUserUttr2IShistory(self, IS:dict, in_uttr:str, log_info={}):
        user_d = {"speaker": "Client", "utterance": in_uttr}
        _IS = self.ISUtils.appendISList(IS, "history_uttrs", user_d, log_info=log_info)
        return _IS

    