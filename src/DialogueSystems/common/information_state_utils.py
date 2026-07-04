# -*- coding: utf-8 -*-

# Functions for updating the IS (Information State) dict object

import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import copy
import json


class InformationStateUtils():
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def getValue(self, IS:dict, key:str, log_info={}):
        if key in IS:
            return IS[key]
        else:
            self.log.error(f"[IS] [logInfo:{log_info}] not exist key [{key}] in IS {IS}")
            raise NotImplementedError(f"not exist key [{key}] in IS")
        
    def setValue(self, IS:dict, key:str, value, log_info={}, verbose=False):
        _IS = IS
        _IS[key] = value
        if verbose is True:
            self.log.info(f"[IS] [logInfo:{log_info}] set key [{key}] value [{value}] result {_IS}")
        else:
            self.log.info(f"[IS] [logInfo:{log_info}] set key [{key}] value [{value}]")
        return copy.deepcopy(_IS)

    def appendISList(self, IS:dict, key:str, append_value, log_info={}, verbose=False):
        _IS = IS
        if key not in _IS:
            self.log.error(f"[IS] [logInfo:{log_info}] not exist key [{key}] in IS {_IS}")
            raise NotImplementedError(f"not exist key [{key}] in IS")
        if isinstance(_IS[key], list):
            _IS[key].append(append_value)
            if verbose is True:
                self.log.info(f"[IS] [logInfo:{log_info}] append Key [{key}] Value [{append_value}] result {_IS}")
            else:
                self.log.info(f"[IS] [logInfo:{log_info}] append Key [{key}] Value [{append_value}]")
        else:
            raise Exception(f'[IS] [logInfo:{log_info}] key: {key} is not list.')
        return copy.deepcopy(_IS)

    def saveISasJson(self, IS:dict, out_path:str, json_indent=False):
        """ Output the IS to a file (json)
            - UnicodeEncodeError occurred when console input contained partial characters, e.g. from incomplete deletion
            Args:
                json_indent: int or False
        """
        with open(out_path, 'w', encoding='utf-8') as f:
            if json_indent is False:
                json.dump(IS, f, ensure_ascii=False)
            else:
                json.dump(IS, f, ensure_ascii=False, indent=json_indent)

        self.log.info(f"[IS] output IS info to {out_path}")