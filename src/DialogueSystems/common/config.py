# -*- coding: utf-8 -*-

import yaml
from types import SimpleNamespace
import logging, logging.config
import copy

class Config(object):
    # global
    config = {}
    config_dict = None
    yaml_path = None
    def __init__(self):
        self.log = logging.getLogger(__name__)
    
    def set_config(self, yaml_path):
        yaml_path = yaml_path
        with open(yaml_path, 'r') as yml:
            _d = yaml.safe_load(yml)
            Config.config_dict = _d
            Config.config = SimpleNamespace(**_d)   # Access using ClassName.variableName

        self.log.info(f'loaded yaml config: {Config.config}')


    def get_config_dict(self):
        return copy.deepcopy(Config.config_dict)