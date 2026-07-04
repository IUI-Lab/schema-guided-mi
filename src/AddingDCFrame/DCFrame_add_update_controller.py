# -*- coding: utf-8 -*-

"""
DCFrame Addition-Based Update Controller

This module provides AddingDCFrameUpdateController, which merges a new candidate DCFrame
(e.g. produced by an LLM) into an existing base DCFrame using an addition-only strategy:
values present in the candidate but absent from the base are added; nothing is deleted.

Core workflow (adding_base_update_DCFrame):
  0. Build an instance key DB from the base DCFrame, keyed by each instance's content[0] value.
  1. Add new instance content (content index 0) detected in the diff.
  2. Add new goal_and_ideal content values.
  3. Add remaining attribute values (e.g. detail, harm_effect) to the matched instances.

Dependencies:
  - DCFrameDiffUtils (./DCFrame_diff_utils.py): flattening, diffing, and filtering DCFrames.
"""

import sys
from pathlib import Path
import pprint
import copy
import logging, logging.config

from .DCFrame_diff_utils import DCFrameDiffUtils

class AddingDCFrameUpdateController(object):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.DCFrameDiffUtils = DCFrameDiffUtils()
        self.concat_sp_char = '__'

    def _build_instance_key_using_contentIdx0Value(self, DCFrame:dict):
        """
        control each instance based on (<instance_type>, <instance_index(int: from1)>, <content-value_idx0>)
        Returns:
            instance_key_dict (dict):
                { `tuple`(<instance_type>, <instance_index(from1)>, <content-value_idx0>): 0, ... }
        """
        instance_key_DB = {}
        # check only has instance frame_type
        for frame_type in ['problem_and_trouble', 'experience', 'improvement_plan']:
            frame_access_key_in_DCFrame = frame_type + '_frames'
            for frame_instance in DCFrame.get(frame_access_key_in_DCFrame,[]):
                if len(frame_instance) == 0:    # no instance in that frame_type
                    continue

                instance_contents = frame_instance.get('content')   # impossible to have no content
                _key_tuple = (frame_type, frame_instance.get('type_instance_index'), instance_contents[0])
                instance_key_DB[_key_tuple] = 0 # put 0 as val

        return instance_key_DB

    # _add_instance_content()
    # --------------------------------------------------------------------------------------------------------------------------NOTE: current key_DB
    def has_content_key_in_keyDB(self, candidate_ftype, candidate_content_val, instance_key_DB:dict) -> bool:
        """Check if the given ftype and content already exist in instance_key_DB.
        NOTE: The current key_DB only targets content index 0.
        """
        for key_tuple in instance_key_DB.keys():
            # key_tuple: (<instance_type>, <instance_index(from1)>, <content-value_idx0>)
            if key_tuple[0] == candidate_ftype and key_tuple[2] == candidate_content_val:
                return True
        return False

    def _count_frameType_instance_key_DB(self, instance_key_DB:dict, frame_type: str):
        frame_type_instance_num = 0
        for key_tuple in instance_key_DB.keys():
            if key_tuple[0] == frame_type:
                frame_type_instance_num += 1
        return frame_type_instance_num


    def _add_instance_content(self, instance_key_DB:dict, baseDCFrame:dict, add_lst:list):
        """Using the InstanceKeyDB built from the base DCFrame, add entries from add_lst whose
        content val_idx0 is new, and return the updated DCFrame.
        Args:
            instance_key_dict (dict): { `<instance_type>__<instance_index(from1)>__content__<value_idx(from0)>__<value0>`: 0, ... }
            baseDCFrame (dict): previous DCFrame
            add_lst (list): list of added instance
        """
        updated_DCFrame = copy.deepcopy(baseDCFrame)
        rest_add_lst = []

        for concat_val in add_lst:
            # skip case
            if 'goal_and_ideal' in concat_val:
                rest_add_lst.append(concat_val)
                continue
            try:
                ftype, instance_idx, attribute, val_index, val = concat_val.split('__')
            except Exception as e:
                print(f'Unpack Error: {e}\n dealing... {concat_val=}')
                exit()

            if attribute != 'content' or val_index != '0': # only deal with content index 0
                rest_add_lst.append(concat_val)
                continue

            # might be new instance-content, then check new instance content
            if self.has_content_key_in_keyDB(ftype, val, instance_key_DB) is True:
                self.log.debug(f'[add] instance content already exists in instance_key_DB. add_lst - concat_val: {concat_val}')
                continue
            else:
                # regist new instance content
                _current_instance_num = self._count_frameType_instance_key_DB(instance_key_DB, ftype)   # from 1
                _regist_instance_index = _current_instance_num + 1  # from 1
                regist_key_tuple = (ftype, _regist_instance_index, val)
                instance_key_DB[regist_key_tuple] = 0
                

                # add new instance content to DCFrame
                new_instance = {'frame_type': ftype, 'type_instance_index': _regist_instance_index, 'content': [val]}   # frame_type: 'problem_and_trouble', 'experience', ...
                frame_access_key_in_DCFrame = ftype + '_frames' # 'problem_and_trouble_frames', 'experience_frames', ...
                if frame_access_key_in_DCFrame not in updated_DCFrame.keys():
                    updated_DCFrame[frame_access_key_in_DCFrame] = []
                    updated_DCFrame[frame_access_key_in_DCFrame].append(new_instance)
                else:
                    updated_DCFrame[frame_access_key_in_DCFrame].append(new_instance)

        return updated_DCFrame, copy.deepcopy(instance_key_DB), rest_add_lst
    
    
    # _add_no_instance_item2DCFrame()
    # ------------------------------------------------------------------------------------------------------------------------------
    def _add_no_instance_item2DCFrame(self, baseDCFrame:dict, add_lst:list):
        """ update DCFrame about no_instace_item: goal_and_ideal
        Args:
            baseDCFrame (dict): previous DCFrame
            add_lst (list): rest of adding item's list
        """
        updated_DCFrame = copy.deepcopy(baseDCFrame)
        rest_add_lst = []

        for concat_val in add_lst:
            # skip case [goal_and_ideal]
            if 'goal_and_ideal' not in concat_val:
                rest_add_lst.append(concat_val)
                continue
            
            # [add goal_and_ideal] value if new
            if 'goal_frame' not in updated_DCFrame.keys():  # initialize goal_frame
                updated_DCFrame['goal_frame'] = {'frame_type': 'goal_and_ideal', 'content': []}

            exist_goal_and_ideal_content_vals = updated_DCFrame['goal_frame'].get('content')
            _ftype, _attr, _attr_idx, _val = concat_val.split(self.concat_sp_char)
            if _val not in exist_goal_and_ideal_content_vals:
                updated_DCFrame['goal_frame']['content'].append(_val)            
        
        return updated_DCFrame, rest_add_lst

    def _get_instanceKeyContentValue_by_instance_value(self, frame_type:str, instance_idx:int, attr:str, val:str, DCFrame:dict):
        """ get instance_key (<instance_type>, <instance_index(from1)>, <content-value_idx0>) in candidate DCFrame using instance-attribute-value
        """
        # print(f'[debug1] frame_type: {frame_type}, instance_idx: {instance_idx}, attr: {attr}, val: {val}')
        # print(f'[debug2] DCFrame(new_candidateDCFrame): {DCFrame}')
        f_instances = DCFrame.get(frame_type+'_frames', [])
        # print(f'[debug3] {f_instances=}')
        if f_instances == []:
            print(f'Error can not find {frame_type=}, {instance_idx=}, {attr=}, {val=} in {DCFrame}'); exit()
        
        def search_instance_by_instance_idx(instances:list, instance_idx:int):
            # print(f'[debug] {instances=}, {instance_idx=}')
            for instance in instances:
                if instance.get('type_instance_index') == instance_idx:
                    return instance
            return None
        
        tgt_instance = search_instance_by_instance_idx(f_instances, instance_idx)
        tgt_instance_attr_val = tgt_instance.get(attr, [])
        if tgt_instance_attr_val == []:
            print(f'Error can not find tgt_instance {frame_type=}, {instance_idx=}, {attr=}, {val=} in {DCFrame}'); exit()
        if val not in tgt_instance_attr_val:
            print(f'Error can not find `val: {val}` in {tgt_instance_attr_val=}, {DCFrame=}'); exit()
        
        # find instance-attr-value in DCFrame, then return instance_key's Content Index 0 value
        tgt_instance_contents = tgt_instance.get('content', [])

        return tgt_instance_contents[0]

    #  _add_instance_items2DCFrame()
    # ------------------------------------------------------------------------------------------------------------------------------
    def _add_instance_items2DCFrame(self, baseDCFrame:dict, instance_key_DB, candidate_DCFrame, add_lst:list):
        """
        Add attribute values to the DCFrame.
        NOTE: add_lst may include `content` attributes, because _add_instance_content only handles
        content index 0; if a candidate has multiple content values, the rest are handled here.
        add_lst is based on the candidate_DCFrame's indices and values. For each entry, this method
        locates the corresponding instance in candidate_DCFrame, looks up the updated instance_key_DB
        to determine the correct index in updated_DCFrame, and appends the value.
        NOTE: No duplicates are added to instance attribute value lists.
        Args:
            baseDCFrame (dict): base DCFrame to be extended
            instance_key_DB (dict): DB keyed by content index 0
            candidate_DCFrame (dict): tentative DCFrame produced by the LLM; used as reference to build the update
        """
        updated_DCFrame = copy.deepcopy(baseDCFrame)
        for concat_val in add_lst:
            ftype, instance_idx, attribute, attr_index, val = concat_val.split('__')

            # get instance_key(content-index0value) from candidate_DCFrame conditioned by instance_idx, attribute, val
            key_content_val_candidateDCFrame = self._get_instanceKeyContentValue_by_instance_value(ftype, int(instance_idx), attribute, val, candidate_DCFrame)
            # print(f'{concat_val=} -> searched key_content_val_candidateDCFrame: {key_content_val_candidateDCFrame}')

            # get instance_index (in updated DCFrame) using key_content_val_candidateDCFrame
            def search_instance_key_DB_by_content_val(instance_key_DB:dict, ftype:str, key_content_val:str):
                """
                Returns:
                    key (tuple): (<instance_type>, <instance_index(from1)>, <content-value_idx0>, <content-value0>)
                """
                for key, val in instance_key_DB.items():
                    if key[0] == ftype and key[2] == key_content_val:
                        return key
                self.log.error(f'Error: can not find key_content_val: {key_content_val} in instance_key_DB')
                return None
            
            updated_DCFrame_instance_index = search_instance_key_DB_by_content_val(instance_key_DB, ftype, key_content_val_candidateDCFrame)[1]    # (<instance_type>, <instance_index(from1)>, <content-value_idx0>, <content-value0>)
            # print(f'{updated_DCFrame_instance_index=}')
            
            # add-diff's val to instance's attribute values on updated_DCFrame
            def get_vals_by_type_instance_index_in_updated_DCFrame(updated_DCFrame:dict, ftype:str, type_instance_index:int):
                """Retrieve the instance with the specified type_instance_index from updated_DCFrame.
                """
                frame_access_key_in_DCFrame = ftype + '_frames'
                type_instances = updated_DCFrame.get(frame_access_key_in_DCFrame, None)
                if type_instances == None:
                    self.log.error(f'Error: can not find {ftype=} in {updated_DCFrame=}')   # should have been created by _add_instance_content
                for _instance in type_instances:
                    if _instance.get('type_instance_index') == type_instance_index:
                        return _instance
                

            target_instance = get_vals_by_type_instance_index_in_updated_DCFrame(updated_DCFrame, ftype, updated_DCFrame_instance_index)
            if target_instance == None:
                print(f"Error: can not find target_instance {concat_val=}, {updated_DCFrame_instance_index=}"); exit()      # instanceはすでに存在しているはず

            target_inst_attr_vals = target_instance.get(attribute, [])
            if val not in target_inst_attr_vals:
                target_inst_attr_vals.append(val)
                target_instance[attribute] = target_inst_attr_vals      # update updated_DCFrame
            else:
                self.log.warning(f'Warning: {val=} already exists in {updated_DCFrame_instance_index=} - `target_inst_attr_vals`')
                # self.log.warning(f'Warning: {val=} already exists in {target_inst_attr_vals=}')
            
        return updated_DCFrame

    def adding_base_update_DCFrame(self, base_DCFrame, new_candidate_DCFrame, verbose=False):
        """
        Diff the base_DCFrame and the candidates, and create updated DCFrame based on the base_DCFrame with new items added.
        0. make instance DB key -> instance `content` index 0 value
        1. 
        """
        # Prepare
        # ------------------------------------------------------------------------------------------------------------------------------ 

        # Flatten key-val Format: <frames_type>__<instance_type>__<instance_idx>__<key>__<list_idx>__<value>
        
        # Filter new_candidate_DCFrame
        _flat_new_candidate_DCFrame = self.DCFrameDiffUtils.get_flat_key_value(new_candidate_DCFrame)
        _flat_new_candidate_DCFrame, reject_lst = self.DCFrameDiffUtils.filter_incosistency_at_framesType_and_instanceType(_flat_new_candidate_DCFrame)
        
        # modifiy format: <instance_type>__<instance_index>__<key>__<list_idx>__<value>
        _flat_base_DCFrame = self.DCFrameDiffUtils.get_flat_key_value(base_DCFrame)
        _flat_base_DCFrame = self.DCFrameDiffUtils.rm_framesType_from_flat_key(_flat_base_DCFrame)
        _flat_new_candidate_DCFrame = self.DCFrameDiffUtils.rm_framesType_from_flat_key(_flat_new_candidate_DCFrame)

        ### get diff of DCFrame
        add_lst, rm_lst, keep_lst = self.DCFrameDiffUtils.get_DCFrame_diff(_flat_base_DCFrame, _flat_new_candidate_DCFrame)
        self.log.debug(f'filtered {add_lst=}')
        self.log.debug(f'filtered {rm_lst=}')

        # 0. build instance's key dict (key: (<instance_type>, <instance_index(from1)>, content-<value_idx0>-<value0>)) from base DCFrame
        # ------------------------------------------------------------------------------------------------------------------------------
        instance_key_DB = self._build_instance_key_using_contentIdx0Value(base_DCFrame)
        if verbose:
            print(f'----------------- 0. build instance key DB Result -----------------')
            pprint.pprint(instance_key_DB)

        ### adding update on base_DCFrame
        # 1. deal instance's `content` value
        # ------------------------------------------------------------------------------------------------------------------------------
        updated_DCFrame, instance_key_DB, rest_add_lst = self._add_instance_content(instance_key_DB, base_DCFrame, add_lst)
        if verbose:
            print(f'----------------- 1. deal instance content Result -----------------')
            pprint.pprint(updated_DCFrame); print('=== updated instance_key_DB ==='); pprint.pprint(instance_key_DB)
        
        # 2. deal goal_frameZ
        # ------------------------------------------------------------------------------------------------------------------------------
        updated_DCFrame, rest_add_lst = self._add_no_instance_item2DCFrame(updated_DCFrame, rest_add_lst)
        if verbose:
            print(f'----------------- 2. deal goal_frame Result -----------------')
            pprint.pprint(updated_DCFrame)
            print(f'{rest_add_lst=}')

        # 3. deal other instance's attribute-values
        # ------------------------------------------------------------------------------------------------------------------------------
        updated_DCFrame = self._add_instance_items2DCFrame(updated_DCFrame, instance_key_DB, new_candidate_DCFrame, rest_add_lst)
        if verbose:
            print(f'----------------- 3. deal other attributes Result -----------------')
            pprint.pprint(updated_DCFrame); print('=== updated instance_key_DB ==='); pprint.pprint(instance_key_DB)

        self.log.debug(f'{updated_DCFrame=}')
        return updated_DCFrame


# TEST
def add_test_1():
    p = {}
    n = {'goal_frame': {'frame_type': 'goal_and_ideal', 'content': ['want to reach a healthy body weight', 'want to enjoy wearing clothes']}, 'problem_and_trouble_frames': [{'frame_type': 'problem_and_trouble', 'type_instance_index': 1, 'content': ['eat too much', 'like delicious food'], 'detail': ['detail1val'], 'harm_effect': [], 'necessity_to_improve': []}, {'frame_type': 'problem_and_trouble', 'type_instance_index': 2, 'content': ['tend to accumulate stress', 'bottle up stress alone'], 'detail': [], 'harm_effect': [], 'necessity_to_improve': []}]}
    Ctr = AddingDCFrameUpdateController()
    Ctr.adding_base_update_DCFrame(p, n)

def add_test_duplicate_content():

    p = {'goal_frame': {'frame_type': 'goal_and_ideal', 'content': ['want to reach a healthy body weight', 'want to enjoy wearing clothes']}, 'problem_and_trouble_frames': [{'frame_type': 'problem_and_trouble', 'type_instance_index': 1, 'content': ['eat too much', 'like delicious food'], 'detail': ['detail1val'], 'harm_effect': [], 'necessity_to_improve': []}, {'frame_type': 'problem_and_trouble', 'type_instance_index': 2, 'content': ['tend to accumulate stress', 'bottle up stress alone'], 'detail': [], 'harm_effect': [], 'necessity_to_improve': []}]}

    n = {'goal_frame': {'frame_type': 'goal_and_ideal', 'content': ['want to reach a healthy body weight', 'want to enjoy wearing clothes']}, 'problem_and_trouble_frames': [{'frame_type': 'problem_and_trouble', 'type_instance_index': 1, 'content': ['eat too much', 'like delicious food'], 'detail': ['detail1val'], 'harm_effect': [], 'necessity_to_improve': []}, {'frame_type': 'problem_and_trouble', 'type_instance_index': 2, 'content': ['tend to accumulate stress', 'bottle up stress alone'], 'detail': [], 'harm_effect': [], 'necessity_to_improve': []}, {'frame_type': 'problem_and_trouble', 'type_instance_index': 3, 'content': ['eat too much', 'same content key but different val'], 'detail': [], 'harm_effect': [], 'necessity_to_improve': []}]}
    Ctr = AddingDCFrameUpdateController()
    Ctr.adding_base_update_DCFrame(p, n, True)  

def add_annomi_s121_104():
    # inconsistenfy at framesType and instanceType  after: improvement_plan_frames -> frame_type: 'problem_and_trouble'
    p = {"goal_frame": {"frame_type": "goal_and_ideal", "content": []}, "problem_and_trouble_frames": [{"frame_type": "problem_and_trouble", "type_instance_index": 1, "content": ["Possible diabetes 2"], "detail": ["Revealed during a checkup with GP", "Client feels perfectly well", "Client understands people with diabetes are usually plumper than they are", "Client is here on GP's command", "GP will put client on a program if condition worsens", "Client feels absolutely fine now", "Diagnosis came as a surprise", "Client is very surprised", "Diagnosis came out of the blue", "Client doesn't fit the typical Type II diabetes look", "Client understands that people with diabetes are usually overweight", "Client is not overweight", "Client looks after themselves to a limited extent", "Client is a busy person and burns off calories", "Client's lifestyle doesn't fit with Type II diabetes", "Client thought GP must be joking", "Client doesn't think GP was wasting their time", "GP has a lot to do", "Tests have come through", "Client has signs of diabetes 2", "Client doesn't feel like they have diabetes", "Physically, client feels absolutely fine", "Client didn't see anything wrong with themselves at all", "Client gets around and sleeps pretty well", "Client is compos mentis", "Client hasn't done much research", "Client has been busy", "GP mentioned something about family history", "Client thinks GP said something about it being hereditary", "GP mentioned it might be a family thing", "GP mentioned it might be a family thing", "Client thinks it could be hereditary", "Client's father had diabetes", "Client's mother mentioned that client's father had diabetes 2", "Client's father's brother had diabetes 2", "Client's father had diabetes 2", "Client's father's brother had diabetes 2", "Client hasn't seen his father's brother for years", "Client's father's brother might have diabetes 2", "Client's father's brother might have diabetes 2 because he is a pretty cuddly guy", "Client's father's brother is pretty big", "Client's father's brother fits the stereotype of diabetes 2", "Client's father's brother is stereotypical", "Client hasn't seen his father's brother for years", "Client hasn't seen his father's brother for many years", "Client's father's brother doesn't look after himself well", "Client's father's brother is a little bit bigger", "Client's father's brother fits the stereotype of diabetes 2", "Client is aware of family history of diabetes", "Client acknowledges diabetes is hereditary", "Client acknowledges diabetes is hereditary", "Client acknowledges diabetes is hereditary", "Client acknowledges diabetes is hereditary", "Client doesn't know what causes diabetes 2", "Diagnosis came upon client from the GP and took them back", "Client doesn't know what causes diabetes 2", "Client thought hereditary factors might be the explanation", "Client doesn't know what causes diabetes 2", "Client thought hereditary factors might be the explanation", "Client doesn't know what causes diabetes 2", "Client thought hereditary factors might be the explanation", "Client thought hereditary factors might be the explanation", "Client thought hereditary factors might be the explanation", "Client's family members with diabetes lived long lives", "Relatives relatively went into their 70s", "Relatives relatively went into their 70s", "Relatives relatively went into their 70s", "Relatives relatively went into their 70s", "Client mentions insulin balances out excess in the body", "Insulin balances out excess sugars in the body", "Insulin digests excessive sugars in the body", "Insulin taking off the excess", "Insulin digesting excessive sugars in the body", "Insulin digests excessive sugars in the body", "Client spends more time with old mates in the bar", "Client doesn't leap over five bar gates like they used to", "Client spends more time at the rugby club", "Client goes for easier fish like snapper instead of kings", "Client spends more time at the rugby club and such like", "Client goes for easier fish like snapper instead of kings", "Client has a slightly more sedate lifestyle", "Client is less demanding on their body", "Client doesn't want to do these things anymore", "Client lets the youngsters get home", "Client is less demanding on their body", "Client doesn't really want to do these things anyway", "Client lets the youngsters get home", "Client is less demanding on their body", "Client doesn't really want to do these things anyway", "Client lets the youngsters get home", "Client doesn't really notice", "Client is coming back to the diagnosis from GP", "GP said client has diabetes 2", "Client wonders if they should feel like they have diabetes", "Client is less demanding on their body", "Client doesn't really want to do these things anyway", "Client lets the youngsters get home", "Client doesn't really notice", "Client is coming back to the diagnosis from GP", "GP said client has diabetes 2", "Client wonders if they should feel like they have diabetes", "Client is a busy person", "Client works in insurance", "Client is always hopping from one place to another place", "Client hasn't noticed anything wrong"]}, {"frame_type": "problem_and_trouble", "type_instance_index": 2, "content": ["Gets out of breath sometimes"], "detail": [], "harm_effect": [], "necessity_to_improve": []}, {"frame_type": "problem_and_trouble", "type_instance_index": 3, "content": ["Lack of regular exercise"], "detail": ["Client doesn't do any regular form of exercise anymore"], "harm_effect": [], "necessity_to_improve": []}], "problem_and_trouble_links": [], "experience_frames": [], "improvement_plan_frames": []}
    
    n_104 = {"goal_frame": {"frame_type": "goal_and_ideal", "content": []}, "problem_and_trouble_frames": [{"frame_type": "problem_and_trouble", "type_instance_index": 1, "content": ["Possible diabetes 2"], "detail": ["Revealed during a checkup with GP", "Client feels perfectly well", "Client understands people with diabetes are usually plumper than they are", "Client is here on GP's command", "GP will put client on a program if condition worsens", "Client feels absolutely fine now", "Diagnosis came as a surprise", "Client is very surprised", "Diagnosis came out of the blue", "Client doesn't fit the typical Type II diabetes look", "Client understands that people with diabetes are usually overweight", "Client is not overweight", "Client looks after themselves to a limited extent", "Client is a busy person and burns off calories", "Client's lifestyle doesn't fit with Type II diabetes", "Client thought GP must be joking", "Client doesn't think GP was wasting their time", "GP has a lot to do", "Tests have come through", "Client has signs of diabetes 2", "Client doesn't feel like they have diabetes", "Physically, client feels absolutely fine", "Client didn't see anything wrong with themselves at all", "Client gets around and sleeps pretty well", "Client is compos mentis", "Client hasn't done much research", "Client has been busy", "GP mentioned something about family history", "Client thinks GP said something about it being hereditary", "GP mentioned it might be a family thing", "GP mentioned it might be a family thing", "Client thinks it could be hereditary", "Client's father had diabetes", "Client's mother mentioned that client's father had diabetes 2", "Client's father's brother had diabetes 2", "Client's father had diabetes 2", "Client's father's brother had diabetes 2", "Client hasn't seen his father's brother for years", "Client's father's brother might have diabetes 2", "Client's father's brother might have diabetes 2 because he is a pretty cuddly guy", "Client's father's brother is pretty big", "Client's father's brother fits the stereotype of diabetes 2", "Client's father's brother is stereotypical", "Client hasn't seen his father's brother for years", "Client hasn't seen his father's brother for many years", "Client's father's brother doesn't look after himself well", "Client's father's brother is a little bit bigger", "Client's father's brother fits the stereotype of diabetes 2", "Client is aware of family history of diabetes", "Client acknowledges diabetes is hereditary", "Client acknowledges diabetes is hereditary", "Client acknowledges diabetes is hereditary", "Client acknowledges diabetes is hereditary", "Client doesn't know what causes diabetes 2", "Diagnosis came upon client from the GP and took them back", "Client doesn't know what causes diabetes 2", "Client thought hereditary factors might be the explanation", "Client doesn't know what causes diabetes 2", "Client thought hereditary factors might be the explanation", "Client doesn't know what causes diabetes 2", "Client thought hereditary factors might be the explanation", "Client thought hereditary factors might be the explanation", "Client thought hereditary factors might be the explanation", "Client's family members with diabetes lived long lives", "Relatives relatively went into their 70s", "Relatives relatively went into their 70s", "Relatives relatively went into their 70s", "Relatives relatively went into their 70s", "Client mentions insulin balances out excess in the body", "Insulin balances out excess sugars in the body", "Insulin digests excessive sugars in the body", "Insulin taking off the excess", "Insulin digesting excessive sugars in the body", "Insulin digests excessive sugars in the body", "Client spends more time with old mates in the bar", "Client doesn't leap over five bar gates like they used to", "Client spends more time at the rugby club", "Client goes for easier fish like snapper instead of kings", "Client spends more time at the rugby club and such like", "Client goes for easier fish like snapper instead of kings", "Client has a slightly more sedate lifestyle", "Client is less demanding on their body", "Client doesn't want to do these things anymore", "Client lets the youngsters get home", "Client is less demanding on their body", "Client doesn't really want to do these things anyway", "Client lets the youngsters get home", "Client is less demanding on their body", "Client doesn't really want to do these things anyway", "Client lets the youngsters get home", "Client doesn't really notice", "Client is coming back to the diagnosis from GP", "GP said client has diabetes 2", "Client wonders if they should feel like they have diabetes", "Client is less demanding on their body", "Client doesn't really want to do these things anyway", "Client lets the youngsters get home", "Client doesn't really notice", "Client is coming back to the diagnosis from GP", "GP said client has diabetes 2", "Client wonders if they should feel like they have diabetes", "Client is a busy person", "Client works in insurance", "Client is always hopping from one place to another place", "Client hasn't noticed anything wrong"]}, {"frame_type": "problem_and_trouble", "type_instance_index": 2, "content": ["Gets out of breath sometimes"], "detail": [], "harm_effect": [], "necessity_to_improve": []}], "problem_and_trouble_links": [], "experience_frames": [], "improvement_plan_frames": [{"frame_type": "problem_and_trouble", "type_instance_index": 3, "content": ["Lack of regular exercise"], "detail": ["Client doesn't do any regular form of exercise anymore", "Client doesn't do any regular form of exercise anymore"], "harm_effect": [], "necessity_to_improve": []}]}
    Ctr = AddingDCFrameUpdateController()
    Ctr.adding_base_update_DCFrame(p, n_104, verbose=False)


def rm_test_1():
    """ NOTE: Deletion is not handled by this module."""
    p = {'goal_frame': {'frame_type': 'goal_and_ideal', 'content': ['want to reach a healthy body weight', 'want to enjoy wearing clothes']}, 'problem_and_trouble_frames': [{'frame_type': 'problem_and_trouble', 'type_instance_index': 1, 'content': ['eat too much', 'like delicious food'], 'detail': ['detail1val'], 'harm_effect': [], 'necessity_to_improve': []}, {'frame_type': 'problem_and_trouble', 'type_instance_index': 2, 'content': ['tend to accumulate stress', 'bottle up stress alone'], 'detail': [], 'harm_effect': [], 'necessity_to_improve': []}]}
    n = {'goal_frame': {'frame_type': 'goal_and_ideal', 'content': ['want to reach a healthy body weight', 'want to enjoy wearing clothes']}, 'problem_and_trouble_frames': [{'frame_type': 'problem_and_trouble', 'type_instance_index': 1, 'content': ['eat too much', 'like delicious food'], 'detail': ['detail1val'], 'harm_effect': [], 'necessity_to_improve': []}, {'frame_type': 'problem_and_trouble', 'type_instance_index': 2, 'content': ['tend to accumulate stress'], 'detail': [], 'harm_effect': [], 'necessity_to_improve': []}]}
    Ctr = AddingDCFrameUpdateController()
    Ctr.adding_base_update_DCFrame(p, n)

def add_test_using_LLM_extracted_DCFrame():
    """ Integration test using one session's LLM-extracted DCFrame sequence."""
    test_ExtractedDCFrame_jsonl = '../../LLMs/extract_DCFrame/outputs/en_AnnoMI_full_extract_DCFrame/_out_tmp/1step_SrcCtx5_prompt20241009/instanced_ctx5_conbined_121_from91temp0.5.jsonl'
    err_log_path = 'err_log.txt'
    out_LLM_DCFrame_key = 'DCFrame_out'
    loaded_extracted_DCFrame_list = []
    import json
    import traceback
    with open(test_ExtractedDCFrame_jsonl, 'r') as f:
        for line in f:
            loaded_extracted_DCFrame_list.append(json.loads(line))
    err_of = open(err_log_path, 'w')

    previous_DCFrame = {}
    Ctr = AddingDCFrameUpdateController()

    for line_num, candidate_src_tgt_dict in enumerate(loaded_extracted_DCFrame_list):
        print(f'processing... line_num: {line_num+1} / {len(loaded_extracted_DCFrame_list)}')
        updated_DCFrame = Ctr.adding_base_update_DCFrame(previous_DCFrame, candidate_src_tgt_dict[out_LLM_DCFrame_key]['function_res']['arguments'])  # NOTE Check store structure
        """
        try:
            updated_DCFrame = Ctr.adding_base_update_DCFrame(previous_DCFrame, candidate_src_tgt_dict[out_LLM_DCFrame_key]['function_res']['arguments'])  # NOTE Check store structure
            
        except Exception as e:
            line_info = f'Error: {e}\n dealing... {line_num+1}, /{len(loaded_extracted_DCFrame_list)}'
            err_trace = traceback.format_exc()
            err_of.write(f"{line_info}\n{err_trace}\n")
        """

        print(f'updated_DCFrame: {updated_DCFrame}')

        # prepare fon next instance
        previous_DCFrame = updated_DCFrame

        # break
        if line_num+1 == 200:
            break

if __name__ == "__main__":
    ### Project root
    PROJ_PATH = str(Path(__file__).resolve().parents[2])
    sys.path.append(PROJ_PATH)
    logging.config.fileConfig(PROJ_PATH +"/logging.conf")

    # add_test_1()
    # add_test_duplicate_content()
    add_annomi_s121_104()
    # add_test_using_LLM_extracted_DCFrame()

    # rm_test_1()
