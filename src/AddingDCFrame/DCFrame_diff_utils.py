# -*- coding: utf-8 -*-

"""
DCFrame Diff Utilities

This module provides utilities for comparing (diffing) two DCFrame dictionaries.

A DCFrame is a structured dict containing several frame types:
  - goal_frame            : a single instance describing the goal/ideal state
  - problem_and_trouble_frames : a list of problem instances
  - experience_frames     : a list of experience instances
  - improvement_plan_frames   : a list of improvement-plan instances

Core workflow:
  1. Flatten a DCFrame into a list of concatenated strings with the format
       <frames_type>__<instance_type>__<instance_idx>__<key>__<list_idx>__<value>
  2. Compute set-based diffs (added / removed / kept) between two flat lists.
  3. Filter or decompose the resulting diff entries for downstream use.
"""

import os, sys
from pathlib import Path
import pprint
import logging, logging.config

class DCFrameDiffUtils(object):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.concat_sp_char = '__'

    # Dealing format: <frames_type>__<instance_type>__<instance_idx>__<key>__<list_idx>__<value>
    # -----------------------------------------------------------------------------------------------
    ### Frame Level
    def get_flat_key_value(self, DCFrame:dict):
        """
        Returns:
            list: ["<frames_type>__<instance_type>__<instance_idx>__<key>__<list_idx>__<value>", ...]
        """
        flat_DCFrame_key_val = []
        for frames_type, val in DCFrame.items():
            # Str-Values
            if frames_type == 'problem_and_trouble_links':
                pass
            
            # 1 Dict-Value
            elif frames_type == 'goal_frame':
                flat_vals = self.get_flat_instance_key_value(val)
                # Error case
                if flat_vals == -1:
                    self.log.error(f'Error -1 in goal_frame then skip: {val=}')
                    continue
                elif flat_vals == -2:
                    self.log.error(f'Error -2 in Other-frame then skip: {val=}')
                    continue
                flat_vals = [frames_type + self.concat_sp_char + val for val in flat_vals]  # add frames_type on head
                # print(f'flat-1dict type:{frame_type} -> {flat_vals}') # debug
                flat_DCFrame_key_val.extend(flat_vals)

            # Multi Instance-Value
            elif frames_type in ('problem_and_trouble_frames', 'experience_frames', 'improvement_plan_frames'):
                flat_vals = self.get_multi_instance_key_value(val)
                flat_vals = [frames_type + self.concat_sp_char + val for val in flat_vals]  # add frames_type on head
                # print(f'flat-Multi-Instance type:{frame_type} -> {flat_vals}')    # debug
                flat_DCFrame_key_val.extend(flat_vals)
                
            else:
                self.log.error(f'Unknown Frame Type: {frames_type}')
                raise ValueError(f'Unknown Frame Type: {frames_type}')
        return flat_DCFrame_key_val
    
    def rm_framesType_from_flat_key(self, concat_key_val_lst:list):
        """
         <frames_type>__<instance_type>__<instance_idx>__<key>__<list_idx>__<value>
         ->
        <frames_type>__<instance_type>__<instance_idx>__<key>__<list_idx>__<value>
        """
        rm_framesType_lst = []
        for concat_str in concat_key_val_lst:
            dec_str = concat_str.split(self.concat_sp_char)
            rm_framesType_lst.append(self.concat_sp_char.join(dec_str[1:]))

        return rm_framesType_lst

    ### Key-Value List Level
    def concatenate_key_values(self, key, values):
        """ 
        Current format: element: key + __ + list_idx + __ + value
        Returns:
            tuple: (KEY__VAL1, KEY__VAL2, ...)"""
        # return tuple([key + self.concat_sp_char + value for value in values]) # NOTE: without index
        return tuple([key + self.concat_sp_char + str(i) + self.concat_sp_char + value for i, value in enumerate(values)]) # NOTE: with index

    ### Instance Level
    def get_flat_instance_key_value(self, DCInst: dict):
        """
        <frame_type(inside of Instance)>__<instance_idx(1start)__<key>__<list_idx>__<value>

        Returns:
            Error: -1: type_instnace_index is None (goal_frame) & frame_type is None
        """
        flat_key_val_tuples = []
        # deal 1 DCFrame instance
        keys_has_list = tuple(set(DCInst.keys()) - set(('frame_type', 'type_instance_index'))) # keys that do not hold a list as value NOTE: current
        frameIdx = DCInst.get('type_instance_index', None)
        if frameIdx is None:
            # <frame_type>
            frameType_Idx = DCInst.get('frame_type', None)        # case: goal_and_ideal
            if frameType_Idx is None:
                self.log.error(f'frameIdx is None & frame_type is None: {DCInst=}')
                return -1
        else:
            # <frame_type>__<type_instance_index>
            _frameType = DCInst.get('frame_type', None)
            if _frameType is None:
                self.log.error(f'frame_type is None: {DCInst=}')
                return -2
            frameType_Idx = DCInst['frame_type'] + self.concat_sp_char + str(frameIdx)  # case: other Frame-type

        for _key in keys_has_list:  # keys that hold a list as value
            key_val_tuple = self.concatenate_key_values(_key, DCInst[_key])
            if len(key_val_tuple) == 0:
                continue
            else:
                for key_val in key_val_tuple:
                    flat_key_val_tuples.append(frameType_Idx + self.concat_sp_char + key_val)

        return flat_key_val_tuples
    
    def get_multi_instance_key_value(self, DCInstances:list):
        # Multi instance loop
        multi_instance_flat_key_val_tuples = []    # stores key-values for all instances
        for DCInst in DCInstances:
            multi_instance_flat_key_val_tuples.extend(self.get_flat_instance_key_value(DCInst))
        return multi_instance_flat_key_val_tuples
    
    
    
    # Filter
    # -----------------------------------------------------------------------------------------------

    def filter_incosistency_at_framesType_and_instanceType(self, flat_key_val_lst:list):
        filtered_lst = []
        reject_lst = []
        pair_framesType_instanceType = {'goal_frame': 'goal_and_ideal',
                                        'problem_and_trouble_frames': 'problem_and_trouble',
                                        'experience_frames': 'experience',
                                        'improvement_plan_frames': 'improvement_plan'
                                        }
        for concat_str in flat_key_val_lst:
            dec_str = concat_str.split(self.concat_sp_char)
            if dec_str[0] == 'goal_frame':
                if len(dec_str) != 5:
                    print(f"found goal_and_ideal str len !=5, {concat_str=}"); exit()


            elif len(dec_str) != 6:
                print(f"found decomposed str len !=6, {concat_str=}"); exit()
            
            frames_type = dec_str[0]
            frame_instance_type = dec_str[1]
            if pair_framesType_instanceType.get(frames_type) != frame_instance_type:
                reject_lst.append(concat_str)
            else:
                filtered_lst.append(concat_str)
        
        if len(reject_lst) > 0:
            self.log.warning(f"rejected lst (framesType != isntanceType): {reject_lst}")
        return filtered_lst, reject_lst




    def decompose_concatenate(self, concat_str_lst:str):
        """Decompose frame-instance-key-val that has been flattened into a single string.
        """
        decomposed_dict = {'goal_and_ideal':[], 'problem_and_trouble_frames':[], 
                           'experience_frames':[], 'improvement_plan_frames':[]}
        for concat_str in concat_str_lst:
            div_t = concat_str.split(self.concat_sp_char)
            if 'goal_and_ideal' in concat_str:
                if len(div_t) != 4:
                    self.log.error(f'div concat_str != 4 case. {concat_str=}')
                else:
                    decomposed_dict['goal_and_ideal'].append(div_t)

            elif 'problem_and_trouble_links' in concat_str:
                continue    # TODO
            else:
                if len(div_t) != 5:
                    self.log.error(f'div concat_str != 5 case. {concat_str=}')
                else:
                    decomposed_dict[div_t[0]].append(div_t)

        return decomposed_dict


    ### diff
    def get_DCFrame_diff(self, before_lst:list, after_lst:list) -> tuple:
        """
        Args:
            before_lst (list): flat_key_value list
            after_lst (list):    
        Returns:
        """
        # return list(diff(before_lst, after_lst))
        add_lst = list(set(after_lst) - set(before_lst))
        rm_lst = list(set(before_lst) - set(after_lst))
        keep_lst = list(set(before_lst) & set(after_lst))
        return add_lst, rm_lst, keep_lst
    
    def filter_by_frame_and_content_key_index0(self, diff_lst):
        """Filter items containing 'content' with index=0.
            Excludes goal_and_ideal.
        """
        filtered_diff_lst = []
        for diff_el in diff_lst:
            if 'goal_and_ideal' in diff_el:
                continue
            elif 'content__0' in diff_el:
                filtered_diff_lst.append(diff_el)
            else:
                continue

        return filtered_diff_lst
    

# ----------------  TEST DCFrame Diff
def test_filter():
    p = {"goal_frame": {"content": ["A", "B"]}}
    DiffUtils = DCFrameDiffUtils()
    concat_p = DiffUtils.get_flat_key_value(p)


def diff_test1():    
    # test: add problem-content-value
    p = {"goal_frame": {"frame_type": "goal_and_ideal", "content": ["want to reach a healthy body weight"]}, "problem_and_trouble_frames": [{"frame_type": "problem_and_trouble", "type_instance_index": 1, "content": ["eat too much", "like delicious food", "eat too much oily Chinese food", "cannot improve eating habits"], "detail": [], "harm_effect": [], "necessity_to_improve": []}], "problem_and_trouble_links": [], "experience_frames": [], "improvement_plan_frames": []}
    n = {"goal_frame": {"frame_type": "goal_and_ideal", "content": ["want to reach a healthy body weight"]}, "problem_and_trouble_frames": [{"frame_type": "problem_and_trouble", "type_instance_index": 1, "content": ["eat too much", "like delicious food", "eat too much oily Chinese food", "cannot improve eating habits", "NEW_ADD"], "detail": [], "harm_effect": [], "necessity_to_improve": []}], "problem_and_trouble_links": [], "experience_frames": [], "improvement_plan_frames": []}
    print(f'--- previous DCFrame\n{p}\n--- current DCFrame\n{n}\n--- diff')
    DiffUtils = DCFrameDiffUtils()
    concat_p = DiffUtils.get_flat_key_value(p)
    concat_n = DiffUtils.get_flat_key_value(n)
    add_lst, rm_lst, keep_lst = DiffUtils.get_DCFrame_diff(concat_p, concat_n)
    print('------ add lst')
    pprint.pprint(add_lst)



if __name__ == "__main__":
    test_filter()
    diff_test1()