# -*- coding: utf-8 -*-


import os, sys
from pathlib import Path
import pprint
import logging, logging.config
import json
import time
import functools
import copy
import pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


from .doc_embedding.embedding_xlm_r_multilingual import EmbeddingXLMRMultilingual

class FindNearDialogueStateController(object):

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

        self.DS_DB = {}   # {uid `str` <session_num>__<target_last_csv_idx_list> : instance_dict}

        self.DS_embeddings = None
        self.idx2uid = None
        self.embedding_matrix = None

        # embedding model 
        self.embeddingModelCtr = None

    # ------- Set Dialogue State DB --------------------------------
    def set_DS_DB_using_SrcTgtPolicyInstances(self, src_tgt_policy_instances_dir: str):
        jsonls = self.get_jsonl_files(src_tgt_policy_instances_dir)

        for jsonl_path in jsonls:
            self.log.debug('reading jsonl: ' + jsonl_path)
            uid_inst_dict = {}  # {uid `str` <session_num>__<target_last_csv_idx_list> : instance_dict}

            filename_without_ext = Path(jsonl_path).stem  # assume ~~~_session_<SESSION_NUM>.jsonl
            session_num = filename_without_ext.split("_")[-1]
            src_tgt_policy_instances = self.get_instances_from_jsonl(src_tgt_policy_instances_dir + jsonl_path)
            for inst_d in src_tgt_policy_instances:
                csv_idx_list = inst_d['target'].get('csv_idx_list', None)
                if csv_idx_list is None:
                    self.log.error(f"last_csv_idx_list is None. {inst_d['target']}")
                    exit()
                
                uid = session_num + "__" + str(csv_idx_list[-1])
                uid_inst_dict[uid] = inst_d
            
            self.DS_DB.update(uid_inst_dict)    # NOTE: assumes uid collisions cannot occur
        
        self.log.debug(f"DS_DB instance num: {len(self.DS_DB)}")
        self.log.debug(f"DS_DB keys: {list(self.DS_DB.keys())[:5]}")

    def get_jsonl_files(self, jsonl_dir:str):
        """ Get jsonl files in the directory
        Returns:
            `List` : jsonl files in the directory
        """
        jsonl_files = [f for f in os.listdir(jsonl_dir) if f.endswith('.jsonl')]
        return jsonl_files
    
    def get_instances_from_jsonl(self, json_path):
        """ Get instances from jsonl file
        Returns:
            `List` : instances in the jsonl file
        """
        with open(json_path, 'r') as f:
            instances = [json.loads(line) for line in f]
        return instances
    
    # ------- Set Dialogue Context Embeddings --------------------------------
    def set_DSembeddingMatrix_from_pickle(self, pickle_path:str):
        """
        Args: 
            pickle : {uid `str` <session_num>__<target_last_csv_idx_list> : embedding `np.array`}
        """
        with open(pickle_path, 'rb') as f:
            uid_DSembeddings_dict = pickle.load(f)
        self.log.debug(f"DS_embeddings instance num: {len(uid_DSembeddings_dict)}")
        self.log.debug(f"DS_embeddings keys: {list(uid_DSembeddings_dict.keys())[:5]}")

        _idx2uid = {}
        arr = []
        for idx, uid in enumerate(uid_DSembeddings_dict.keys()):
            _idx2uid[idx] = uid
            arr.append(uid_DSembeddings_dict[uid])
        
        embedding_matrix = np.array(arr)
        self.idx2uid = _idx2uid
        self.embedding_matrix = embedding_matrix
        self.log.debug(f"set idx2uid len: {len(self.idx2uid)}")
        self.log.debug(f"set embedding_matrix shape: {self.embedding_matrix.shape}")


    # ------- Get/ Set Embedding Model --------------------------------
    def get_embedding_model_controller(self):
        if self.embeddingModelCtr is None:
            self.log.error("Embedding model controller is not set. Please set it using set_embedding_model() method.")
            return None
        return self.embeddingModelCtr
    
    def set_embedding_model(self, tokenizer_path, model_path, device):
        _ctr = EmbeddingXLMRMultilingual()
        _ctr.set_model(tokenizer_path, model_path, device)
        self.embeddingModelCtr = _ctr

    # ------- Get Near Dialogue State --------------------------------
    def _get_similarity_idx_in_sentences_embeddings(self, query_sentence_vec, top_n=10):
        """ Compute cosine similarity between query_sentence_vec and the sentences_cls_embedding matrix
            Args:
                query_sentence_vec: 2d array : must have the same dimensionality for cosine_similarity to work
        """
        # print('query_sentence_vec shape=', query_sentence_vec.shape)  # debug
        query_similarity = cosine_similarity(query_sentence_vec, Y=self.embedding_matrix)[0]    # return similarity score with Y ndarray (sentences,)
        # print('similarity', query_similarity); print('type similarity', type(query_similarity))   # debug
        query_topn_similarity_indices = np.argsort(query_similarity)[::-1][:top_n]  # get indices sorted by descending query_similarity score
        
        topn_scores = query_similarity[query_topn_similarity_indices]
        # print(query_topn_similarity_indices); print(topn_scores)  # debug
        return query_topn_similarity_indices, topn_scores

    def get_similarity_dialogue_state(self, query_str:str, top_n:int):
        """
        Returns:
            `List` : [{'uid':str, 'similarity_score':float, 'DS_instance':dict}, ...]
        """
        self.log.debug('query_str: {}'.format(query_str))
        query_vec = np.array([self.embeddingModelCtr.get_cls_embedding_1sentence(query_str)])

        query_topn_similarity_matrix_indices, topn_scores = self._get_similarity_idx_in_sentences_embeddings(query_vec,top_n)
        
        # print('indices type: {}, el type:{}'.format(type(query_topn_similarity_matrix_indices),type(query_topn_similarity_matrix_indices[0])))    # debug
        # print('scores type: {}, el type:{}'.format(type(topn_scores),type(topn_scores[0])))   # debug

        # idx & score & annotate -> dict
        ret_list = []
        for in_matrix_idx, similarity_score in zip(query_topn_similarity_matrix_indices, topn_scores):
            uid = self.idx2uid[in_matrix_idx]
            inst_d = self.DS_DB[uid]
            ret_list.append({'uid':uid, 'similarity_score':float(similarity_score), 'DS_instance':inst_d})
        return ret_list

    def print_similarity_result(self, result_list):
        for idx, ret_d in enumerate(result_list):
            print(f"--- {idx} ---")
            print(f"uid: {ret_d['uid']}")
            print(f"similarity_score: {ret_d['similarity_score']}")
            print(f"DS_instance: {ret_d['DS_instance']}")
            print("")

