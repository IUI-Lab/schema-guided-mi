# -*- coding: utf-8 -*-

# Formats Src-Tgt Policy Instances (with Counselor as target) into embedding-model input text.

from pathlib import Path
import logging
import logging.config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DialogueStateInformationFormatUtils(object):
    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    # ------- History Utterance Info --------------------------------
    def format_history_str(self, src_tgt_dict:dict):
        """Role1: Utterance1\nRole2: Utterance2\n..."""
        src_uttr_list = []
        for src_uttr_dict in src_tgt_dict['source']['n_contexts']:
            if src_uttr_dict is None:
                continue
            src_uttr_list.append(f"{src_uttr_dict['speaker']}: {src_uttr_dict['concat_utterance']}")
        src_uttr = '\n'.join(src_uttr_list)
        return src_uttr

    def format_history(self, src_tgt_dict:dict):
        """
        (History)
        Role1: Utterance1
        Role2: Utterance2
        """
        history_str = self.format_history_str(src_tgt_dict)
        return history_str
    # 
    # ------- Conbination --------------------------------
    def template_hisory_DCFrame(self, history_str, DCFrame_str):
        return f"### History\n{history_str}\n\n### Dialogue_contnet\n{DCFrame_str}"
    
    def format_history_DCFrame(self, src_tgt_dict:dict):
        """
        # Histrory
        <N_turn Utterance not include target> 

        # DCFrame
        <DCFrame>
        """
        history_str = self.format_history_str(src_tgt_dict)
        DCFrame = src_tgt_dict.get('DCFrame', None)
        if DCFrame is None:
            self.log.error(f"DCFrame is None. {src_tgt_dict}")
            raise ValueError(f"DCFrame is missing in src_tgt_dict: {src_tgt_dict}")

        return self.template_hisory_DCFrame(history_str, str(DCFrame))


    # ------- For Few-shot Prompt Format ----------------
    def sample_format_history_DCFrame_policy(self, history_str:str, DCFrame_str:str, policy_d:dict):
        intent_str = policy_d.get('intent')
        focuses_str = policy_d.get('focuses')
        seek_frame_type_str = policy_d.get('seek_frame_type')
        seek_attribute_str = policy_d.get('seek_attribute')
        
        return f"""### History
{history_str}

### Dialogue_content
{DCFrame_str}

### Counselor_policy
- intent: {intent_str}

- focuses: {focuses_str}

- seek_frame_type: {seek_frame_type_str}

- seek_attribute: {seek_attribute_str}"""



def test():
    pass

if __name__ == "__main__":
    logging.config.fileConfig(PROJECT_ROOT / "logging.conf")  # logging using logging.conf
    test()