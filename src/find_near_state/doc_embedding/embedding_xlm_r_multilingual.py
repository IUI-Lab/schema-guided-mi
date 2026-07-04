# -*- coding: utf-8 -*-

"""XLM-RoBERTa multilingual [CLS] embedding wrapper, plus utilities to download
the pretrained model/tokenizer used elsewhere in this package.

Requires:
uv pip install protobuf sentencepiece tiktoken transformers torch --torch-backend=auto
"""

import logging
import logging.config
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EmbeddingXLMRMultilingual(object):
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = None

    def set_model(self, tokenizer_path, model_path, device):
        # Model & Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)

        # device
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def get_cls_embedding_1sentence(self, sentence):
        """Encode a single sentence and return its [CLS] embedding.

        NOTE: truncates to 4096 tokens (including SOS/EOS); anything beyond
        that is discarded, so long inputs lose their tail.

        Returns:
            `np.array` (768(embedding size), )
        """
        MAX_LENGTH = 4096  # 4098 raised an error
        tokenized_tensor = self.tokenizer.encode(
            sentence, add_special_tokens=True, return_tensors="pt", max_length=MAX_LENGTH, truncation=True
        )  # (1, token_seq_len)
        tokenized_tensor = tokenized_tensor.to(self.device)  # cpu -> device
        with torch.no_grad():
            outputs = self.model(tokenized_tensor)
        cls_vector = outputs[0][0, 0, :]  # hidden_state (instance_idx, token_seq_idx, token's dim)

        return cls_vector.to('cpu').detach().numpy()  # move to cpu: numpy requires cpu memory

    def dl_XLMRLong(self):
        name = "markussagen/xlm-roberta-longformer-base-4096"

        model = AutoModel.from_pretrained(name)
        model_save_dir = PROJECT_ROOT / 'src' / 'find_near_state' / 'models' / 'xlmrLong'
        model.save_pretrained(model_save_dir)
        print(f'saved! [{model_save_dir}]')

        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer_save_dir = PROJECT_ROOT / 'src' / 'find_near_state' / 'models' / 'xlmrLong'
        tokenizer.save_pretrained(tokenizer_save_dir)
        print(f'saved! [{tokenizer_save_dir}]')


def dl_model_and_tokenizer():
    """Download the XLM-R Longformer model and tokenizer into models/."""
    EmbeddingCtr = EmbeddingXLMRMultilingual()
    EmbeddingCtr.dl_XLMRLong()



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # temporary setting

    dl_model_and_tokenizer()

