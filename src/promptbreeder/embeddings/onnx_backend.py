from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from .embedding_base import EmbeddingModelBase

class ONNXEmbeddingModel(EmbeddingModelBase):
    def __init__(
        self,
        model_name: str = "bge-micro",
        max_length: int = 512,
        local_path=None,
    ):
        super().__init__()
        self.model_registry = {
            "bge-large": {
                "repo": "neuralmagic/bge-small-en-v1.5-quant",
                "remote_file": "model.onnx",
                "tokenizer": "BAAI/bge-large-en-v1.5",
            },
            "bge-base": {
                "repo": "neuralmagic/bge-base-en-v1.5-quant",
                "remote_file": "model.onnx",
                "tokenizer": "BAAI/bge-base-en-v1.5",
            },
            "bge-small": {
                "repo": "neuralmagic/bge-small-en-v1.5-quant",
                "remote_file": "model.onnx",
                "tokenizer": "BAAI/bge-small-en-v1.5",
            },
            "gte-small": {
                "repo": "TaylorAI/galactic-models",
                "remote_file": "model_quantized.onnx",
                "tokenizer": "Supabase/gte-small",
            },
            "minilm-l6": {
                "repo": "Supabase/all-MiniLM-L6-v2",
                "remote_file": "onnx/model_quantized.onnx",
                "tokenizer": "Supabase/all-MiniLM-L6-v2",
            },
            "gte-tiny": {
                "repo": "TaylorAI/galactic-models",
                "remote_file": "gte-tiny.onnx",
                "tokenizer": "Supabase/gte-small",
            },
            "bge-micro": {
                "repo": "TaylorAI/galactic-models",
                "remote_file": "bge-micro-v2.onnx",
                "tokenizer": "BAAI/bge-small-en-v1.5",
            }
        }
        if local_path is None:
            local_path = hf_hub_download(
                self.model_registry[model_name]["repo"],
                filename=self.model_registry[model_name]["remote_file"],
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_registry[model_name]["tokenizer"]
        )
        self.tokenizer.model_max_length = 1_000_000 # avoids printing stupid warning
        self.max_length = max_length
        self.providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(local_path, providers=self.providers)

    def embed(
        self,
        text: str,
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        input = self.split_and_tokenize_single(
            text, pad=pad, split_strategy=split_strategy
        )
        outs = []
        for seq in range(len(input["input_ids"])):
            if not pad:
                assert (
                    np.mean(input["attention_mask"][seq]) == 1
                ), "pad=False but attention_mask has 0s"
            out = self.session.run(
                None,
                {
                    "input_ids": input["input_ids"][seq : seq + 1],
                    "attention_mask": input["attention_mask"][seq : seq + 1],
                    "token_type_ids": input["token_type_ids"][seq : seq + 1],
                },
            )[0]  # 1, seq_len, hidden_size
            trimmed = out[
                0, np.array(input["attention_mask"][seq]) == 1, :
            ]  # chunk_seq_len, hidden_size
            outs.append(trimmed)
        outs = np.concatenate(outs, axis=0)  # full_seq_len, hidden_size
        avg = np.mean(outs, axis=0)  # hidden_size
        if normalize:
            avg = avg / np.linalg.norm(avg)
        return avg

    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        result = []
        for text in texts:
            result.append(
                self.embed(
                    text,
                    normalize=normalize,
                    pad=pad,
                    split_strategy=split_strategy,
                )
            )

        return np.array(result)
