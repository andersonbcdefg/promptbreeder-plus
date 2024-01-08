from .embedding_base import EmbeddingModelBase
from typing import Literal

try:
    import mlx.core as mx
    import mlx.nn as nn
except:
    print("MLX not installed. To use, you must be on Apple silicon and run `pip install mlx`.")

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy
import numpy as np
from mlx.utils import tree_unflatten
from transformers import BertTokenizer, BertConfig, AutoModel

def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key

class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = nn.MultiHeadAttention(config.hidden_size, config.num_attention_heads, bias=True)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config)
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        # self.pooler = nn.Linear(config.intermediate_size, config.vocab_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        if token_type_ids is None:
            # just make it all 0s
            token_type_ids = mx.zeros_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask) # shape: B, L, D
        return y
    
    @classmethod
    def from_hugging_face(cls, model_path: str):
        config = BertConfig.from_pretrained(model_path)
        torch_weights = AutoModel.from_pretrained(model_path).state_dict()

        # figure out how to convert torch weights to mx weights
        mx_weights = {
            replace_key(key): mx.array(tensor.numpy()).astype(mx.float16) for key, tensor in torch_weights.items()
        }
        mlx_model = cls(config)
        mlx_model.update(
            tree_unflatten(list(mx_weights.items()))
        )
        nn.QuantizedLinear.quantize_module(mlx_model, bits=8)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        return mlx_model, tokenizer


class MLXEmbeddingModel(EmbeddingModelBase):
    def __init__(
        self,
        model_path: str, # path to model on huggingface. must be BERT
        max_length: int = 512
    ):
        super().__init__()
        self.model, self.tokenizer = Bert.from_hugging_face(model_path)
        self.max_length = max_length

    def embed(
        self,
        text: str,
        normalize: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        tokenized = self.split_and_tokenize_single(
            text, pad=True, split_strategy=split_strategy
        )
        outs = self.model(
            input_ids=mx.array(tokenized["input_ids"]),
            token_type_ids=mx.array(tokenized["token_type_ids"]),
            attention_mask=mx.array(tokenized["attention_mask"]),
        )
        outs = np.array(outs) # n_chunks, seq_len, hidden_size
        # first pool over the seq_len dimension
        attn_mask = np.array(tokenized["attention_mask"]) # n_chunks, seq_len
        pooled = np.sum(outs * np.expand_dims(attn_mask, -1), axis=1) / np.sum(attn_mask, axis=1, keepdims=True)

        # then average over the chunks dimension
        avg = np.mean(pooled, axis=0)
        if normalize:
            avg = avg / np.linalg.norm(avg)
        return avg

    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        tokenized = self.split_and_tokenize_batch(
            texts, pad=True, split_strategy=split_strategy
        )
        outs = self.model(
            input_ids=mx.array(tokenized["input_ids"]),
            token_type_ids=mx.array(tokenized["token_type_ids"]),
            attention_mask=mx.array(tokenized["attention_mask"]),
        )
        outs = np.array(outs) # n_chunks, seq_len, hidden_size
        
        # pool over the seq_len dimension
        attn_mask = np.array(tokenized["attention_mask"]) # b, seq_len
        pooled = np.sum(outs * np.expand_dims(attn_mask, -1), axis=1) / np.sum(attn_mask, axis=1, keepdims=True) # b, hidden_size
        
        # # use offsets to average over each text's chunks
        # assert len(offsets) == len(texts) + 1, "offsets isn't expected length"
        # result = []
        # for i in range(len(offsets) - 1):
        #     lo, hi = offsets[i], offsets[i + 1]
        #     chunk = pooled[lo:hi]
        #     avg = np.mean(chunk, axis=0)
        #     if normalize:
        #         avg = avg / np.linalg.norm(avg)
        #     result.append(avg)

        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled