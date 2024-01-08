from abc import ABC, abstractmethod
from typing import Literal, Union
import numpy as np
import awkward as ak

class EmbeddingModelBase(ABC):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.max_length = 512
    
    def _pad(self, tokens: ak.Array, max_length: int, pad_id: int):
        tokens = ak.pad_none(tokens, target=max_length, axis=-1, clip=True)
        tokens = ak.fill_none(tokens, pad_id)
        return tokens

    def __call__(
        self,
        input: Union[str, list[str]],
    ) -> Union[list[list[float]], list[float], np.ndarray]:
        if isinstance(input, str):
            return self.embed(input)
        elif isinstance(input, list):
            return self.embed_batch(input)
        else:
            raise ValueError(f"Input must be str or list[str], not {type(input)}")

    def split_and_tokenize_single(
        self,
        text: str,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ) -> dict[str, list[list[int]]]:
        """
        Split and tokenize a single text to prepare it for the embedding model.
        Padding is only necessary if running more than 1 sequence thru the model at once.
        Splitting happens when the model exceeds the max_length (usually 512).
        You can either truncate the text, or split into chunks. Chunking can be "greedy"
        (as many 512 chunks as possible), or "even" (split into even-ish chunks with np.array_split).
        """

        # first make into tokens
        tokenized = self.tokenizer(text)  # (seq_len, )
        input_ids = tokenized["input_ids"]

        # if don't have to pad and don't have to split into chunks, we're done
        if not pad and len(input_ids) <= self.max_length:
            # wrap in list to make consistent with split_and_tokenize_batch
            return {k: [tokenized[k]] for k in tokenized}

        # handle splitting
        if split_strategy == "truncate":
            for k in tokenized:
                tokenized[k] = [tokenized[k][: self.max_length]]

        elif split_strategy == "greedy":
            for k in tokenized:
                tokenized[k] = [
                    tokenized[k][idx : idx + self.max_length]
                    for idx in range(0, len(tokenized[k]), self.max_length)
                ]

        elif split_strategy == "even":
            for k in tokenized:
                tokenized[k] = [
                    arr.tolist()
                    for arr in np.array_split(
                        tokenized[k],
                        int(np.ceil(len(tokenized[k]) / self.max_length)),
                    )
                ]

        else:
            raise ValueError(
                f"split_strategy must be 'truncate', 'greedy', or 'even', not {split_strategy}"
            )

        # pad if applicable
        if pad:
            # first make sure list is nested
            if not isinstance(tokenized["input_ids"][0], list):
                for k in tokenized:
                    tokenized[k] = [tokenized[k]]

            # get pad token
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0

            pad_len = max(
                [
                    len(tokenized["input_ids"][i])
                    for i in range(len(tokenized["input_ids"]))
                ]
            )
            for k in tokenized:
                jagged = ak.Array(tokenized[k])
                padded = self._pad(jagged, pad_len, pad_token_id)
                tokenized[k] = padded.tolist()

        return tokenized

    def split_and_tokenize_batch(
        self,
        texts: str,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ) -> dict:
        """
        Tokenize the text and pad if applicable.

        :param text: The input text to be tokenized.
        :type text: str
        :return: Returns a tuple. dictionary containing tokenized and padded 'input_ids',
        'attention_mask' and 'token_type_ids', along with a list of offsets.
        :rtype: Tuple[Dict[str, numpy.ndarray], list[int]]

        Example:

        .. code-block:: python

            tokenized_text = model.split_and_tokenize('sample text')
        """
        result = {}
        # offsets = [0]
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        if self.max_length is None:
            raise ValueError("max_length is not initialized.")

        # first tokenize without padding
        tokens = ak.Array(self.tokenizer(texts)["input_ids"])    
        pad_token_id = self.tokenizer.pad_token_id
        longest = max(len(t) for t in tokens)
        tokens = np.array(self._pad(tokens, longest, pad_token_id))
        mask = np.where(tokens == pad_token_id, 0, 1)
        token_type_ids = np.zeros_like(tokens)

        return {
            "input_ids": tokens,
            "attention_mask": mask,
            "token_type_ids": token_type_ids
        }

    @abstractmethod
    def embed(
        self, text: str, normalize: bool = False
    ) -> Union[list[float], np.ndarray]:
        pass

    @abstractmethod
    def embed_batch(
        self, texts: list[str], normalize: bool = False
    ) -> Union[list[list[float]], np.ndarray]:
        pass