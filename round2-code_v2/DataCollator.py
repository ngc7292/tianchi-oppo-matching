# -*- coding: utf-8 -*-
"""
__title__="DataCollator"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
import torch

from typing import Optional, List
from transformers.tokenization_utils_base import BatchEncoding


def get_special_tokens_mask(
        tokenizer, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
) -> List[int]:
    """
    Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
    special tokens using the tokenizer ``prepare_for_model`` method.

    Args:
        token_ids_0 (:obj:`List[int]`):
            List of IDs.
        token_ids_1 (:obj:`List[int]`, `optional`):
            Optional second list of IDs for sequence pairs.
        already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the token list is already formatted with special tokens for the model.

    Returns:
        :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    """

    if already_has_special_tokens:
        if token_ids_1 is not None:
            raise ValueError(
                "You should not supply a second sequence if the provided sequence of "
                "ids is already formatted with special tokens for the model."
            )
        return list(map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id] else 0, token_ids_0))

    if token_ids_1 is not None:
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    return [1] + ([0] * len(token_ids_0)) + [1]

def _collate_batch(examples, data_tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if data_tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({data_tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], data_tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if data_tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


class DataCollatorForLanguageModelingNgram:
    def __init__(self, data_tokenizer, mlm=True, mlm_probability=0.15):
        self.tokenizer = data_tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        2-gram mask
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 是否变成2-gram。 0.4
        ngram_matrix = torch.full(labels.shape, 0.6)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            # special_tokens_mask = [
            #     get_special_tokens_mask(self.tokenizer, val, already_has_special_tokens=True) for val in labels.tolist()
            # ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_backup = masked_indices.clone()
        ngram_indices = torch.bernoulli(ngram_matrix).bool()

        # 判断是否把1-gram 变成2-gram
        # todo：考虑采用共现特征提取n-gram
        for i in range(masked_indices.shape[0]):
            for j in range(masked_indices.shape[1]):
                if masked_backup[i][j] == True:
                    # 需要mask的
                    if ngram_indices[i][j] == True:
                        # 非特殊字符，才能变成2-gram
                        if j + 1 < masked_indices.shape[1] and not special_tokens_mask[i][j + 1]:
                            masked_indices[i][j + 1] = True

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
