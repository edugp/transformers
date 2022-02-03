# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for Data2Vec
"""

from math import trunc
from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...file_utils import PaddingStrategy, TensorType
from ...utils import logging
from ...tokenization_utils import PreTrainedTokenizer
from ...models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor


logger = logging.get_logger(__name__)


def is_batched(raw_data: Union[np.ndarray, List[str], List[np.ndarray], List[List[str]]]) -> bool:
    return bool(
        isinstance(raw_data, (list, tuple))
        and (isinstance(raw_data[0], np.ndarray) or isinstance(raw_data[0], (tuple, list)))
    )


class Data2VecTextFeatureExtractor:
    """
    Constructs a Data2Vec text feature extractor
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        raw_text: Union[np.ndarray, List[str], List[np.ndarray], List[List[str]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        # always return batch
        if not is_batched(raw_text):
            raw_text = [raw_text]

        # add inputs to BatchFeature
        encoded_outputs = BatchFeature({"input_values": raw_text})
        encoded_outputs.update(
            self.tokenizer(
                raw_text, padding=padding, return_tensors=return_tensors, max_length=max_length, truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, **kwargs))
        return encoded_outputs
    


class Data2VecAudioFeatureExtractor(Wav2Vec2FeatureExtractor):
    """
    Constructs a Data2Vec audio feature extractor
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, return_attention_mask=return_attention_mask, do_normalize=do_normalize,  **kwargs)
