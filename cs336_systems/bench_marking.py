import timeit
import torch
from torch import nn as nn
from collections import Counter, defaultdict
from typing import Iterable, Iterator
from pydantic import BaseModel, Field, field_validator
import cs336_basics

import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    vocab_size: int = Field(gt=0, description="Size of the vocabulary")
    context_length: int = Field(gt=0, description="Length of the context")
    d_model: int = Field(gt=0, description="Dimension of the model")
    num_layers: int = Field(gt=0, description="Number of layers in the model")
    num_heads: int = Field(gt=0, description="Number of attention heads")
    d_ff: int = Field(gt=0, description="Dimension of the feedforward network")
    rope_theta: bool = Field(default=False, description="Use RoPE positional encoding")


def init_model(config: ModelConfig) -> nn.Module:
    return cs336_basics.model.BasicsTransformerLM(**config.model_dump())


def random_batch(config: ModelConfig, batch_size: int) -> torch.Tensor:
    return torch.randint(0, config.vocab_size, (batch_size, config.context_length))


def benchmark_model(
    model: nn.Module, data: torch.Tensor, back: bool = False, num_warmup: int = 0, num_execution: int = 0
) -> float:
    batch_tensor, targets = data[:, :-1], data[:, 1:]
    warmup_times, exec_times = [], []

    for _ in range(num_warmup):
        sta_time = timeit.default_timer()
        model.forward(batch_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = timeit.default_timer()
        warmup_times.append(end_time - sta_time)
    logger.info(f"Done Warmup times: {warmup_times}")

    for _ in range(num_execution):
        sta_time = timeit.default_timer()
        logits = model.forward(batch_tensor)

        if back:
            loss = cs336_basics.nn_utils.cross_entropy(logits, targets)
            loss.backward()

        end_time = timeit.default_timer()
        exec_times.append(end_time - sta_time)
    logger.info(f"Done Execution times: {exec_times}")

    logger.info(
        f"Avg Warmup time: {sum(warmup_times) / len(warmup_times) if warmup_times else 0}, the standard deviation: {torch.std(torch.tensor(warmup_times)) if warmup_times else 0}"
    )
    logger.info(
        f"Avg Execution time: {sum(exec_times) / len(exec_times) if exec_times else 0}, Back: {back}, the standard deviation: {torch.std(torch.tensor(exec_times)) if exec_times else 0}"
    )

    return sum(exec_times) / len(exec_times) if exec_times else 0
