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


def benchmark_model(model: nn.Module, data: torch.Tensor, num_warmup: int = 0, num_execution: int = 0) -> float:
    for _ in range(num_warmup):
        model.forward(data)

    exec_time = 0
    for _ in range(num_execution):
        exec_time += timeit.timeit(lambda: model.forward(data), setup=lambda: torch.cuda.synchronize(), number=1)
    logger.info(f"Execution time over {num_execution} runs: {exec_time:.6f} seconds")
    logger.info(f"Average time per run: {exec_time / num_execution:.6f} seconds")

    return exec_time / num_execution
