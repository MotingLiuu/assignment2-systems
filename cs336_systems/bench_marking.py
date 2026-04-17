import timeit
import torch
from torch import nn as nn
from collections import Counter, defaultdict
from typing import Iterable, Iterator
from pydantic import BaseModel, Field, field_validator
import cs336_basics
import pandas as pd
import os

import logging

logger = logging.getLogger(__name__)


# Use Pydantic BaseModel to define the model configuration
class ModelConfig(BaseModel):
    vocab_size: int = Field(gt=0, description="Size of the vocabulary")
    context_length: int = Field(gt=0, description="Length of the context")
    d_model: int = Field(gt=0, description="Dimension of the model")
    num_layers: int = Field(gt=0, description="Number of layers in the model")
    num_heads: int = Field(gt=0, description="Number of attention heads")
    d_ff: int = Field(gt=0, description="Dimension of the feedforward network")
    rope_theta: float = Field(default=10000, description="Use RoPE positional encoding")

# Create a model to benchmark
def init_model(config: ModelConfig) -> nn.Module:
    # model_dump is a method provided by Pydantic BaseModel that returns the model's data as a dictionary
    return cs336_basics.model.BasicsTransformerLM(**config.model_dump()) 

# Generate random input data for benchmarking
def random_batch(config: ModelConfig, batch_size: int) -> torch.Tensor:
    return torch.randint(0, config.vocab_size, (batch_size, config.context_length))


def benchmark_model(
    model: nn.Module, data: torch.Tensor, back: bool = False, num_warmup: int = 0, num_execution: int = 0
) -> dict:
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

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = timeit.default_timer()
        exec_times.append(end_time - sta_time)
    logger.info(f"Done Execution times: {exec_times}")
    assert len(warmup_times) == num_warmup, f"Expected {num_warmup} warmup times, got {len(warmup_times)}"
    assert len(exec_times) == num_execution, f"Expected {num_execution} execution times, got {len(exec_times)}"

    avg_warmup = sum(warmup_times) / len(warmup_times) if warmup_times else 0
    std_warmup = float(torch.std(torch.tensor(warmup_times))) if warmup_times else 0
    avg_exec = sum(exec_times) / len(exec_times) if exec_times else 0
    std_exec = float(torch.std(torch.tensor(exec_times[1:]))) if len(exec_times) > 1 else 0

    logger.info(f"Avg Warmup time: {avg_warmup}, the standard deviation: {std_warmup}\n")
    logger.info(f"Avg Execution time: {avg_exec}, Back: {back}\n")
    if len(exec_times) > 1:
        logger.info(f"The standard deviation of execution times (excluding the first run): {std_exec}\n")

    return {
        "back": back,
        "num_warmup": num_warmup,
        "num_execution": num_execution,
        "avg_warmup_time": avg_warmup,
        "std_warmup_time": std_warmup,
        "avg_exec_time": avg_exec,
        "std_exec_time": std_exec,
    }


def save_benchmark_results(
    results: list[dict],
    output_dir: str = "result",
    filename: str = "benchmark_results.md",
) -> pd.DataFrame:
    """Convert benchmark results to DataFrame and save as markdown table."""
    processed = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "config"}
        if "config" in r:
            row.update(r["config"])
        processed.append(row)
    df = pd.DataFrame(processed)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"Saved benchmark results to {output_path}")
    return df
