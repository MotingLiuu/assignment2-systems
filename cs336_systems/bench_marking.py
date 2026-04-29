from contextlib import nullcontext
import timeit
import torch
from torch import nn as nn
from collections import Counter, defaultdict
from typing import Iterable, Iterator
from pydantic import BaseModel, Field, field_validator
import cs336_basics
import pandas as pd
import os
from torch.profiler import profile, record_function, ProfilerActivity

import nvtx
domain = nvtx.get_domain("bench_marking")

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

# Initialize the optimizer for benchmarking
def init_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return cs336_basics.optimizer.AdamW(model.parameters(), lr=lr)


def benchmark_model(
    model: nn.Module, data: torch.Tensor, back: bool = False, optim: torch.optim.Optimizer | None = None, num_warmup: int = 0, num_execution: int = 0, mixed_precision: bool = False, memo_profile: str | None = None, profiler_result: str | None = None
) -> dict:
    batch_tensor, targets = data[:, :-1], data[:, 1:]
    warmup_times, exec_times = [], []
    
    grad_cm = nullcontext() if back else torch.no_grad()
    amp_cm = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()

    for i in range(num_warmup):
        sta_time = timeit.default_timer()

        with nvtx.annotate(f"warmup_step{i}", color="blue"):
            with grad_cm, amp_cm:
                logits = model(batch_tensor)
                if back:
                    loss = cs336_basics.nn_utils.cross_entropy(logits, targets)
            
            if back:
                loss.backward()
                if optim is not None:
                    optim.step()
                    optim.zero_grad(set_to_none=True) 
            torch.cuda.synchronize() if torch.cuda.is_available() else None

        end_time = timeit.default_timer()
        warmup_times.append(end_time - sta_time)
    logger.info(f"Done Warmup times: {warmup_times}")

    if memo_profile and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    profiler_cm = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_execution, repeat=1),
        record_shapes=True, profile_memory=True, with_stack=True, with_flops=True,
    ) if profiler_result and torch.cuda.is_available() else nullcontext()
    
    with profiler_cm as prof:
        for i in range(num_execution):
            sta_time = timeit.default_timer()
            
            with record_function("## forward ##"):
                with nvtx.annotate(f"execution_step{i}_forward", color="green"):
                    with grad_cm, amp_cm:  
                        logits = model(batch_tensor)
                        if back:
                            loss = cs336_basics.nn_utils.cross_entropy(logits, targets)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
        
            if back:
                with record_function("## backward ##"):
                    with nvtx.annotate(f"execution_step{i}_backward", color="orange"):
                        loss.backward() 
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        
                if optim is not None:
                    with record_function("## optimization ##"):
                        with nvtx.annotate(f"execution_step{i}_optim", color="red"):
                            optim.step()
                            optim.zero_grad(set_to_none=True)
                            torch.cuda.synchronize() if torch.cuda.is_available() else None
                
            end_time = timeit.default_timer()
            exec_times.append(end_time - sta_time)
            if prof is not None:
                prof.step()

    if prof is not None and profiler_result is not None:
        prof.export_memory_timeline(f"{profiler_result}.html")
        logger.info(f"Saved profiler trace to {profiler_result}")

    if memo_profile and torch.cuda.is_available():
        torch.cuda.memory._dump_snapshot(memo_profile)
        torch.cuda.memory._record_memory_history(enabled=None)

    logger.info(f"Done Execution times: {exec_times}")
    assert len(warmup_times) == num_warmup
    assert len(exec_times) == num_execution

    avg_warmup = sum(warmup_times) / len(warmup_times) if warmup_times else 0
    std_warmup = float(torch.std(torch.tensor(warmup_times))) if len(warmup_times) > 1 else 0
    avg_exec = sum(exec_times) / len(exec_times) if exec_times else 0
    std_exec = float(torch.std(torch.tensor(exec_times))) if len(exec_times) > 1 else 0 

    logger.info(f"Avg Warmup time: {avg_warmup}, the standard deviation: {std_warmup}\n")
    logger.info(f"Avg Execution time: {avg_exec}, Back: {back}\n")

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
    output_path: str = "result/benchmark_results.md",
) -> pd.DataFrame:
    """Convert benchmark results to DataFrame and save as markdown table."""
    processed = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "config"}
        if "config" in r:
            row.update(r["config"])
        processed.append(row)
    df = pd.DataFrame(processed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"Saved benchmark results to {output_path}")
    return df
