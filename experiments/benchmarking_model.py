import gc

import torch
from cs336_systems import bench_marking as benchmark
from typing import Iterable, Iterator
import logging
import pandas as pd
import argparse

import nvtx
domain = nvtx.get_domain("benchmarking_model")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

logger.info(f"Using device: {device}")


def get_args():
    parser = argparse.ArgumentParser(description="Benchmarking Transformer Models")
    parser.add_argument("--back", action="store_true", help="Enable backward pass during benchmarking")
    parser.add_argument("--num_warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--num_execution", type=int, default=10, help="Number of execution iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for input data")
    parser.add_argument("--output", type=str, default="result/benchmark_results.md", help="Output path for benchmark results (e.g., result/benchmark_results.md)")
    parser.add_argument("--optim", action="store_true", help="Enable optimization step during benchmarking")
    return parser.parse_args()


class benchmarking:
    def __init__(self, configs: Iterable[benchmark.ModelConfig]):
        self.configs = list(configs)
        self.results = []

    def add_config(self, config: benchmark.ModelConfig) -> None:
        self.configs.append(config)

    def run_benchmark(self, back: bool = False, num_warmup: int = 0, num_execution: int = 0, batch_size=1, optim: bool = False) -> None:
        for config in self.configs:
            
            # Use NVTX to annotate the model initialization for better profiling visualization
            with nvtx.annotate(f"model_init_{config.d_model}d_{config.num_layers}l", color="purple"):
                model = benchmark.init_model(config)
                model.bfloat16().to(device)
                logger.info(f"Initialized model with config: {config}")
                torch.cuda.synchronize() if torch.cuda.is_available() else None
    
            random_data = benchmark.random_batch(config, batch_size)
            random_data = random_data.to(device)
            
            # Initialize optimizer if optim flag is set
            optimizer = None
            if optim:
                optimizer = benchmark.init_optimizer(model, lr=1e-3)

            result = benchmark.benchmark_model(model, random_data, back, optim=optimizer if optim else None, num_warmup=num_warmup, num_execution=num_execution)
            result["config"] = config.model_dump()
            self.results.append(result)
            
            del model
            del random_data

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def save_results(self, output_path: str = "result/benchmark_results.md") -> None:
        benchmark.save_benchmark_results(self.results, output_path)


if __name__ == "__main__":
    args = get_args()

    small_config = benchmark.ModelConfig(
        vocab_size=10000, context_length=1024, d_model=768, num_layers=12, num_heads=12, d_ff=3072
    )
    medium_config = benchmark.ModelConfig(
        vocab_size=10000, context_length=1024, d_model=1024, num_layers=24, num_heads=16, d_ff=4096
    )
    large_config = benchmark.ModelConfig(
        vocab_size=10000, context_length=1024, d_model=1280, num_layers=32, num_heads=20, d_ff=5120
    )
    xlarge_config = benchmark.ModelConfig(
        vocab_size=10000, context_length=1024, d_model=1600, num_layers=48, num_heads=25, d_ff=6400
    )
    b27_config = benchmark.ModelConfig(
        vocab_size=10000, context_length=1024, d_model=2560, num_layers=32, num_heads=32, d_ff=10240
    )

    all_configs = [small_config, medium_config, large_config, xlarge_config, b27_config]
    baseline_benchmark = benchmarking(all_configs)
    baseline_benchmark.run_benchmark(
        back=args.back, num_warmup=args.num_warmup, num_execution=args.num_execution, batch_size=args.batch_size, optim=args.optim
    )
    baseline_benchmark.save_results(output_path=args.output)
