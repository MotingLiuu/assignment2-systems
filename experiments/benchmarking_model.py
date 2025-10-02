import torch
from cs336_systems import bench_marking as benchmark
from typing import Iterable, Iterator
import logging
import pandas as pd
import argparse


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
    return parser.parse_args()


class benchmarking:
    def __init__(self, configs: Iterable[benchmark.ModelConfig]):
        self.configs = list(configs)

    def add_config(self, config: benchmark.ModelConfig) -> None:
        self.configs.append(config)

    def run_benchmark(self, back: bool = False, num_warmup: int = 0, num_execution: int = 0, batch_size=1) -> None:
        for config in self.configs:
            model = benchmark.init_model(config)
            model.to(device)
            logger.info(f"Initialized model with config: {config}")
            random_data = benchmark.random_batch(config, batch_size)
            random_data.to(device)
            benchmark.benchmark_model(model, random_data, back, num_warmup, num_execution)


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

    baseline_benchmark = benchmarking([small_config])
    baseline_benchmark.run_benchmark(
        back=args.back, num_warmup=args.num_warmup, num_execution=args.num_execution, batch_size=args.batch_size
    )
