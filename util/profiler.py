"""Profiling utilities for identifying bottlenecks in training."""

import time
import torch
from contextlib import contextmanager
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TimingStats:
    """Track timing statistics for named operations."""

    def __init__(self):
        self.stats: Dict[str, dict] = {}
        self.current_timers: Dict[str, float] = {}

    def start(self, name: str):
        """Start timing an operation."""
        self.current_timers[name] = time.perf_counter()

    def stop(self, name: str):
        """Stop timing an operation and record the duration."""
        if name not in self.current_timers:
            logger.warning(f"Timer '{name}' was never started")
            return

        elapsed = time.perf_counter() - self.current_timers[name]
        del self.current_timers[name]

        if name not in self.stats:
            self.stats[name] = {
                "count": 0,
                "total_time": 0.0,
                "max_time": 0.0,
                "min_time": float("inf"),
            }

        self.stats[name]["count"] += 1
        self.stats[name]["total_time"] += elapsed
        self.stats[name]["max_time"] = max(self.stats[name]["max_time"], elapsed)
        self.stats[name]["min_time"] = min(self.stats[name]["min_time"], elapsed)

    @contextmanager
    def time(self, name: str):
        """Context manager for timing a block of code."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get_report(self, sort_by: str = "total") -> str:
        """Generate a formatted timing report.

        Args:
            sort_by: Sort by 'total', 'avg', 'max', 'count'
        """
        if not self.stats:
            return "No timing data collected."

        report = "\n" + "=" * 100 + "\n"
        report += f"{'Operation':<40} {'Calls':>8} {'Total(s)':>12} {'Avg(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}\n"
        report += "-" * 100 + "\n"

        # Sort stats
        sort_key_map = {
            "total": lambda x: x[1]["total_time"],
            "avg": lambda x: x[1]["total_time"] / x[1]["count"],
            "max": lambda x: x[1]["max_time"],
            "count": lambda x: x[1]["count"],
        }
        sorted_stats = sorted(
            self.stats.items(),
            key=sort_key_map.get(sort_by, sort_key_map["total"]),
            reverse=True,
        )

        for name, stats in sorted_stats:
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            report += (
                f"{name:<40} "
                f"{stats['count']:>8} "
                f"{stats['total_time']:>12.4f} "
                f"{avg_time*1000:>12.2f} "
                f"{stats['min_time']*1000:>12.2f} "
                f"{stats['max_time']*1000:>12.2f}\n"
            )

        total_time = sum(s["total_time"] for s in self.stats.values())
        report += "-" * 100 + "\n"
        report += f"{'TOTAL':<40} {'':<8} {total_time:>12.4f}\n"
        report += "=" * 100 + "\n"

        return report

    def get_summary(self) -> dict[str, object]:
        """Return timing stats in a JSON-serializable format."""
        if not self.stats:
            return {"stats": {}, "total_time": 0.0}

        total_time = sum(s["total_time"] for s in self.stats.values())
        summary = {}
        for name, stats in self.stats.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0.0
            percentage = (
                (stats["total_time"] / total_time * 100.0) if total_time > 0 else 0.0
            )
            summary[name] = {
                "count": stats["count"],
                "total_time": stats["total_time"],
                "avg_time_ms": avg_time * 1000.0,
                "min_time_ms": stats["min_time"] * 1000.0,
                "max_time_ms": stats["max_time"] * 1000.0,
                "percentage": percentage,
            }

        return {"stats": summary, "total_time": total_time}

    def reset(self):
        """Reset all timing statistics."""
        self.stats = {}
        self.current_timers = {}

    def get_percentage_report(self) -> str:
        """Generate a report showing percentage of total time."""
        if not self.stats:
            return "No timing data collected."

        total_time = sum(s["total_time"] for s in self.stats.values())

        report = "\n" + "=" * 80 + "\n"
        report += f"{'Operation':<40} {'Time(s)':>12} {'Percentage':>12}\n"
        report += "-" * 80 + "\n"

        sorted_stats = sorted(
            self.stats.items(), key=lambda x: x[1]["total_time"], reverse=True
        )

        for name, stats in sorted_stats:
            percentage = (
                (stats["total_time"] / total_time * 100) if total_time > 0 else 0
            )
            report += f"{name:<40} {stats['total_time']:>12.4f} {percentage:>11.1f}%\n"

        report += "-" * 80 + "\n"
        report += f"{'TOTAL':<40} {total_time:>12.4f} {100.0:>11.1f}%\n"
        report += "=" * 80 + "\n"

        return report


class GPUTimingStats(TimingStats):
    """Timing stats that synchronize with CUDA for accurate GPU timing."""

    def start(self, name: str):
        """Start timing with GPU synchronization."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        super().start(name)

    def stop(self, name: str):
        """Stop timing with GPU synchronization."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        super().stop(name)


def profile_dataloader(dataloader, num_batches: int = 5) -> TimingStats:
    """Profile data loading time.

    Args:
        dataloader: PyTorch DataLoader to profile
        num_batches: Number of batches to profile

    Returns:
        TimingStats object with profiling results
    """
    stats = TimingStats()

    print(f"Profiling dataloader for {num_batches} batches...")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        with stats.time("dataloader_iteration"):
            # Just loading the batch
            pass

        # Also measure data transfer to GPU
        if torch.cuda.is_available():
            with stats.time("data_to_gpu"):
                batch = batch.to("cuda")

    print(stats.get_report())
    return stats


def profile_training_step(model, dataloader, num_batches: int = 5) -> TimingStats:
    """Profile a complete training step breakdown.

    Args:
        model: PyTorch Lightning model
        dataloader: DataLoader
        num_batches: Number of batches to profile

    Returns:
        TimingStats with detailed breakdown
    """
    stats = GPUTimingStats()
    device = next(model.parameters()).device

    print(f"Profiling training step for {num_batches} batches...")
    print(f"Model device: {device}")

    model.eval()  # Disable dropout etc for consistent timing

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Data loading already happened, measure transfer
            with stats.time("1_data_to_device"):
                batch = batch.to(device)

            # Forward pass
            with stats.time("2_forward_pass"):
                out = model(batch)

            # Loss computation
            with stats.time("3_loss_computation"):
                if hasattr(model, "compute_loss"):
                    out_dict = {"all": out}
                    losses = model.compute_loss(out_dict, batch.edge_labels)

            # Metrics computation
            with stats.time("4_metrics_computation"):
                if hasattr(model, "evaluate"):
                    results = model.evaluate(out, batch, stage="val")

            print(f"Completed batch {i+1}/{num_batches}")

    print(stats.get_report())
    print("\nPercentage breakdown:")
    print(stats.get_percentage_report())

    return stats
