from dataclasses import dataclass, field
from typing import Any

import torch
from torch_geometric.data import Data


@dataclass
class PostProcessInput:
    mesh: Data
    labels: torch.Tensor
    logits: torch.Tensor | None
    stage: str
    label_kind: str
    class_names: list[str] | None = None


@dataclass
class PostProcessOutput:
    labels: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePostProcessor:
    def __init__(
        self,
        enabled: bool = True,
        run_on_validation: bool = True,
        run_on_test: bool = True,
    ):
        self.enabled = bool(enabled)
        self.run_on_validation = bool(run_on_validation)
        self.run_on_test = bool(run_on_test)

    def should_run(self, stage: str) -> bool:
        if not self.enabled:
            return False
        if stage == "val":
            return self.run_on_validation
        if stage == "test":
            return self.run_on_test
        return False

    def __call__(self, data: PostProcessInput) -> PostProcessOutput:
        if not self.should_run(data.stage):
            return PostProcessOutput(labels=data.labels)
        return self.process(data)

    def process(self, data: PostProcessInput) -> PostProcessOutput:
        raise NotImplementedError


class NoOpPostProcessor(BasePostProcessor):
    def process(self, data: PostProcessInput) -> PostProcessOutput:
        return PostProcessOutput(labels=data.labels)
