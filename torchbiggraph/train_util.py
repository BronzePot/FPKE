import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


from torch.optim import Optimizer
from torchbiggraph.batching import AbstractBatchProcessor

from torchbiggraph.checkpoint_manager import (
    MetadataProvider,
)
from torchbiggraph.config import ConfigSchema
from torchbiggraph.edgelist import EdgeList


from torchbiggraph.graph_storages import EDGE_STORAGES, ENTITY_STORAGES
from torchbiggraph.losses import LOSS_FUNCTIONS, AbstractLossFunction
from torchbiggraph.model import MultiRelationEmbedder, make_model
from torchbiggraph.stats import Stats
from torchbiggraph.types import (
    UNPARTITIONED,
    EntityName,
    Partition,
)

#Train
class IterationManager(MetadataProvider):
    def __init__(
        self,
        num_epochs: int,
        edge_paths: List[str],
        num_edge_chunks: int,
        *,
        iteration_idx: int = 0,
    ) -> None:
        self.num_epochs = num_epochs
        self.edge_paths = edge_paths
        self.num_edge_chunks = num_edge_chunks
        self.iteration_idx = iteration_idx

    @property
    def epoch_idx(self) -> int:
        return self.iteration_idx // self.num_edge_chunks // self.num_edge_paths

    @property
    def num_edge_paths(self) -> int:
        return len(self.edge_paths)

    @property
    def edge_path_idx(self) -> int:
        return self.iteration_idx // self.num_edge_chunks % self.num_edge_paths

    @property
    def edge_path(self) -> str:
        return self.edge_paths[self.edge_path_idx]

    @property
    def edge_chunk_idx(self) -> int:
        return self.iteration_idx % self.num_edge_chunks

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        while self.epoch_idx < self.num_epochs:
            yield self.epoch_idx, self.edge_path_idx, self.edge_chunk_idx
            self.iteration_idx += 1

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "iteration/num_epochs": self.num_epochs,
            "iteration/epoch_idx": self.epoch_idx,
            "iteration/num_edge_paths": self.num_edge_paths,
            "iteration/edge_path_idx": self.edge_path_idx,
            "iteration/edge_path": self.edge_path,
            "iteration/num_edge_chunks": self.num_edge_chunks,
            "iteration/edge_chunk_idx": self.edge_chunk_idx,
        }

    def __add__(self, delta: int) -> "IterationManager":
        return IterationManager(
            self.num_epochs,
            self.edge_paths,
            self.num_edge_chunks,
            iteration_idx=self.iteration_idx + delta,
        )

class Trainer(AbstractBatchProcessor):
    def __init__(
        self,
        model_optimizer: Optimizer,
        loss_fn: AbstractLossFunction,
        relation_weights: List[float],
    ) -> None:
        super().__init__(loss_fn, relation_weights)
        self.model_optimizer = model_optimizer
        self.unpartitioned_optimizers: Dict[EntityName, Optimizer] = {}
        self.partitioned_optimizers: Dict[Tuple[EntityName, Partition], Optimizer] = {}

    def _process_one_batch(
        self, model: MultiRelationEmbedder, batch_edges: EdgeList
    ) -> Stats:
        model.zero_grad()

        scores, reg = model(batch_edges)

        loss = self.calc_loss(scores, batch_edges)

        stats = Stats(
            loss=float(loss),
            reg=float(reg) if reg is not None else 0.0,
            violators_lhs=int((scores.lhs_neg > scores.lhs_pos.unsqueeze(1)).sum()),
            violators_rhs=int((scores.rhs_neg > scores.rhs_pos.unsqueeze(1)).sum()),
            count=len(batch_edges),
        )
        if reg is not None:
            (loss + reg).backward()
        else:
            loss.backward()
        self.model_optimizer.step(closure=None)
        for optimizer in self.unpartitioned_optimizers.values():
            optimizer.step(closure=None)
        for optimizer in self.partitioned_optimizers.values():
            optimizer.step(closure=None)

        return stats

def get_num_edge_chunks(config: ConfigSchema) -> int:
    if config.num_edge_chunks is not None:
        return config.num_edge_chunks

    max_edges_per_bucket = 0
    # We should check all edge paths, all lhs partitions and all rhs partitions,
    # but the combinatorial explosion could lead to thousands of checks. Let's
    # assume that edges are uniformly distributed among buckets (this is not
    # exactly the case, as it's the entities that are uniformly distributed
    # among the partitions, and edge assignments to buckets are a function of
    # that, thus, for example, very high degree entities could skew this), and
    # use the size of bucket (0, 0) as an estimate of the average bucket size.
    # We still do it for all edge paths as there could be semantic differences
    # between them which lead to different sizes.
    for edge_path in config.edge_paths:
        edge_storage = EDGE_STORAGES.make_instance(edge_path)
        max_edges_per_bucket = max(
            max_edges_per_bucket,
            edge_storage.get_number_of_edges(UNPARTITIONED, UNPARTITIONED),
        )
    return max(1, math.ceil(max_edges_per_bucket / config.max_edges_per_chunk))

