import ctypes
import ctypes.util
import logging
import os
import time
from collections import defaultdict
from multiprocessing.connection import wait as mp_wait
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.multiprocessing as mp
from torchbiggraph.batching import AbstractBatchProcessor, process_in_batches
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.graph_storages import EDGE_STORAGES, ENTITY_STORAGES
from torchbiggraph.model import MultiRelationEmbedder
from torchbiggraph.parameter_sharing import ParameterServer, ParameterSharer
from torchbiggraph.row_adagrad import RowAdagrad
from torchbiggraph.stats import Stats, StatsHandler
from torchbiggraph.train_util import Trainer
from torchbiggraph.types import (
    EntityName,
    FloatTensorType,
    GPURank,
    LongTensorType,
    Partition,
    Rank,
    Side,
    SubPartition,
)
from torchbiggraph.util import (
    add_logfile
)


try:
    from torchbiggraph import _C

    CPP_INSTALLED = True
except ImportError:
    CPP_INSTALLED = False


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


class TimeKeeper:
    def __init__(self):
        self.t = self._get_time()
        self.sub_ts = {}

    def _get_time(self) -> float:
        return time.monotonic()

    def start(self, name: str) -> None:
        self.sub_ts[name] = self._get_time()

    def stop(self, name: str) -> float:
        start_t = self.sub_ts.pop(name)
        stop_t = self._get_time()
        delta_t = stop_t - start_t
        self.t += delta_t
        return delta_t

    def unaccounted(self) -> float:
        t = self._get_time()
        return t - self.t

GPU_STATE_TRAIN=0
GPU_STATE_EVAL=1

class SubprocessArgs(NamedTuple):
    state:int
    lhs_types: Set[str]
    rhs_types: Set[str]
    lhs_part: Partition
    rhs_part: Partition
    lhs_subpart: SubPartition
    rhs_subpart: SubPartition
    next_lhs_subpart: Optional[SubPartition]
    next_rhs_subpart: Optional[SubPartition]
    model: MultiRelationEmbedder
    trainer: Trainer
    all_embs: Dict[Tuple[EntityName, Partition], FloatTensorType]
    subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice]
    subbuckets: Dict[
        Tuple[int, int], Tuple[LongTensorType, LongTensorType, LongTensorType]
    ]
    batch_size: int
    lr: float


class SubprocessReturn(NamedTuple):
    gpu_idx: GPURank
    stats: Stats


class GPUProcess(mp.get_context("spawn").Process):
    def __init__(
        self,
        gpu_idx: GPURank,
        subprocess_init: Optional[Callable[[], None]] = None,
        embedding_storage_freelist: Optional[Set[torch.FloatStorage]] = None,
        log_path=None
    ) -> None:
        super().__init__(daemon=True, name=f"GPU #{gpu_idx}")
        self.gpu_idx = gpu_idx
        self.master_endpoint, self.worker_endpoint = mp.get_context("spawn").Pipe()
        self.subprocess_init = subprocess_init
        self.sub_holder: Dict[
            Tuple[EntityName, Partition, SubPartition],
            Tuple[torch.nn.Parameter, RowAdagrad],
        ] = {}
        self.embedding_storage_freelist = embedding_storage_freelist
        self.log_path=log_path

    @property
    def my_device(self) -> torch.device:
        return torch.device("cuda", index=self.gpu_idx)

    def run(self) -> None:
        torch.set_num_threads(1)
        torch.cuda.set_device(self.my_device)
        add_logfile(logger,self.log_path,f"{self.gpu_idx}")
        if self.subprocess_init is not None:
            self.subprocess_init()
        self.master_endpoint.close()
        for s in self.embedding_storage_freelist:
            assert s.is_shared()
            cptr = ctypes.c_void_p(s.data_ptr())
            csize = ctypes.c_size_t(s.size() * s.element_size())
            cflags = ctypes.c_uint(0)
            # FIXME: broken by D20249187
            #cudart = torch.cuda.cudart()
            cudart = ctypes.cdll.LoadLibrary(ctypes.util.find_library("cudart"))
            #print(s.size())
            #print(csize)
            res = cudart.cudaHostRegister(cptr, csize, cflags)
            if res==1:
                print(s.size(),s.element_size(),csize)
            torch.cuda.check_error(res)
            assert s.is_pinned()
        logger.info(f"GPU subprocess {self.gpu_idx} up and running")
        while True:
            try:
                job: SubprocessArgs = self.worker_endpoint.recv()
            except EOFError:
                break
            if job.state==GPU_STATE_TRAIN:
                stats = self.do_one_job(
                    lhs_types=job.lhs_types,
                    rhs_types=job.rhs_types,
                    lhs_part=job.lhs_part,
                    rhs_part=job.rhs_part,
                    lhs_subpart=job.lhs_subpart,
                    rhs_subpart=job.rhs_subpart,
                    next_lhs_subpart=job.next_lhs_subpart,
                    next_rhs_subpart=job.next_rhs_subpart,
                    model=job.model,
                    trainer=job.trainer,
                    all_embs=job.all_embs,
                    subpart_slices=job.subpart_slices,
                    subbuckets=job.subbuckets,
                    batch_size=job.batch_size,
                    lr=job.lr,
                )
            else:
                stats = self.do_one_eval_job(
                    lhs_types=job.lhs_types,
                    rhs_types=job.rhs_types,
                    lhs_part=job.lhs_part,
                    rhs_part=job.rhs_part,
                    lhs_subpart=job.lhs_subpart,
                    rhs_subpart=job.rhs_subpart,
                    next_lhs_subpart=job.next_lhs_subpart,
                    next_rhs_subpart=job.next_rhs_subpart,
                    model=job.model,
                    evaluator=job.trainer,
                    all_embs=job.all_embs,
                    subpart_slices=job.subpart_slices,
                    subbuckets=job.subbuckets,
                    batch_size=job.batch_size,
                    lr=job.lr,
                )


            self.worker_endpoint.send(
                SubprocessReturn(gpu_idx=self.gpu_idx, stats=stats)
            )

    def do_one_job(  # noqa
        self,
        lhs_types: Set[str],
        rhs_types: Set[str],
        lhs_part: Partition,
        rhs_part: Partition,
        lhs_subpart: SubPartition,
        rhs_subpart: SubPartition,
        next_lhs_subpart: Optional[SubPartition],
        next_rhs_subpart: Optional[SubPartition],
        model: MultiRelationEmbedder,
        trainer: Trainer,
        all_embs: Dict[Tuple[EntityName, Partition], FloatTensorType],
        subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice],
        subbuckets: Dict[
            Tuple[int, int], Tuple[LongTensorType, LongTensorType, LongTensorType]
        ],
        batch_size: int,
        lr: float,
    ) -> Stats:
        tk = TimeKeeper()

        for embeddings in all_embs.values():
            assert embeddings.is_pinned()

        occurrences: Dict[
            Tuple[EntityName, Partition, SubPartition], Set[Side]
        ] = defaultdict(set)
        for entity_name in lhs_types:
            occurrences[entity_name, lhs_part, lhs_subpart].add(Side.LHS)
        for entity_name in rhs_types:
            occurrences[entity_name, rhs_part, rhs_subpart].add(Side.RHS)

        if lhs_part != rhs_part:  # Bipartite
            assert all(len(v) == 1 for v in occurrences.values())
        ##############################################################################
        """ if self.gpu_idx==0:
            print("\nnew job begin")
            print(self.sub_holder.keys())
            allocated_memory=torch.cuda.memory_allocated(self.my_device)/1024/1024/1024
            print(allocated_memory)
            print("\n") """
        """ ####################################################################
        gpu_memory=0
        for (
            (entity_name, part, subpart),
            (gpu_embeddings, gpu_optimizer),
        ) in self.sub_holder.items():
            (num_ent,dim)=gpu_embeddings.shape
            gpu_memory+=num_ent*dim*4/1024/1024/1024
        gpu_memory=gpu_memory*2
        num_edges = subbuckets[lhs_subpart, rhs_subpart][0].shape[0]
        gpu_memory+=num_edges*8*3/1024/1024/1024

        print(self.my_device,gpu_memory)
        allocated_memory=torch.cuda.memory_allocated(self.my_device)/1024/1024/1024
        print(allocated_memory)
        print(torch.cuda.memory_reserved(self.my_device)/1024/1024/1024)
        print(torch.cuda.memory_summary())
        if allocated_memory>10:
            print("self.sub_holder")
            raise
        ###################################################################### """
        ############################################################################
        tk.start("copy_to_device")
        for entity_name, part, subpart in occurrences.keys():
            if (entity_name, part, subpart) in self.sub_holder:
                continue
            embeddings = all_embs[entity_name, part]
            optimizer = trainer.partitioned_optimizers[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            # TODO have two permanent storages on GPU and move stuff in and out
            # from them
            # logger.info(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * embeddings.shape[1] * 4:,} bytes")
            gpu_embeddings = torch.empty(
                (subpart_slice.stop - subpart_slice.start, embeddings.shape[1]),
                dtype=torch.float32,
                device=self.my_device,
            )
            gpu_embeddings.copy_(embeddings[subpart_slice], non_blocking=True)
            gpu_embeddings = torch.nn.Parameter(gpu_embeddings)
            gpu_optimizer = RowAdagrad([gpu_embeddings], lr=lr)
            (cpu_state,) = optimizer.state.values()
            (gpu_state,) = gpu_optimizer.state.values()
            # logger.info(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * 4:,} bytes")
            gpu_state["sum"].copy_(cpu_state["sum"][subpart_slice], non_blocking=True)

            self.sub_holder[entity_name, part, subpart] = (
                gpu_embeddings,
                gpu_optimizer,
            )
        logger.debug(
            f"Time spent copying subparts to GPU: {tk.stop('copy_to_device'):.4f} s"
        )

        for (
            (entity_name, part, subpart),
            (gpu_embeddings, gpu_optimizer),
        ) in self.sub_holder.items():
            for side in occurrences[entity_name, part, subpart]:
                model.set_embeddings(entity_name, side, gpu_embeddings)
                trainer.partitioned_optimizers[
                    entity_name, part, subpart
                ] = gpu_optimizer

        tk.start("translate_edges")
        num_edges = subbuckets[lhs_subpart, rhs_subpart][0].shape[0]
        edge_perm = torch.randperm(num_edges)
        edges_lhs, edges_rhs, edges_rel = subbuckets[lhs_subpart, rhs_subpart]
        _C.shuffle(edges_lhs, edge_perm, os.cpu_count())
        _C.shuffle(edges_rhs, edge_perm, os.cpu_count())
        _C.shuffle(edges_rel, edge_perm, os.cpu_count())
        assert edges_lhs.is_pinned()
        assert edges_rhs.is_pinned()
        assert edges_rel.is_pinned()
        gpu_edges = EdgeList(
            EntityList.from_tensor(edges_lhs),
            EntityList.from_tensor(edges_rhs),
            edges_rel,
        ).to(self.my_device, non_blocking=True)
        logger.debug(f"GPU #{self.gpu_idx} got {num_edges} edges")
        logger.debug(
            f"Time spent copying edges to GPU: {tk.stop('translate_edges'):.4f} s"
        )

        tk.start("processing")
        stats = process_in_batches(
            batch_size=batch_size, model=model, batch_processor=trainer, edges=gpu_edges
        )
        logger.debug(f"Time spent processing: {tk.stop('processing'):.4f} s")

        next_occurrences: Dict[
            Tuple[EntityName, Partition, SubPartition], Set[Side]
        ] = defaultdict(set)
        if next_lhs_subpart is not None:
            for entity_name in lhs_types:
                next_occurrences[entity_name, lhs_part, next_lhs_subpart].add(Side.LHS)
        if next_rhs_subpart is not None:
            for entity_name in rhs_types:
                next_occurrences[entity_name, rhs_part, next_rhs_subpart].add(Side.RHS)
        """ ####################################################################################
        if self.gpu_idx==0:
            print("\nbefore release")
            print(self.sub_holder.keys())
            allocated_memory=torch.cuda.memory_allocated(self.my_device)/1024/1024/1024
            print(allocated_memory)
            print("\n")
        ################################################################################ """
        tk.start("copy_from_device")
        for (entity_name, part, subpart), (gpu_embeddings, gpu_optimizer) in list(
            self.sub_holder.items()
        ):
            if (entity_name, part, subpart) in next_occurrences:
                continue
            embeddings = all_embs[entity_name, part]
            optimizer = trainer.partitioned_optimizers[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            embeddings[subpart_slice].data.copy_(
                gpu_embeddings.detach(), non_blocking=True
            )
            del gpu_embeddings
            (cpu_state,) = optimizer.state.values()
            (gpu_state,) = gpu_optimizer.state.values()
            cpu_state["sum"][subpart_slice].copy_(gpu_state["sum"], non_blocking=True)
            del gpu_state["sum"]
            del self.sub_holder[entity_name, part, subpart]
        logger.debug(
            f"Time spent copying subparts from GPU: {tk.stop('copy_from_device'):.4f} s"
        )
        """ ###################################################################################
        if self.gpu_idx==0:
            print("\nafter release")
            print(self.sub_holder.keys())
            allocated_memory=torch.cuda.memory_allocated(self.my_device)/1024/1024/1024
            print(allocated_memory)
            print("\n")
        #################################################################################### """
        logger.debug(f"do_one_job: Time unaccounted for: {tk.unaccounted():.4f} s")

        return stats
    
    def do_one_eval_job(  # noqa
        self,
        lhs_types: Set[str],
        rhs_types: Set[str],
        lhs_part: Partition,
        rhs_part: Partition,
        lhs_subpart: SubPartition,
        rhs_subpart: SubPartition,
        next_lhs_subpart: Optional[SubPartition],
        next_rhs_subpart: Optional[SubPartition],
        model: MultiRelationEmbedder,
        evaluator: Trainer,
        all_embs: Dict[Tuple[EntityName, Partition], FloatTensorType],
        subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice],
        subbuckets: Dict[
            Tuple[int, int], Tuple[LongTensorType, LongTensorType, LongTensorType]
        ],
        batch_size: int,
        lr: float,
    ) -> Stats:
        tk = TimeKeeper()

        for embeddings in all_embs.values():
            assert embeddings.is_pinned()

        occurrences: Dict[
            Tuple[EntityName, Partition, SubPartition], Set[Side]
        ] = defaultdict(set)
        for entity_name in lhs_types:
            occurrences[entity_name, lhs_part, lhs_subpart].add(Side.LHS)
        for entity_name in rhs_types:
            occurrences[entity_name, rhs_part, rhs_subpart].add(Side.RHS)

        if lhs_part != rhs_part:  # Bipartite
            assert all(len(v) == 1 for v in occurrences.values())
        
        tk.start("copy_to_device")
        for entity_name, part, subpart in occurrences.keys():
            if (entity_name, part, subpart) in self.sub_holder:
                continue
            embeddings = all_embs[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            # TODO have two permanent storages on GPU and move stuff in and out
            # from them
            # logger.info(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * embeddings.shape[1] * 4:,} bytes")
            gpu_embeddings = torch.empty(
                (subpart_slice.stop - subpart_slice.start, embeddings.shape[1]),
                dtype=torch.float32,
                device=self.my_device,
            )
            gpu_embeddings.copy_(embeddings[subpart_slice], non_blocking=False)
            gpu_embeddings = torch.nn.Parameter(gpu_embeddings)
           
            self.sub_holder[entity_name, part, subpart] = (
                gpu_embeddings,None
            )
       

        for (
            (entity_name, part, subpart),
            (gpu_embeddings, gpu_optimizer),
        ) in self.sub_holder.items():
            for side in occurrences[entity_name, part, subpart]:
                model.set_embeddings(entity_name, side, gpu_embeddings)

        tk.start("translate_edges")
        num_edges = subbuckets[lhs_subpart, rhs_subpart][0].shape[0]
        edges_lhs, edges_rhs, edges_rel = subbuckets[lhs_subpart, rhs_subpart]
        assert edges_lhs.is_pinned()
        assert edges_rhs.is_pinned()
        assert edges_rel.is_pinned()
        gpu_edges = EdgeList(
            EntityList.from_tensor(edges_lhs),
            EntityList.from_tensor(edges_rhs),
            edges_rel,
        ).to(self.my_device, non_blocking=True)
        
        tk.start("processing")
        stats = process_in_batches(
            batch_size=batch_size, model=model, batch_processor=evaluator, edges=gpu_edges
        )

        next_occurrences: Dict[
            Tuple[EntityName, Partition, SubPartition], Set[Side]
        ] = defaultdict(set)
        if next_lhs_subpart is not None:
            for entity_name in lhs_types:
                next_occurrences[entity_name, lhs_part, next_lhs_subpart].add(Side.LHS)
        if next_rhs_subpart is not None:
            for entity_name in rhs_types:
                next_occurrences[entity_name, rhs_part, next_rhs_subpart].add(Side.RHS)
        
        tk.start("release GPU embeddings")
        for (entity_name, part, subpart), (gpu_embeddings, gpu_optimizer) in list(
            self.sub_holder.items()
        ):
            if (entity_name, part, subpart) in next_occurrences:
                continue
            self.sub_holder.pop((entity_name, part, subpart))

        return stats

class GPUProcessPool:
    def __init__(
        self,
        num_gpus: int,
        subprocess_init: Optional[Callable[[], None]] = None,
        embedding_storage_freelist: Optional[Set[torch.FloatStorage]] = None,
        log_path=None,
    ) -> None:
        self.processes: List[GPUProcess] = [
            GPUProcess(gpu_idx, subprocess_init, embedding_storage_freelist,log_path)
            for gpu_idx in range(num_gpus)
        ]
        for p in self.processes:
            p.start()
            p.worker_endpoint.close()

    @property
    def num_gpus(self):
        return len(self.processes)

    def schedule(self, gpu_idx: GPURank, args: SubprocessArgs) -> None:
        self.processes[gpu_idx].master_endpoint.send(args)

    def wait_for_next(self) -> Tuple[GPURank, SubprocessReturn]:
        all_objects = [p.sentinel for p in self.processes] + [
            p.master_endpoint for p in self.processes
        ]
        ready_objects = mp_wait(all_objects)
        for obj in ready_objects:
            for p in self.processes:
                if obj is p.sentinel:
                    raise RuntimeError(
                        f"GPU worker #{p.gpu_idx} (PID: {p.pid}) terminated "
                        f"unexpectedly with exit code {p.exitcode}"
                    )
                if obj is p.master_endpoint:
                    res = p.master_endpoint.recv()
                    return p.gpu_idx, res

    def close(self):
        pass

    def join(self):
        for p in self.processes:
            p.master_endpoint.close()
            p.join()


def build_nonbipartite_schedule(num_partition,num_worker) -> List[List[int]]:
    sch=[[] for i in range(num_worker)]
    for x in range(0,num_partition,num_worker*2):
        for y in range(0,num_partition,num_worker*2):
            a=[]
            for i in range(num_worker):
                hid=x+i
                tid=y+i
                sch[i].append((hid,tid))
            #print(a)
            for i in range(num_worker):
                hid=x+num_worker+i
                tid=y+num_worker+i
                sch[i].append((hid,tid))
            
            #print(sch)
            gz=1
            while(gz<=num_worker):
                for offset in range(gz):
                    for i in range(num_worker):
                        hid=int(x+(i//gz*2)*gz+i%gz)
                        tid=int(y+(i//gz*2+1)*gz+(i+offset)%gz)
                        sch[i].append((hid,tid))
                        sch[i].append((tid,hid))
                gz*=2
    return sch


def build_bipartite_schedule(num_partition,num_worker) -> List[List[int]]:
    sch=[[] for i in range(num_worker)]
    for x in range(0,num_partition,num_worker):
        for y in range(0,num_partition,num_worker):
            for offset in range(num_worker):
                for i in range(num_worker):
                    hid=x+(i+offset)%num_worker
                    tid=y+i
                    sch[i].append((hid,tid))
    return sch


def build_nonbipartite_schedule_inner_v2(size: int) -> List[List[int]]:
    if size <= 0 or size % 2 != 0:
        raise ValueError("Bad")
    if size == 2:
        return [[(0, 1), (1, 0)]]
    half = size // 2
    pre = [[(i, (i + j) % half + half) for j in range(half)] for i in range(half)]
    post = [[(i + half, (i + j) % half) for j in range(half)] for i in range(half)]
    mid = build_nonbipartite_schedule_inner_v2(half)
    res = []
    res.extend([pre[i] + mid[i] + post[i] for i in range(half // 2)])
    res.extend(
        [
            pre[i + half // 2]
            + [(x + half, y + half) for x, y in mid[i]]
            + post[i + half // 2]
            for i in range(half // 2)
        ]
    )
    return res


def build_nonbipartite_schedule_v2(size: int) -> List[List[int]]:
    if size == 1:
        return [[(0, 0)]]
    if size <= 0 or size % 2 != 0:
        raise ValueError("Bad")
    half = size // 2
    res = build_nonbipartite_schedule_inner_v2(size)
    return [[(i, i)] + res[i] + [(half + i, half + i)] for i in range(half)]


def build_bipartite_schedule_v2(size: int) -> List[List[int]]:
    return [[(i, (i + j) % size) for j in range(size)] for i in range(size)]



