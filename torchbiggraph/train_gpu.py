#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import ctypes
import ctypes.util
import logging
import os
import time
from collections import defaultdict
from multiprocessing.connection import wait as mp_wait
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import random
import pickle

import torch
import torch.multiprocessing as mp
from torchbiggraph.batching import AbstractBatchProcessor, process_in_batches
from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.graph_storages import EDGE_STORAGES, ENTITY_STORAGES
from torchbiggraph.model import MultiRelationEmbedder
from torchbiggraph.parameter_sharing import ParameterServer, ParameterSharer
from torchbiggraph.row_adagrad import RowAdagrad
from torchbiggraph.stats import Stats, StatsHandler
from torchbiggraph.train_cpu import Trainer, TrainingCoordinator
from torchbiggraph.types import (
    SINGLE_TRAINER,
    Bucket,
    EntityName,
    FloatTensorType,
    GPURank,
    LongTensorType,
    ModuleStateDict,
    OptimizerStateDict,
    Partition,
    Rank,
    Side,
    SubPartition,
)
from torchbiggraph.util import (
    BucketLogger,
    DummyOptimizer,
    EmbeddingHolder,
    SubprocessInitializer,
    allocate_shared_tensor,
    create_pool,
    div_roundup,
    hide_distributed_logging,
    round_up_to_nearest_multiple,
    set_logging_verbosity,
    setup_logging,
    split_almost_equally,
    tag_logs_with_process_name,
    add_logfile,
)
from torchbiggraph.gpu_process import (
    GPU_STATE_TRAIN,
    TimeKeeper,
    SubprocessArgs,
    GPUProcessPool,
    GPU_STATE_EVAL,
    build_bipartite_schedule,
    build_nonbipartite_schedule, 
    build_nonbipartite_schedule_v2,
)
from torchbiggraph.bucket_scheduling import create_buckets_ordered_lexicographically

#from torchbiggraph.gpu_schedule import build_nonbipartite_schedule_pipe


try:
    from torchbiggraph import _C

    CPP_INSTALLED = True
except ImportError:
    CPP_INSTALLED = False


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


NOOP_STATS_HANDLER = StatsHandler()

def get_offset(num_entities,num_subparts,balance):
    num_balance=sum([len(bc) for bc in balance])
    num_last=num_entities-num_balance
    slices=list(split_almost_equally(num_last,num_parts=num_subparts))

    sub_part_offset=[]
    offset=0
    for i in range(num_subparts):
        bc_size=len(balance[i])
        sub_part_offset.append(offset)
        offset+=(bc_size+slices[i].stop-slices[i].start)
    sub_part_offset=torch.tensor(sub_part_offset,dtype=torch.long)

    return sub_part_offset

def get_slices(entity_type,lhs_num_entities,rhs_num_entities,num_subparts,bucket,lhs_offset,rhs_offset):
    subpart_slices={}
    if bucket.lhs!=bucket.rhs:
        for idx in range(num_subparts):
            start=lhs_offset[idx]
            if idx+1<num_subparts:
                end=lhs_offset[idx+1]
            else:
                end=lhs_num_entities
            subpart_slices[entity_type, bucket.lhs, idx]=slice(start,end)
        for idx in range(num_subparts):
            start=rhs_offset[idx]
            if idx+1<num_subparts:
                end=rhs_offset[idx+1]
            else:
                end=rhs_num_entities
            subpart_slices[entity_type, bucket.rhs, idx]=slice(start,end)     
    else:
        for idx in range(num_subparts):
            start=lhs_offset[idx]
            if idx+1<num_subparts:
                end=lhs_offset[idx+1]
            else:
                end=lhs_num_entities
            subpart_slices[entity_type, bucket.lhs, idx]=slice(start,end)
    return subpart_slices
   
def get_perm_and_rev_perm(num_entities,num_subparts,balance,order_tensor):
   
    num_balance=sum([len(bc) for bc in balance])
    num_last=num_entities-num_balance
    #last_od_perm=_C.randperm(num_last,os.cpu_count())
    last_od_perm=torch.randperm(num_last)
    last_od_perm=last_od_perm+num_balance
    slices=list(split_almost_equally(num_last,num_parts=num_subparts))
    perm=torch.tensor([],dtype=torch.long)

    #part_count=[]
    for i in range(num_subparts):
        bc=torch.tensor(balance[i],dtype=torch.long)
        last=last_od_perm[slices[i]]
        part_perm=torch.cat((bc,last),dim=0)
        #part_count.append(part_perm.shape[0])
        perm=torch.cat((perm,part_perm),dim=0)

    rev_perm=order_tensor[perm]
    perm=_C.reverse_permutation(rev_perm,os.cpu_count())
    return perm,rev_perm


class GPUTrainingCoordinator(TrainingCoordinator):
    def __init__(
        self,
        config: ConfigSchema,
        model: Optional[MultiRelationEmbedder] = None,
        trainer: Optional[AbstractBatchProcessor] = None,
        evaluator: Optional[AbstractBatchProcessor] = None,
        rank: Rank = SINGLE_TRAINER,
        subprocess_init: Optional[Callable[[], None]] = None,
        stats_handler: StatsHandler = NOOP_STATS_HANDLER,
        log_path=None,
        eval_config=None
    ):

        super().__init__(
            config, model, trainer, evaluator, rank, subprocess_init, stats_handler,eval_config
        )

        assert config.num_gpus > 0
        if not CPP_INSTALLED:
            raise RuntimeError(
                "GPU support requires C++ installation: "
                "install with C++ support by running "
                "`PBG_INSTALL_CPP=1 pip install .`"
            )

        if config.half_precision:
            for entity in config.entities:
                # need this for tensor cores to work
                assert config.entity_dimension(entity) % 8 == 0
            assert config.batch_size % 8 == 0
            assert config.num_batch_negs % 8 == 0
            assert config.num_uniform_negs % 8 == 0

        assert len(self.holder.lhs_unpartitioned_types) == 0
        assert len(self.holder.rhs_unpartitioned_types) == 0

        num_edge_chunks = self.iteration_manager.num_edge_chunks
        max_edges = 0
        for edge_path in config.edge_paths:
            edge_storage = EDGE_STORAGES.make_instance(edge_path)
            for lhs_part in range(self.holder.nparts_lhs):
                for rhs_part in range(self.holder.nparts_rhs):
                    num_edges = edge_storage.get_number_of_edges(lhs_part, rhs_part)
                    num_edges_per_chunk = div_roundup(num_edges, num_edge_chunks)
                    max_edges = max(max_edges, num_edges_per_chunk)
        self.shared_lhs = allocate_shared_tensor((max_edges,), dtype=torch.long)
        self.shared_rhs = allocate_shared_tensor((max_edges,), dtype=torch.long)
        self.shared_rel = allocate_shared_tensor((max_edges,), dtype=torch.long)

        # fork early for HOGWILD threads
        logger.info("Creating GPU workers...")
        torch.set_num_threads(1)
        self.gpu_pool = GPUProcessPool(
            config.num_gpus,
            subprocess_init,
            {s for ss in self.embedding_storage_freelist.values() for s in ss}
            | {
                self.shared_lhs.storage(),
                self.shared_rhs.storage(),
                self.shared_rel.storage(),
            },
            log_path=log_path
        )
        random.seed(295)
        ##############################################################################
        is_produced=True
        self.sub_part_offset={}
        self.subpart_slices={}
        self.order_tensor={}
        self.balance={}
        self.perm={}
        self.rev_perm={}

        file_path=config.edge_paths[0]
        entity_type="all"
        for  bucket in create_buckets_ordered_lexicographically(
            1, 1
        ):
            edges = edge_storage.load_edges(bucket.lhs, bucket.rhs)
            num_edges = len(edges)

            if bucket.lhs!=bucket.rhs:
                if  is_produced:
                    lhs_count=self.entity_counts[entity_type][bucket.lhs]
                    rhs_count=self.entity_counts[entity_type][bucket.rhs]

                    lhs_order_file_path=file_path+f"/order-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    rhs_order_file_path=file_path+f"/order-{bucket.lhs}-{bucket.rhs}-{bucket.rhs}.pl"

                    with open(lhs_order_file_path,"rb") as lhs_order_file:
                        lhs_order=pickle.load(lhs_order_file)
                    with open(rhs_order_file_path,"rb") as rhs_order_file:
                        rhs_order=pickle.load(rhs_order_file)
                    
                    

                    lhs_balance_file_path=file_path+f"/balance-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    rhs_balance_file_path=file_path+f"/balance-{bucket.lhs}-{bucket.rhs}-{bucket.rhs}.pl"

                    with open(lhs_balance_file_path,"rb") as lhs_balance_file:
                        lhs_balance=pickle.load(lhs_balance_file)
                    with open(rhs_balance_file_path,"rb") as rhs_balance_file:
                        rhs_balance=pickle.load(rhs_balance_file)
                    
                    
                    lhs_offset=get_offset(lhs_count,config.num_subparts,lhs_balance)
                    rhs_offset=get_offset(rhs_count,config.num_subparts,rhs_balance)

                    self.order_tensor[bucket.lhs, bucket.rhs,bucket.lhs]=torch.tensor(lhs_order)
                    self.order_tensor[bucket.lhs, bucket.rhs,bucket.rhs]=torch.tensor(rhs_order)
                    
                    self.balance[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_balance
                    self.balance[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_balance
###########################################################################################
                    # lhs_perm,lhs_rev_perm=get_perm_and_rev_perm(lhs_count,config.num_subparts,lhs_balance,torch.tensor(lhs_order))
                    # rhs_perm,rhs_rev_perm=get_perm_and_rev_perm(rhs_count,config.num_subparts,rhs_balance,torch.tensor(rhs_order))

                    # self.perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_perm
                    # self.perm[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_perm

                    # self.rev_perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_rev_perm
                    # self.rev_perm[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_rev_perm
#################################################################################################
                    self.sub_part_offset[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_offset
                    self.sub_part_offset[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_offset

                    self.subpart_slices[bucket.lhs,bucket.rhs]=get_slices(entity_type,lhs_count,rhs_count,config.num_subparts,bucket,lhs_offset,rhs_offset)
                else:
                    edges = edge_storage.load_edges(bucket.lhs, bucket.rhs)
                    num_edges = len(edges)

                    lhs=edges.lhs.tensor
                    rhs=edges.rhs.tensor

                    lhs_count=self.entity_counts[entity_type][bucket.lhs]
                    rhs_count=self.entity_counts[entity_type][bucket.rhs]

                    lhs_degree=_C.count(lhs_count,lhs,lhs,os.cpu_count())
                    rhs_degree=_C.count(rhs_count,rhs,rhs,os.cpu_count())

                    lhs_order=_C.entity_sort(lhs_degree)
                    rhs_order=_C.entity_sort(rhs_degree)

                    lhs_balance=_C.balanced(lhs_order,lhs_degree,config.num_subparts)
                    rhs_balance=_C.balanced(rhs_order,rhs_degree,config.num_subparts)
###########################################################################################
                    # lhs_perm,lhs_rev_perm=get_perm_and_rev_perm(lhs_count,config.num_subparts,lhs_balance,torch.tensor(lhs_order))
                    # rhs_perm,rhs_rev_perm=get_perm_and_rev_perm(rhs_count,config.num_subparts,rhs_balance,torch.tensor(rhs_order))

                    # self.perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_perm
                    # self.perm[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_perm

                    # self.rev_perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_rev_perm
                    # self.rev_perm[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_rev_perm
#################################################################################################
                    lhs_offset=get_offset(lhs_count,config.num_subparts,lhs_balance)
                    rhs_offset=get_offset(rhs_count,config.num_subparts,rhs_balance)

                    self.order_tensor[bucket.lhs, bucket.rhs,bucket.lhs]=torch.tensor(lhs_order)
                    self.order_tensor[bucket.lhs, bucket.rhs,bucket.rhs]=torch.tensor(rhs_order)

                    self.balance[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_balance
                    self.balance[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_balance

                    self.sub_part_offset[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_offset
                    self.sub_part_offset[bucket.lhs, bucket.rhs,bucket.rhs]=rhs_offset

                    self.subpart_slices[bucket.lhs,bucket.rhs]=get_slices(entity_type,lhs_count,rhs_count,config.num_subparts,bucket,lhs_offset,rhs_offset)


                    lhs_order_file_path=file_path+f"/order-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    rhs_order_file_path=file_path+f"/order-{bucket.lhs}-{bucket.rhs}-{bucket.rhs}.pl"

                    with open(lhs_order_file_path,"wb") as lhs_order_file:
                        lhs_order=pickle.dump(lhs_order,lhs_order_file)
                    with open(rhs_order_file_path,"wb") as rhs_order_file:
                        rhs_order=pickle.dump(rhs_order,rhs_order_file)

                    lhs_balance_file_path=file_path+f"/balance-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    rhs_balance_file_path=file_path+f"/balance-{bucket.lhs}-{bucket.rhs}-{bucket.rhs}.pl"

                    with open(lhs_balance_file_path,"wb") as lhs_balance_file:
                        lhs_balance=pickle.dump(lhs_balance,lhs_balance_file)
                    with open(rhs_balance_file_path,"wb") as rhs_balance_file:
                        rhs_balance=pickle.dump(rhs_balance,rhs_balance_file)
                    
            else:
                if  is_produced:
                    lhs_count=self.entity_counts[entity_type][bucket.lhs]

                    lhs_order_file_path=file_path+f"/order-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    with open(lhs_order_file_path,"rb") as lhs_order_file:
                        lhs_order=pickle.load(lhs_order_file)

                    lhs_balance_file_path=file_path+f"/balance-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    with open(lhs_balance_file_path,"rb") as lhs_balance_file:
                        lhs_balance=pickle.load(lhs_balance_file)
                    
                    lhs_offset=get_offset(lhs_count,config.num_subparts,lhs_balance)

                    self.order_tensor[bucket.lhs, bucket.rhs,bucket.lhs]=torch.tensor(lhs_order)

                    self.balance[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_balance
####################################################################################
                    # lhs_perm,lhs_rev_perm=get_perm_and_rev_perm(lhs_count,config.num_subparts,lhs_balance,torch.tensor(lhs_order))
                    # self.perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_perm
                    # self.rev_perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_rev_perm
####################################################################################
                    
                    self.sub_part_offset[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_offset

                    self.subpart_slices[bucket.lhs,bucket.rhs]=get_slices(entity_type,lhs_count,lhs_count,config.num_subparts,bucket,lhs_offset,lhs_offset)
                else:
                    edges = edge_storage.load_edges(bucket.lhs, bucket.rhs)
                    num_edges = len(edges)
                    
                    lhs=edges.lhs.tensor
                    rhs=edges.rhs.tensor
                    lhs_count=self.entity_counts[entity_type][bucket.lhs]

                    lhs_degree=_C.count(lhs_count,lhs,rhs,os.cpu_count())

                    lhs_order=_C.entity_sort(lhs_degree)
                    #print("OK")
                    lhs_balance=_C.balanced(lhs_order,lhs_degree,config.num_subparts)
                    

                    lhs_offset=get_offset(lhs_count,config.num_subparts,lhs_balance)

                    self.order_tensor[bucket.lhs, bucket.rhs,bucket.lhs]=torch.tensor(lhs_order)

                    self.balance[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_balance

####################################################################################
                    # lhs_perm,lhs_rev_perm=get_perm_and_rev_perm(lhs_count,config.num_subparts,lhs_balance,torch.tensor(lhs_order))
                    # self.perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_perm
                    # self.rev_perm[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_rev_perm
####################################################################################

                    self.sub_part_offset[bucket.lhs, bucket.rhs,bucket.lhs]=lhs_offset

                    self.subpart_slices[bucket.lhs,bucket.rhs]=get_slices(entity_type,lhs_count,lhs_count,config.num_subparts,bucket,lhs_offset,lhs_offset)

                    lhs_order_file_path=file_path+f"/order-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"

                    with open(lhs_order_file_path,"wb") as lhs_order_file:
                        lhs_order=pickle.dump(lhs_order,lhs_order_file)
                    
                    lhs_balance_file_path=file_path+f"/balance-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    with open(lhs_balance_file_path,"wb") as lhs_balance_file:
                        lhs_balance=pickle.dump(lhs_balance,lhs_balance_file)
                    
            
            logger.info(f"Bucket( {bucket.lhs} , {bucket.rhs} ) permed")
            ''' lhs_degree_file_path=file_path+f"degree-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"

            if os.path.exists(lhs_order_file_path):
                with open(lhs_order_file_path,"rb") as lhs_order_file:
                    lhs_order=pickle.load(lhs_order_file)
            else:
                with open(lhs_order_file_path,"wb") as lhs_order_file:
                    lhs_degree_file_path=file_path+f"degree-{bucket.lhs}-{bucket.rhs}-{bucket.lhs}.pl"
                    if os.path.exists(lhs_degree_file_path):
                        with open(lhs_order_file_path,"rb") as lhs_degree_file:
                            lhs_degree=pickle.load(lhs_degree_file)
                    else:
                        with open(lhs_degree_file_path,"wb") as lhs_degree_file:
                            lhs_degree=_C.count(lhs_count,lhs,lhs,os.cpu_count())
                            pickle.dump(lhs_degree,lhs_degree_file) '''










        ##############################################################################

    # override
    def _coordinate_train(self, edges, eval_edge_idxs, epoch_idx) -> Stats:
        tk = TimeKeeper()

        config = self.config
        holder = self.holder
        cur_b = self.cur_b
        bucket_logger = self.bucket_logger
        num_edges = len(edges)
        if cur_b.lhs == cur_b.rhs and config.num_gpus > 1:
            num_subparts= config.num_subparts
        else:
            num_subparts = config.num_subparts

        edges_lhs = edges.lhs.tensor
        edges_rhs = edges.rhs.tensor
        edges_rel = edges.rel
        if eval_edge_idxs is not None:
            bucket_logger.debug("Removing eval edges")
            tk.start("remove_eval")
            num_eval_edges = len(eval_edge_idxs)
            edges_lhs[eval_edge_idxs] = edges_lhs[-num_eval_edges:].clone()
            edges_rhs[eval_edge_idxs] = edges_rhs[-num_eval_edges:].clone()
            edges_rel[eval_edge_idxs] = edges_rel[-num_eval_edges:].clone()
            edges_lhs = edges_lhs[:-num_eval_edges]
            edges_rhs = edges_rhs[:-num_eval_edges]
            edges_rel = edges_rel[:-num_eval_edges]
            bucket_logger.debug(
                f"Time spent removing eval edges: {tk.stop('remove_eval'):.4f} s"
            )

        bucket_logger.debug("Splitting edges into sub-buckets")
        tk.start("mapping_edges")
        # randomly permute the entities, to get a random subbucketing
        perm_holder = {}
        rev_perm_holder = {}
        for (entity, part), embs in holder.partitioned_embeddings.items():
            #randseed=random.randint(0,10000)
            #perm=torch.randperm(self.entity_counts[entity][part])
            perm,rev_perm=get_perm_and_rev_perm(self.entity_counts[entity][part],config.num_subparts,self.balance[cur_b.lhs,cur_b.rhs,part],self.order_tensor[cur_b.lhs,cur_b.rhs,part])
            #perm = _C.randperm(self.entity_counts[entity][part], os.cpu_count(),randseed)
            #perm=self.perm[cur_b.lhs,cur_b.rhs,part]
            #rev_perm=self.rev_perm[cur_b.lhs,cur_b.rhs,part]
            _C.shuffle(embs, perm, os.cpu_count())
            optimizer = self.trainer.partitioned_optimizers[entity, part]
            (optimizer_state,) = optimizer.state.values()
            _C.shuffle(optimizer_state["sum"], perm, os.cpu_count())
            #print(self.sub_part_offset[cur_b.lhs,cur_b.rhs,cur_b.lhs])
            #print(self.sub_part_offset[cur_b.lhs,cur_b.rhs,cur_b.rhs])
            perm_holder[entity, part] = perm
            #rev_perm = _C.reverse_permutation(perm, os.cpu_count())
            rev_perm_holder[entity, part] = rev_perm

        ''' subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice] = {}
        for entity_name, part in holder.partitioned_embeddings.keys():
            num_entities = self.entity_counts[entity_name][part]
            for subpart, subpart_slice in enumerate(
                split_almost_equally(num_entities, num_parts=num_subparts)
            ):
                subpart_slices[entity_name, part, subpart] = subpart_slice
        '''
        subpart_slices=self.subpart_slices[cur_b.lhs,cur_b.rhs]

    
        
        subbuckets = _C.sub_bucket_(
            edges_lhs,
            edges_rhs,
            edges_rel,
            self.entity_counts["all"][cur_b.lhs],
            perm_holder["all", cur_b.lhs],
            self.sub_part_offset[cur_b.lhs,cur_b.rhs,cur_b.lhs],
            self.entity_counts["all"][cur_b.rhs],
            perm_holder["all", cur_b.rhs],
            self.sub_part_offset[cur_b.lhs,cur_b.rhs,cur_b.rhs],
            self.shared_lhs,
            self.shared_rhs,
            self.shared_rel,
            num_subparts,
            num_subparts,
            os.cpu_count(),
        )
        #for key in subbuckets:
        #    print(subbuckets[key][0].shape[0])
        bucket_logger.debug(
            "Time spent splitting edges into sub-buckets: "
            f"{tk.stop('mapping_edges'):.4f} s"
        )
        bucket_logger.debug("Done splitting edges into sub-buckets")
        bucket_logger.debug(f"{subpart_slices}")

        tk.start("scheduling")
        busy_gpus: Set[int] = set()
        all_stats: List[Stats] = []
        if cur_b.lhs != cur_b.rhs:  # Graph is bipartite!!
            gpu_schedules = build_bipartite_schedule(num_subparts,config.num_gpus)
        else:
            gpu_schedules = build_nonbipartite_schedule_v2(num_subparts)#(num_subparts,config.num_gpus)
            #if config.is_morepart:
                #gpu_schedules=build_nonbipartite_schedule_pipe(num_subparts)
            #else:
                #gpu_schedules = build_nonbipartite_schedule(num_subparts)
                #gpu_schedules=build_nonbipartite_schedule_pipe(num_subparts)
        #print(gpu_schedules)
        for s in gpu_schedules:
            s.append(None)
            s.append(None)
        index_in_schedule = [0 for _ in range(self.gpu_pool.num_gpus)]
        locked_parts = set()

        def schedule(gpu_idx: GPURank) -> None:
            if gpu_idx in busy_gpus:
                return
            this_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx]]
            next_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx] + 1]
            if this_bucket is None:
                return
            subparts = {
                (e, cur_b.lhs, this_bucket[0]) for e in holder.lhs_partitioned_types
            } | {(e, cur_b.rhs, this_bucket[1]) for e in holder.rhs_partitioned_types}
            if any(k in locked_parts for k in subparts):
                return
            for k in subparts:
                locked_parts.add(k)
            busy_gpus.add(gpu_idx)
            bucket_logger.debug(
                f"GPU #{gpu_idx} gets {this_bucket[0]}, {this_bucket[1]}"
            )
            for embs in holder.partitioned_embeddings.values():
                assert embs.is_shared()
            self.gpu_pool.schedule(
                gpu_idx,
                SubprocessArgs(
                    state=GPU_STATE_TRAIN,
                    lhs_types=holder.lhs_partitioned_types,
                    rhs_types=holder.rhs_partitioned_types,
                    lhs_part=cur_b.lhs,
                    rhs_part=cur_b.rhs,
                    lhs_subpart=this_bucket[0],
                    rhs_subpart=this_bucket[1],
                    next_lhs_subpart=next_bucket[0]
                    if next_bucket is not None
                    else None,
                    next_rhs_subpart=next_bucket[1]
                    if next_bucket is not None
                    else None,
                    trainer=self.trainer,
                    model=self.model,
                    all_embs=holder.partitioned_embeddings,
                    subpart_slices=subpart_slices,
                    subbuckets=subbuckets,
                    batch_size=config.batch_size,
                    lr=config.lr,
                ),
            )

        for gpu_idx in range(self.gpu_pool.num_gpus):
            schedule(gpu_idx)
        while busy_gpus:
            gpu_idx, result = self.gpu_pool.wait_for_next()
            assert gpu_idx == result.gpu_idx
            all_stats.append(result.stats)
            busy_gpus.remove(gpu_idx)
            this_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx]]
            next_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx] + 1]
            subparts = {
                (e, cur_b.lhs, this_bucket[0]) for e in holder.lhs_partitioned_types
            } | {(e, cur_b.rhs, this_bucket[1]) for e in holder.rhs_partitioned_types}
            for k in subparts:
                locked_parts.remove(k)
            index_in_schedule[gpu_idx] += 1
            if next_bucket is None:
                bucket_logger.debug(f"GPU #{gpu_idx} finished its schedule")
            for gpu_idx in range(config.num_gpus):
                schedule(gpu_idx)

        assert len(all_stats) == num_subparts * num_subparts
        time_spent_scheduling = tk.stop("scheduling")
        bucket_logger.debug(
            f"Time spent scheduling sub-buckets: {time_spent_scheduling:.4f} s"
        )
        bucket_logger.info(f"Speed: {num_edges / time_spent_scheduling:,.0f} edges/sec")

        tk.start("rev_perm")

        for (entity, part), embs in holder.partitioned_embeddings.items():
            rev_perm = rev_perm_holder[entity, part]
            optimizer = self.trainer.partitioned_optimizers[entity, part]
            _C.shuffle(embs, rev_perm, os.cpu_count())
            (state,) = optimizer.state.values()
            _C.shuffle(state["sum"], rev_perm, os.cpu_count())

        bucket_logger.debug(
            f"Time spent mapping embeddings back from sub-buckets: {tk.stop('rev_perm'):.4f} s"
        )

        logger.debug(
            f"_coordinate_train: Time unaccounted for: {tk.unaccounted():.4f} s"
        )

        return Stats.sum(all_stats).average()
