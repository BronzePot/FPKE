#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
import time
from functools import partial
from collections import defaultdict
from multiprocessing.connection import wait as mp_wait
from typing import Callable, Generator, List, Optional, Tuple,Callable, Dict, NamedTuple, Set
import ctypes
import os
import random

import torch
import torch.multiprocessing as mp
from torchbiggraph.batching import AbstractBatchProcessor, call, process_in_batches
from torchbiggraph.bucket_scheduling import create_buckets_ordered_lexicographically,create_buckets_ordered_by_layer
from torchbiggraph.checkpoint_manager import CheckpointManager
from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.graph_storages import EDGE_STORAGES,ENTITY_STORAGES
from torchbiggraph.losses import LOSS_FUNCTIONS
from torchbiggraph.model import MultiRelationEmbedder, Scores, make_model
from torchbiggraph.train_util import IterationManager,get_num_edge_chunks
from torchbiggraph.eval import RankingEvaluator
from torchbiggraph.gpu_process import (
    TimeKeeper,
    SubprocessArgs,
    GPUProcessPool,
    GPU_STATE_EVAL,
    build_bipartite_schedule,
    build_nonbipartite_schedule,
)
from torchbiggraph.stats import Stats, average_of_sums
from torchbiggraph.types import (
    UNPARTITIONED,
    Bucket,
    EntityName,
    GPURank,
    Partition,
    SubPartition,
)
from torchbiggraph.util import (
    EmbeddingHolder,
    SubprocessInitializer,
    allocate_shared_tensor,
    div_roundup,
    set_logging_verbosity,
    setup_logging,
    split_almost_equally,
    tag_logs_with_process_name,
    add_logfile,
)
from torchbiggraph import _C

logger = logging.getLogger("torchbiggraph")


def do_eval_and_report_stats(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    entity_counts = None,
    embedding_storage_freelist = None,
    embeddings_buffer = None,
    gpu_pool = None,
    log_path = None,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> Generator[Tuple[Optional[int], Optional[Bucket], Stats], None, None]:
    """Computes eval metrics (mr/mrr/r1/r10/r50) for a checkpoint with trained
    embeddings.
    """
    tag_logs_with_process_name(f"Evaluator")

    num_edge_chunks = get_num_edge_chunks(config)
    iteration_manager = IterationManager(
            config.num_epochs,
            config.edge_paths,
            num_edge_chunks,
            iteration_idx=0,
        )

    if evaluator is None:
        evaluator = RankingEvaluator(
            loss_fn=LOSS_FUNCTIONS.get_class(config.loss_fn)(margin=config.margin),
            relation_weights=[relation.weight for relation in config.relations],
        )

    if config.verbose > 0:
        import pprint

        pprint.PrettyPrinter().pprint(config.to_dict())

    checkpoint_manager = CheckpointManager(config.checkpoint_path)

    def load_embeddings(entity: EntityName, part: Partition,out=None) -> torch.nn.Parameter:
        embs, _ = checkpoint_manager.read(entity, part, out)
        assert embs.is_shared()
        return torch.nn.Parameter(embs)


    holder = EmbeddingHolder(config)

    if model is None:
        model = make_model(config)
    model.share_memory()
    state_dict, _ = checkpoint_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    if entity_counts is None:
        entity_storage = ENTITY_STORAGES.make_instance(config.entity_path)
        entity_counts: Dict[str, List[int]] = {}
        for entity, econf in config.entities.items():
            entity_counts[entity] = []
            for part in range(econf.num_partitions):
                entity_counts[entity].append(entity_storage.load_count(entity, part))

    buffer_size=4
    if embedding_storage_freelist is None:
        embedding_storage_freelist: Dict[
            EntityName, Set[torch.FloatStorage]
        ] = defaultdict(set)
        for entity_type, counts in entity_counts.items():
            max_count = max(counts)
            
            for _ in range(buffer_size):
                embedding_storage_freelist[entity_type].add(
                    allocate_shared_tensor(
                        (max_count, config.entity_dimension(entity_type)),
                        dtype=torch.float,
                    ).storage()
                )
    
    num_edge_chunks = iteration_manager.num_edge_chunks
    max_edges = 0
    for edge_path in config.edge_paths:
        edge_storage = EDGE_STORAGES.make_instance(edge_path)
        for lhs_part in range(holder.nparts_lhs):
            for rhs_part in range(holder.nparts_rhs):
                num_edges = edge_storage.get_number_of_edges(lhs_part, rhs_part)
                num_edges_per_chunk = div_roundup(num_edges, num_edge_chunks)
                max_edges = max(max_edges, num_edges_per_chunk)
    shared_lhs = allocate_shared_tensor((max_edges,), dtype=torch.long)
    shared_rhs = allocate_shared_tensor((max_edges,), dtype=torch.long)
    shared_rel = allocate_shared_tensor((max_edges,), dtype=torch.long)

    if gpu_pool is None:
        #TODO
        gpu_pool=GPUProcessPool(
            config.num_gpus,
            subprocess_init,
            {s for ss in embedding_storage_freelist.values() for s in ss}
            | {
                shared_lhs.storage(),
                shared_rhs.storage(),
                shared_rel.storage(),
            },
            log_path
        )
    
    if embeddings_buffer is None:
        embeddings_buffer = {}
        ''' for entity in holder.lhs_unpartitioned_types | holder.rhs_unpartitioned_types:
            count = entity_counts[entity][UNPARTITIONED]
            s = embedding_storage_freelist[entity].pop()
            dimension = config.entity_dimension(entity)
            embs = torch.FloatTensor(s).view(-1, dimension)[:count]
            #embs = load_embeddings(entity, UNPARTITIONED, embs)
            embeddings_buffer[entity] = (embs,None) '''
        ''' for entity in holder.lhs_partitioned_types | holder.rhs_partitioned_types:
            for idx in range(holder.nparts_lhs):
                part=holder.nparts_lhs-1-idx
                if len(embeddings_buffer)>=buffer_size:
                    break
                if (entity,part) in embeddings_buffer:
                    continue
                count = entity_counts[entity][part]
                s = embedding_storage_freelist[entity].pop()
                dimension = config.entity_dimension(entity)
                embs = torch.FloatTensor(s).view(-1, dimension)[:count]
                embs = load_embeddings(entity, part, embs)
                embeddings_buffer[entity,part]= (embs,None) '''
    
    all_stats: List[Stats] = []
    for epoch_idx, edge_path_idx, edge_chunk_idx in iteration_manager:
        tk = TimeKeeper()
        logger.info(
            f"Starting epoch {epoch_idx + 1} / {iteration_manager.num_epochs}, "
            f"edge path {edge_path_idx + 1} / {iteration_manager.num_edge_paths}, "
            f"edge chunk {edge_chunk_idx + 1} / {iteration_manager.num_edge_chunks}"
        )
        edge_storage = EDGE_STORAGES.make_instance(edge_path)

        all_edge_path_stats = []
        # FIXME This order assumes higher affinity on the left-hand side, as it's
        # the one changing more slowly. Make this adaptive to the actual affinity.
        for bucket in create_buckets_ordered_by_layer(
            holder.nparts_lhs,
            holder.nparts_rhs,
            order=config.bucket_order,
            generator=random.Random()
        ):
           
            all_bucket_stats=[]
            tic = time.perf_counter()
            # logger.info(f"{bucket}: Loading entities")

            old_parts = set(holder.partitioned_embeddings.keys())
            new_parts = {(e, bucket.lhs) for e in holder.lhs_partitioned_types} | {
                (e, bucket.rhs) for e in holder.rhs_partitioned_types
            }

            for entity, part in old_parts - new_parts:
                del holder.partitioned_embeddings[entity, part]
            for entity, part in new_parts - old_parts:
                if (entity, part) in embeddings_buffer:
                    (embs,_)=embeddings_buffer[entity,part]
                    holder.partitioned_embeddings[entity, part] = embs
                else:
                    if len(embeddings_buffer)>=buffer_size:
                        for entity_,part_ in embeddings_buffer:
                            if entity_==entity:
                                break
                        embs,_=embeddings_buffer.pop((entity_, part_))
                        embedding_storage_freelist[entity].add(embs.storage())
                    count = entity_counts[entity][part]
                    s = embedding_storage_freelist[entity].pop()
                    dimension = config.entity_dimension(entity)
                    embs = torch.FloatTensor(s).view(-1, dimension)[:count]
                    embs = load_embeddings(entity, part, embs)
                    embeddings_buffer[entity,part]= (embs,None)
                    holder.partitioned_embeddings[entity, part] = embs
                        
                

            #model.set_all_embeddings(holder, bucket)

            logger.info(f"{bucket}: Loading edges")
            edges = edge_storage.load_chunk_of_edges(
                    bucket.lhs,
                    bucket.rhs,
                    edge_chunk_idx,
                    iteration_manager.num_edge_chunks,
                    shared=True,
                )
            num_edges = len(edges)

            load_time = time.perf_counter() - tic
            tic = time.perf_counter()
            # logger.info(f"{bucket}: Launching and waiting for workers")

            num_subparts = config.num_subparts

            edges_lhs = edges.lhs.tensor
            edges_rhs = edges.rhs.tensor
            edges_rel = edges.rel

            perm_holder = {}
            rev_perm_holder = {}
            for (entity, part), embs in holder.partitioned_embeddings.items():
                perm=torch.randperm(entity_counts[entity][part])
                _C.shuffle(embs, perm, os.cpu_count())
                perm_holder[entity, part] = perm
                rev_perm = _C.reverse_permutation(perm, os.cpu_count())
                rev_perm_holder[entity, part] = rev_perm
            
            subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice] = {}
            for entity_name, part in holder.partitioned_embeddings.keys():
                num_entities = entity_counts[entity_name][part]
                for subpart, subpart_slice in enumerate(
                    split_almost_equally(num_entities, num_parts=num_subparts)
                ):
                    subpart_slices[entity_name, part, subpart] = subpart_slice
            
            subbuckets = _C.sub_bucket(
                edges_lhs,
                edges_rhs,
                edges_rel,
                [entity_counts[r.lhs][bucket.lhs] for r in config.relations],
                [perm_holder[r.lhs, bucket.lhs] for r in config.relations],
                [entity_counts[r.rhs][bucket.rhs] for r in config.relations],
                [perm_holder[r.rhs, bucket.rhs] for r in config.relations],
                shared_lhs,
                shared_rhs,
                shared_rel,
                num_subparts,
                num_subparts,
                os.cpu_count(),
                config.dynamic_relations,
            )

            tk.start("scheduling")
            busy_gpus: Set[int] = set()
            all_stats: List[Stats] = []
            if bucket.lhs != bucket.rhs:  # Graph is bipartite!!
                gpu_schedules = build_bipartite_schedule(num_subparts,config.num_gpus)
            else:
                gpu_schedules = build_nonbipartite_schedule(num_subparts,config.num_gpus)
            
                
            for s in gpu_schedules:
                s.append(None)
                s.append(None)


            index_in_schedule = [0 for _ in range(gpu_pool.num_gpus)]
            def schedule(gpu_idx: GPURank) -> None:
                if gpu_idx in busy_gpus:
                    return
                this_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx]]
                next_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx] + 1]
                if this_bucket is None:
                    return
                subparts = {
                    (e, bucket.lhs, this_bucket[0]) for e in holder.lhs_partitioned_types
                } | {(e, bucket.rhs, this_bucket[1]) for e in holder.rhs_partitioned_types}
                
                
                busy_gpus.add(gpu_idx)
                
                for embs in holder.partitioned_embeddings.values():
                    assert embs.is_shared()
                gpu_pool.schedule(
                    gpu_idx,
                    SubprocessArgs(
                        state=GPU_STATE_EVAL,
                        lhs_types=holder.lhs_partitioned_types,
                        rhs_types=holder.rhs_partitioned_types,
                        lhs_part=bucket.lhs,
                        rhs_part=bucket.rhs,
                        lhs_subpart=this_bucket[0],
                        rhs_subpart=this_bucket[1],
                        next_lhs_subpart=next_bucket[0]
                        if next_bucket is not None
                        else None,
                        next_rhs_subpart=next_bucket[1]
                        if next_bucket is not None
                        else None,
                        trainer=evaluator,
                        model=model,
                        all_embs=holder.partitioned_embeddings,
                        subpart_slices=subpart_slices,
                        subbuckets=subbuckets,
                        batch_size=config.batch_size,
                        lr=config.lr,
                    ),
                )

            for gpu_idx in range(gpu_pool.num_gpus):
                schedule(gpu_idx)
            
            while busy_gpus:
                gpu_idx, result = gpu_pool.wait_for_next()
                assert gpu_idx == result.gpu_idx
                gpu_idx=gpu_idx
                all_bucket_stats.append(result.stats)
                busy_gpus.remove(gpu_idx)
                this_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx]]
                next_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx] + 1]
                subparts = {
                    (e, bucket.lhs, this_bucket[0]) for e in holder.lhs_partitioned_types
                } | {(e, bucket.rhs, this_bucket[1]) for e in holder.rhs_partitioned_types}
            
                index_in_schedule[gpu_idx] += 1
                #if next_bucket is None:
                    #bucket_logger.debug(f"GPU #{gpu_idx} finished its schedule")
                for gpu_idx in range(config.num_gpus):
                    schedule(gpu_idx)

            tk.start("rev_perm")

            for (entity, part), embs in holder.partitioned_embeddings.items():
                rev_perm = rev_perm_holder[entity, part]
                _C.shuffle(embs, rev_perm, os.cpu_count())
            logger.debug(
                f"Time spent mapping embeddings back from sub-buckets: {tk.stop('rev_perm'):.4f} s"
            )
            
            compute_time = time.perf_counter() - tic
            logger.info(
                f"{bucket}: Processed {num_edges} edges in {compute_time:.2g} s "
                f"({num_edges / compute_time / 1e6:.2g}M/sec); "
                f"load time: {load_time:.2g} s"
            )

            total_bucket_stats = Stats.sum(all_bucket_stats)
            all_edge_path_stats.append(total_bucket_stats)
            mean_bucket_stats = total_bucket_stats.average()
            logger.info(
                f"Stats for edge path {edge_path_idx + 1} / {len(config.edge_paths)}, "
                f"bucket {bucket}: {mean_bucket_stats}"
            )

            model.clear_all_embeddings()



        total_edge_path_stats = Stats.sum(all_edge_path_stats)
        all_stats.append(total_edge_path_stats)
        mean_edge_path_stats = total_edge_path_stats.average()
        logger.info("")
        logger.info(
            f"Stats for edge path {edge_path_idx + 1} / {len(config.edge_paths)}: "
            f"{mean_edge_path_stats}"
        )
        logger.info("")


    mean_stats = Stats.sum(all_stats).average()
    logger.info("")
    logger.info(f"Stats: {mean_stats}")
    logger.info("")

    gpu_pool.close()
    gpu_pool.join()

    return None, None, mean_stats

    


def do_eval(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    entity_counts = None,
    embedding_storage_freelist = None,
    embeddings_buffer = None,
    gpu_pool = None,
    log_path = None,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> None:
    # Create and run the generator until exhaustion.
    print("OK")
    do_eval_and_report_stats(
        config, 
        model, 
        evaluator, 
        entity_counts,
        embedding_storage_freelist,
        embeddings_buffer,
        gpu_pool,
        log_path,
        subprocess_init=subprocess_init
        )


def main():
    setup_logging()
    config_help = "\n\nConfig parameters:\n\n" + "\n".join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    opt = parser.parse_args()

    loader = ConfigFileLoader()
    config = loader.load_config(opt.config, opt.param)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    do_eval(config, subprocess_init=subprocess_init)


if __name__ == "__main__":
    main()
