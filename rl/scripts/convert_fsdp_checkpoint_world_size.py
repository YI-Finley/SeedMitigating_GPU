#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
VERL_DIR = os.path.join(ROOT_DIR, "verl")
if VERL_DIR not in sys.path:
    sys.path.insert(0, VERL_DIR)

import torch
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


def create_fsdp_device_mesh(device_type: str, world_size: int):
    return init_device_mesh(device_type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an FSDP checkpoint to a new world size by re-sharding from the saved HuggingFace model."
    )
    parser.add_argument("--source-checkpoint-dir", required=True, help="Path to source global_step_xxx checkpoint dir")
    parser.add_argument("--output-checkpoint-dir", required=True, help="Path to converted global_step_xxx checkpoint dir")
    parser.add_argument(
        "--hf-subdir",
        default="actor/huggingface",
        help="HuggingFace model subdir relative to source checkpoint dir",
    )
    parser.add_argument("--copy-data-state", action="store_true", help="Copy data.pt into converted checkpoint")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype used when materializing the model before re-sharding",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    local_rank, rank, world_size = initialize_global_process_group()
    device_name = get_device_name()
    torch_device = torch.device(f"{device_name}:{local_rank}") if device_name != "cpu" else torch.device("cpu")
    dtype = resolve_dtype(args.dtype)

    source_checkpoint_dir = os.path.abspath(args.source_checkpoint_dir)
    output_checkpoint_dir = os.path.abspath(args.output_checkpoint_dir)
    source_hf_dir = os.path.join(source_checkpoint_dir, args.hf_subdir)
    output_actor_dir = os.path.join(output_checkpoint_dir, "actor")
    output_hf_dir = os.path.join(output_actor_dir, "huggingface")

    if rank == 0:
        if not os.path.isdir(source_checkpoint_dir):
            raise FileNotFoundError(f"Source checkpoint dir not found: {source_checkpoint_dir}")
        if not os.path.isdir(source_hf_dir):
            raise FileNotFoundError(f"Source HuggingFace dir not found: {source_hf_dir}")
        if os.path.exists(output_actor_dir):
            shutil.rmtree(output_actor_dir)
        os.makedirs(output_checkpoint_dir, exist_ok=True)
    torch.distributed.barrier()

    print(
        f"[RESHARD] rank={rank} local_rank={local_rank} world_size={world_size} "
        f"device={device_name} source={source_hf_dir} target={output_actor_dir}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(source_hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_hf_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(torch_device)

    device_mesh = create_fsdp_device_mesh(device_name, world_size)
    mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    model = FSDP(
        model,
        use_orig_params=False,
        device_id=torch_device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=device_mesh,
    )

    checkpoint_config = OmegaConf.create({"save_contents": ["model"], "load_contents": ["model"]})
    checkpoint_manager = FSDPCheckpointManager(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        processing_class=tokenizer,
        checkpoint_config=checkpoint_config,
    )
    checkpoint_manager.save_checkpoint(local_path=output_actor_dir, hdfs_path=None, global_step=0)

    torch.distributed.barrier()
    if rank == 0:
        if os.path.isdir(output_hf_dir):
            shutil.rmtree(output_hf_dir)
        shutil.copytree(source_hf_dir, output_hf_dir)

        src_fsdp_config = os.path.join(source_checkpoint_dir, "actor", "fsdp_config.json")
        dst_meta = {
            "source_checkpoint_dir": source_checkpoint_dir,
            "source_hf_dir": source_hf_dir,
            "source_fsdp_config": None,
            "target_world_size": world_size,
            "target_mesh_shape": [world_size],
            "target_mesh_dim_names": ["fsdp"],
            "load_mode": "model_only",
        }
        if os.path.isfile(src_fsdp_config):
            with open(src_fsdp_config, "r", encoding="utf-8") as f:
                dst_meta["source_fsdp_config"] = json.load(f)
        with open(os.path.join(output_checkpoint_dir, "conversion_meta.json"), "w", encoding="utf-8") as f:
            json.dump(dst_meta, f, indent=2, ensure_ascii=False)

        if args.copy_data_state:
            src_data = os.path.join(source_checkpoint_dir, "data.pt")
            if os.path.isfile(src_data):
                shutil.copy2(src_data, os.path.join(output_checkpoint_dir, "data.pt"))

    torch.distributed.barrier()
    destroy_global_process_group()


if __name__ == "__main__":
    main()