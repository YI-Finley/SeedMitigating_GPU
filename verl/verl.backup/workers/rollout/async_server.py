# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async server factory for different rollout backends."""

import importlib
import logging
from typing import Optional, Type

import ray

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class vLLMAsyncServerAdapter:
    """Adapter to make vLLMHttpServer compatible with AgentLoop interface."""

    def __init__(self, config, rollout_dp_size: int, rollout_dp_rank: int, name_prefix: str):
        """
        Initialize vLLM async server adapter.

        Args:
            config: Full configuration object
            rollout_dp_size: Data parallel size for rollout
            rollout_dp_rank: Data parallel rank for this server
            name_prefix: Prefix for worker names
        """
        self.config = config
        self.rollout_dp_size = rollout_dp_size
        self.rollout_dp_rank = rollout_dp_rank
        self.name_prefix = name_prefix
        self._server_address = None
        self._server_port = None

        logger.info(f"vLLMAsyncServerAdapter initialized: dp_size={rollout_dp_size}, dp_rank={rollout_dp_rank}")

    def get_server_address(self):
        """Get server address and port."""
        if self._server_address is None:
            import ray.util
            self._server_address = ray.util.get_node_ip_address()
            # Use a deterministic port based on rank
            self._server_port = 8000 + self.rollout_dp_rank
        return self._server_address, self._server_port

    async def init_engine(self):
        """Initialize the vLLM engine."""
        logger.info(f"Initializing vLLM engine for rank {self.rollout_dp_rank}")
        # For now, this is a placeholder. Full vLLM integration would happen here
        pass

    async def sleep(self):
        """Put the engine to sleep (free GPU memory)."""
        logger.info(f"Putting vLLM engine to sleep for rank {self.rollout_dp_rank}")
        pass

    async def wake_up(self):
        """Wake up the engine (load model back to GPU)."""
        logger.info(f"Waking up vLLM engine for rank {self.rollout_dp_rank}")
        pass

    async def generate(self, request_id: str, prompt_ids: list, sampling_params: dict):
        """
        Generate text given prompt token ids.

        Args:
            request_id: Unique request identifier
            prompt_ids: List of prompt token IDs
            sampling_params: Sampling parameters for generation

        Returns:
            List of generated token IDs
        """
        logger.warning(
            f"generate() called on vLLMAsyncServerAdapter - "
            f"Full vLLM async server not yet integrated for NPU. "
            f"Consider using the vLLMAsyncRollout directly instead of agent loop mode."
        )
        # This is a simplified placeholder
        # In production, this would communicate with vLLM engine
        return []


def async_server_class(
    rollout_backend: Optional[str] = None,
    rollout_backend_module: Optional[str] = None,
    rollout_backend_class: Optional[str] = None,
) -> Type:
    """
    Factory function to get the appropriate async server class for a rollout backend.

    Args:
        rollout_backend: Name of the rollout backend ('vllm', 'sglang', etc.)
        rollout_backend_module: Custom module path for the rollout backend
        rollout_backend_class: Custom class name for the rollout backend

    Returns:
        The async server class for the specified backend
    """
    # Handle custom async server
    if rollout_backend_module and rollout_backend_class:
        module = importlib.import_module(rollout_backend_module)
        return getattr(module, rollout_backend_class)

    # Handle built-in backends
    if rollout_backend == "vllm":
        # Use our adapter that matches AgentLoop interface
        return vLLMAsyncServerAdapter
    elif rollout_backend == "sglang":
        from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica
        return SGLangReplica
    else:
        raise ValueError(f"Unknown rollout backend: {rollout_backend}")


__all__ = ["async_server_class"]
