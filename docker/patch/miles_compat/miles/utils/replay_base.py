"""No-op stand-in for miles' routing/indexer replay managers.

The radixark/Megatron-LM miles-main fork (used by the ROCm build) imports
`miles.utils.replay_base.{routing_replay_manager,indexer_replay_manager}` from its
MoE router / moe_utils / DSA paths. vime does not ship the `miles` package, so this
provides no-op managers: replay is disabled and routing falls back to the normal
top-k path (identical behaviour to the feature being off). MoE models such as
Qwen3-30B-A3B then build/convert/train without the miles dependency.
"""


class _NoOpReplayManager:
    def register_to_module(self, module, name=None):
        return None

    def get_topk_fn(self, topk_fn, return_probs=False):
        return topk_fn


routing_replay_manager = _NoOpReplayManager()
indexer_replay_manager = _NoOpReplayManager()
