import torch

from mojo_opset.utils.platform import get_platform

from mojo_opset.core.operator import MojoOperator


class MojoStoreLowrank(MojoOperator):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        label_cache: torch.Tensor,
        key_lr: torch.Tensor,
        block_idxs: torch.Tensor,
        token_idxs: torch.Tensor,
        token_num: int,
    ) -> torch.Tensor:
        assert label_cache.dim() == 4, "Expected label_cache is BNSD"
        assert key_lr.dim() == 3, "Expected key_lr is SND"

        if get_platform() == "ilu":
            from mojo_opset.backends.ttx.kernels import store_label_cache_infer

            return store_label_cache_infer(
                label_cache,
                key_lr,
                block_idxs,
                token_idxs,
                token_num,
            )

        label_cache[block_idxs, :, token_idxs, :] = key_lr[:token_num]
        return label_cache
