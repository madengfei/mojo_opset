import torch

from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.backends.ttx.kernels import quant_group_linear_reduce_sum_impl
from mojo_opset.core import MojoGroupGemm
from mojo_opset.core import MojoQuantGroupLinearReduceSum


class TTXGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2
        assert self.weight.dim() == 3

        M, K = input.shape

        assert input.stride(-1) == 1, "Please make sure input is K-major."

        if self.trans_weight:
            num_groups, N, BK = self.weight.shape
            strideBN, strideBK = self.weight.stride(1), self.weight.stride(2)
        else:
            num_groups, BK, N = self.weight.shape
            strideBK, strideBN = self.weight.stride(1), self.weight.stride(2)

        assert BK == K, "K of input should be equal to K of self.weight."
        assert num_groups == group_list.numel()

        C = input.new_empty(M, N)

        m_grouped_matmul(input, self.weight, C, group_list, num_groups, M, N, K, strideBN, strideBK, self.trans_weight)

        return C


class TTXQuantGroupLinearReduceSum(MojoQuantGroupLinearReduceSum):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        input: torch.Tensor,
        x1_scale: torch.Tensor,
        x2_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert input.dim() == 3, "input must be 3D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight.contiguous()

        b, _, k = input.shape
        b_w, k_w, _ = weight.shape
        assert b == b_w, "input and weight must have same batch size"
        assert k == k_w, "K of input should be equal to K of weight"

        return quant_group_linear_reduce_sum_impl(input, weight, x1_scale, x2_scale)
