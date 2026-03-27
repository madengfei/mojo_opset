from functools import lru_cache

import triton
import triton.language as tl

try:
    from triton.runtime.libentry import libentry as _libentry_impl
except ImportError:

    def _libentry_impl():
        def _decorator(fn):
            return fn

        return _decorator


libentry = _libentry_impl

VEC_ALIGN_BYTES = 256


def ilu_grid_dim_from_row_tasks(n_row_tasks: int) -> int:
    """
    ILU Triton may fail to compile some kernels when grid.x is very small (e.g. 1).
    Match historical behavior by using at least num_vectorcore programs while still
    allowing larger grids when ceil(n_rows / BLOCK_M) exceeds that count.
    """
    n_tasks = int(n_row_tasks)
    if n_tasks <= 0:
        return 0
    nvc = 256
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        raw = props.get("num_vectorcore", props.get("num_aicore"))
        if raw is not None:
            nvc = int(raw)
    except Exception:
        pass
    return max(n_tasks, nvc)


# @lru_cache(maxsize=1)
# def get_num_cores(op_type="vector"):
#     assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
#     props = triton.runtime.driver.active.utils.get_device_properties("iluvatar")
#     if op_type == "vector":
#         return props["num_vectorcore"]
#     # ILU 驱动若未暴露 cube 核数，与 NPU 的 num_aicore 对齐语义时回退到 vectorcore
#     return int(props.get("num_aicore") or props["num_vectorcore"])

exp = tl.exp
exp2 = tl.math.exp2
log = tl.log
log2 = tl.log2
