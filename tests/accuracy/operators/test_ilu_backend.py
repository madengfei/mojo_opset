import os

import pytest

from tests.utils import bypass_not_implemented

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm
from mojo_opset.utils.platform import get_platform

if get_platform() != "ilu":
    pytest.skip("tests in this module require Iluvatar (ilu) platform")

def _mojo_backend_target() -> str:
    raw = os.environ.get("MOJO_BACKEND")
    if raw is not None:
        raw = raw.strip()
    if not raw:
        return "ixformer"
    return raw.lower()


@pytest.mark.parametrize(
    "core_cls, build",
    [
        (MojoRMSNorm, lambda: MojoRMSNorm(norm_size=128, eps=1e-5)),
        (MojoLayerNorm, lambda: MojoLayerNorm(norm_size=128, eps=1e-5)),
        (MojoResidualAddRMSNorm, lambda: MojoResidualAddRMSNorm(norm_size=128, eps=1e-5)),
        (MojoResidualAddLayerNorm, lambda: MojoResidualAddLayerNorm(norm_size=128, eps=1e-5)),
    ],
)
@bypass_not_implemented
def test_mojo_backend_normalization_dispatch(core_cls, build):
    expected = _mojo_backend_target()
    print(f"backend: {expected}")

    impl_cls = core_cls._registry.get(expected)
    got = getattr(impl_cls, "_backend", None)
    op = build()
    assert getattr(type(op), "_backend", None) == got
