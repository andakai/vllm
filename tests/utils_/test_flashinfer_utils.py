# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

import vllm.utils.flashinfer as fi


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o755)


@pytest.fixture
def default_cuda_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide any real toolkit, keep ninja on PATH, redirect /usr/local/cuda."""
    for var in ("CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC"):
        monkeypatch.delenv(var, raising=False)
    _make_exe(tmp_path / "venv" / "bin" / "ninja")
    monkeypatch.setenv("PATH", str(tmp_path / "venv" / "bin"))
    home = tmp_path / "usr_local_cuda"
    monkeypatch.setattr(fi, "_DEFAULT_CUDA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def flashinfer_without_cubin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend flashinfer-python is installed without flashinfer-cubin."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            object() if name == "flashinfer" else real_find_spec(name, *args)
        ),
    )
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: False)
    fi.has_flashinfer.cache_clear()
    yield
    fi.has_flashinfer.cache_clear()


def test_has_flashinfer_finds_toolkit_off_path(default_cuda_home: Path):
    """Bare-metal installs keep nvcc in /usr/local/cuda, off PATH, which is
    where FlashInfer's JIT finds it."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    assert fi.has_flashinfer()


def test_has_flashinfer_requires_ninja(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """FlashInfer runs `ninja` from PATH, which lacks the venv's bin directory
    when vLLM is launched without activating the venv."""
    _make_exe(tmp_path / "cuda" / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "cuda" / "bin"))
    assert not fi.has_flashinfer()


# get_hybrid_num_tokens_buckets(81920) above 2048.
_TUNED_BUCKETS = (2048, 2560, 3072, 3584, 4096, 8192, 16384, 32768, 65536, 81920)


@pytest.mark.parametrize(
    "num_tokens,round_up",
    [
        (4224, False),  # 4096-token chunk + replay rows -> 4096
        (8192, False),  # on a bucket
        (12000, True),  # -> 16384
        (16896, False),  # -> 16384, not the 32768 next power of two
        (67584, False),  # -> 65536
        (100000, False),  # above the tuned range: clamp to 81920
    ],
)
def test_runtime_tuning_maps_to_nearer_bucket(num_tokens: int, round_up: bool):
    assert fi._runtime_tuning_round_up(num_tokens, _TUNED_BUCKETS) is round_up


@pytest.mark.parametrize(
    "buckets,num_tokens,tuning,expected",
    [
        (None, 16896, False, None),  # autotune did not run
        (_TUNED_BUCKETS, None, False, None),
        (_TUNED_BUCKETS, 4096, False, None),  # FlashInfer's own mapping
        (_TUNED_BUCKETS, 16896, True, None),  # inside a tuning pass
        (_TUNED_BUCKETS, 16896, False, False),
        (_TUNED_BUCKETS, 12000, False, True),
    ],
)
def test_runtime_tuning_context(
    monkeypatch: pytest.MonkeyPatch,
    buckets: tuple[int, ...] | None,
    num_tokens: int | None,
    tuning: bool,
    expected: bool | None,
):
    calls = []

    def fake_autotune(*args, **kwargs):
        calls.append((args, kwargs))
        return contextlib.nullcontext()

    monkeypatch.setattr(fi, "autotune", fake_autotune)
    monkeypatch.setattr(fi, "is_flashinfer_autotuning", lambda: tuning)
    monkeypatch.setattr(fi, "_runtime_tuning_buckets", None)
    fi.set_flashinfer_runtime_tuning_buckets(buckets)

    with fi.flashinfer_runtime_tuning_context(num_tokens):
        pass

    if expected is None:
        assert calls == []
    else:
        assert calls == [
            ((False,), {"tuning_buckets": _TUNED_BUCKETS, "round_up": expected})
        ]
