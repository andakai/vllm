# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS KV Connector high-level components:
- TestHf3fsMockClient      : file-backed mock client I/O correctness
- TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
"""

import atexit
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config, set_mtp_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnector,
    HF3FSKVConnectorStats,
    Hf3fsMetadataClient,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.common import (
    HF3FSConnectorMetadata,
    HF3FSRequestMetadata,
    LoadBlockInfo,
    RequestSchedulingState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request


@pytest.fixture
def hf3fs_scheduler_factory(monkeypatch):
    stored_keys: set[str] = set()
    queried_keys: list[str] = []

    def lookup(self, keys):
        queried_keys[:] = keys
        return [key in stored_keys for key in keys]

    # Only the external metadata service is replaced; configs and requests are real.
    monkeypatch.setattr(Hf3fsMetadataClient, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(Hf3fsMetadataClient, "batch_key_exists", lookup)
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
    handlers = {sig: signal.getsignal(sig) for sig in signals}
    connectors = []
    init_none_hash(sha256)

    def create(
        hash_block_size,
        lookahead,
        role=KVConnectorRole.SCHEDULER,
        extra=None,
        block_size=16,
    ):
        config = create_vllm_config(kv_connector="HF3FSKVConnector")
        if lookahead:
            set_mtp_config(config, lookahead)
        if extra:
            config.kv_transfer_config.kv_connector_extra_config.update(extra)
        cache = KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=8,
                        dtype=torch.float16,
                    ),
                )
            ],
            hash_block_size=hash_block_size,
        )
        connector = HF3FSKVConnector(config, role, cache)
        connectors.append(connector)
        return connector, stored_keys, queried_keys

    yield create
    for connector in connectors:
        connector.close()
        atexit.unregister(connector.close)
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def _hashed_request(tokens, hash_block_size, lookahead):
    return Request(
        request_id="test",
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256, lookahead),
    )


@pytest.fixture
def hf3fs_worker(hf3fs_scheduler_factory, monkeypatch, tmp_path, dist_init):
    """Run real CUDA resources and I/O threads against the file-backed client."""
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs."
        "hf3fs_connector.Hf3fsClient",
        MockHf3fsClient,
    )
    worker, _, _ = hf3fs_scheduler_factory(
        4,
        1,
        KVConnectorRole.WORKER,
        dict(
            hf3fs_storage_path=str(tmp_path),
            hf3fs_file_size=1 << 20,
            hf3fs_client_numjobs=1,
            hf3fs_max_device_buffer_count=8,
        ),
    )
    worker.register_kv_caches({"layer": torch.zeros(8, 1, 16, 8, device="cuda")})
    return worker


def test_failed_multibatch_load_keeps_finished_blocks_until_all_io_completes(
    hf3fs_worker, monkeypatch
):
    """An aborted request keeps every destination until all load batches finish."""
    entered = [Event(), Event()]
    release = [Event(), Event()]

    def locations(self, rank, keys):
        index = int(keys[0])
        entered[index].set()
        assert release[index].wait(timeout=10)
        return [None]

    monkeypatch.setattr(Hf3fsMetadataClient, "get_key_locations", locations)
    manager = hf3fs_worker._async_manager
    first = manager.submit_load_operation("load", [1], ["0"])
    last = manager.submit_load_operation("load", [2], ["1"])
    try:
        assert entered[0].wait(timeout=10)
        assert hf3fs_worker.get_finished({"load"}) == (set(), set())
        release[0].set()
        assert entered[1].wait(timeout=10)
        assert first.result(timeout=10) is False
        assert hf3fs_worker.get_finished(set()) == (set(), set())
        release[1].set()
        assert last.result(timeout=10) is False
        assert hf3fs_worker.get_finished(set()) == ({"load"}, {"load"})
        assert hf3fs_worker.get_block_ids_with_load_errors() == {1, 2}
        assert hf3fs_worker.get_block_ids_with_load_errors() == set()
        assert hf3fs_worker.get_finished(set()) == (set(), set())
    finally:
        for event in release:
            event.set()


@pytest.mark.parametrize("operation", ["save", "load"])
def test_preemption_waits_before_reusing_blocks(hf3fs_worker, monkeypatch, operation):
    entered, release, started = Event(), Event(), Event()

    def service(self, rank, keys):
        entered.set()
        assert release.wait(timeout=10)
        return [("key", -1)] if operation == "save" else [None]

    method = "allocate_pages_for_keys" if operation == "save" else "get_key_locations"
    monkeypatch.setattr(Hf3fsMetadataClient, method, service)
    manager = hf3fs_worker._async_manager
    transfer = getattr(manager, f"submit_{operation}_operation")(
        "preempted", [1], ["key"]
    )
    metadata = HF3FSConnectorMetadata()
    metadata.preempted_req_ids = {"preempted"}

    def preempt():
        started.set()
        hf3fs_worker.handle_preemptions(metadata)

    assert entered.wait(timeout=10)
    with ThreadPoolExecutor(max_workers=1) as executor:
        finished = executor.submit(preempt)
        try:
            assert started.wait(timeout=10)
            assert not finished.done()
        finally:
            release.set()
        finished.result(timeout=10)
    assert transfer.result(timeout=10) is False
    assert hf3fs_worker.get_finished(set()) == (set(), set())


@pytest.mark.cpu_test
def test_resume_replaces_old_blocks_and_uses_current_computed_boundary(
    hf3fs_scheduler_factory,
):
    """Recomputed KV after preemption must not be read from the old allocation."""
    connector, _, _ = hf3fs_scheduler_factory(4, 1)
    request = _hashed_request(list(range(49)), 4, 1)
    connector.get_num_new_matched_tokens(request, 0)
    connector.update_state_after_alloc(request, KVCacheBlocks(([],)), 0)

    def scheduled(new_reqs, cached_reqs, num_tokens):
        return SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens={request.request_id: num_tokens},
            total_num_scheduled_tokens=num_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    initial = scheduled(
        [NewRequestData.from_request(request, ([1, 2],))],
        CachedRequestData.make_empty(),
        32,
    )
    assert connector.build_connector_meta(initial).requests[0].block_ids == [1, 2]
    output = scheduled(
        [],
        CachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids={request.request_id},
            num_computed_tokens=[0],
            new_block_ids=[([6, 7],)],
            new_token_ids=[],
            all_token_ids={},
            num_output_tokens=[0],
        ),
        16,
    )
    metadata = connector.build_connector_meta(output)
    assert len(metadata.requests) == 1
    saved = metadata.requests[0]
    assert saved.block_ids == [6, 7]
    assert saved.token_ids == list(range(16))
    assert len(saved.block_hashes) == 1
    assert saved.save_block_op.skip_leading_blocks == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("hash_block_size", [4, 16])
@pytest.mark.parametrize(
    "lookahead,block_size", [(0, 16), (1, 16), (5, 16), (1, 32), (5, 32)]
)
def test_lookup_separates_lookahead_dependencies(
    hf3fs_scheduler_factory, hash_block_size, lookahead, block_size
):
    """Changing a dependency token must miss the affected block, not its prefix."""
    connector, stored, queried = hf3fs_scheduler_factory(
        hash_block_size, lookahead, block_size=block_size
    )
    tokens = list(range(2 * block_size + max(1, lookahead)))
    request = _hashed_request(tokens, hash_block_size, lookahead)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert len(queried) == 2
    stored.update(queried)
    assert connector.get_num_new_matched_tokens(request, 0) == (2 * block_size, True)

    changed = _hashed_request(tokens[:-1] + [999], hash_block_size, lookahead)
    expected = block_size if lookahead else 2 * block_size
    assert connector.get_num_new_matched_tokens(changed, 0) == (expected, True)


@pytest.mark.cpu_test
def test_save_metadata_waits_for_hash_and_snapshots_the_block_prefix(
    hf3fs_scheduler_factory,
):
    """A known key does not authorize saving uncomputed KV or mutable metadata."""
    connector, _, _ = hf3fs_scheduler_factory(4, 1)
    request = _hashed_request(list(range(32)), 4, 1)
    state = RequestSchedulingState(
        request_id=request.request_id,
        request=request,
        token_ids=request.all_token_ids.copy(),
        allocated_block_ids=[1, 2],
    )
    first = HF3FSRequestMetadata.from_scheduling_state(
        state, 16, block_hashes=connector._get_request_block_hashes(request)
    )
    assert first is not None and first.save_block_op is not None
    assert len(first.block_hashes) == state.num_saved_blocks == 1

    request.append_output_token_ids([32])
    keys = connector._get_request_block_hashes(request)
    assert keys is not None and len(keys) == 2
    second = HF3FSRequestMetadata.from_scheduling_state(state, 16, block_hashes=keys)
    assert second is not None and second.save_block_op is not None
    assert second.save_block_op.skip_leading_blocks == 1
    assert second.block_hashes == keys

    # The third key is known from the prompt, but only two blocks will run.
    future = _hashed_request(list(range(49)), 4, 1)
    assert (
        HF3FSRequestMetadata.from_scheduling_state(
            state, 16, block_hashes=connector._get_request_block_hashes(future)
        )
        is None
    )
    state.token_ids.append(999)
    state.allocated_block_ids.append(3)
    assert first.token_ids == list(range(32))
    assert first.block_ids == [1, 2]
    assert len(first.block_hashes) == 1


@pytest.mark.cpu_test
def test_load_metadata_does_not_schedule_a_save():
    """Async load-only metadata cannot save KV before it has arrived."""
    state = RequestSchedulingState(
        request_id="load", token_ids=list(range(32)), allocated_block_ids=[1, 2]
    )
    meta = HF3FSRequestMetadata.from_scheduling_state(
        state,
        16,
        LoadBlockInfo(0, 2, [1, 2]),
        skip_leading_blocks=2,
        block_hashes=["first", "second"],
    )
    assert meta is not None and meta.load_block_op is not None
    assert meta.save_block_op is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def hf3fs_stats():
    """Fresh HF3FSKVConnectorStats instance."""
    return HF3FSKVConnectorStats()


def _make_cuda_event():
    """Return a real CUDA event when available, otherwise a MagicMock."""
    if torch.cuda.is_available():
        return torch.cuda.Event()
    return MagicMock()


# ===========================================================================
# TestHf3fsMockClient
# ===========================================================================


class TestHf3fsMockClient:
    """Tests for hf3fs_mock_client.Hf3fsClient (file-backend mock)."""

    def test_init_creates_file(self, tmp_path):
        """Initializing the client should create the backing file."""
        path = str(tmp_path / "test_file")
        client = MockHf3fsClient(path=path, size=4096, bytes_per_page=512, entries=4)
        assert os.path.exists(path), "Backing file should be created on init"
        assert os.path.getsize(path) == 4096
        client.close()

    @pytest.mark.parametrize(
        "dtype, bytes_per_page",
        [
            (torch.float32, 512),
            (torch.float16, 256),
            (torch.bfloat16, 256),
        ],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_batch_write_and_read_dtype(self, tmp_path, dtype, bytes_per_page):
        """Write a tensor of the given dtype and verify round-trip correctness."""
        path = str(tmp_path / f"rw_{dtype}")
        client = MockHf3fsClient(
            path=path, size=bytes_per_page * 8, bytes_per_page=bytes_per_page, entries=4
        )
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = bytes_per_page // elem_size
        tensor_write = torch.arange(numel, dtype=dtype)
        event = _make_cuda_event()

        results = client.batch_write([0], [tensor_write], event)
        assert results == [bytes_per_page], f"Write should succeed, got {results}"

        tensor_read = torch.zeros(numel, dtype=dtype)
        results = client.batch_read([0], [tensor_read])
        assert results == [bytes_per_page], f"Read should succeed, got {results}"
        assert torch.equal(tensor_write, tensor_read), (
            "Read tensor should match written tensor"
        )
        client.close()

    def test_batch_read_empty_file_returns_error(self, tmp_path):
        """Reading out-of-bounds offset should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "empty_read")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor_read = torch.zeros(numel, dtype=torch.float32)
        results = client.batch_read([size], [tensor_read])  # offset == size => OOB
        assert results[0] == -1, "Out-of-bounds read should return -1"
        client.close()

    def test_batch_write_out_of_bounds_returns_error(self, tmp_path):
        """Writing at an offset beyond file size should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "oob_write")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor = torch.ones(numel, dtype=torch.float32)
        event = _make_cuda_event()
        results = client.batch_write([size], [tensor], event)  # OOB offset
        assert results[0] == -1, "Out-of-bounds write should return -1"
        client.close()

    def test_multiple_tensors_rw(self, tmp_path):
        """Write multiple tensors at different offsets, then read all back."""
        bytes_per_page = 128
        n = 4
        path = str(tmp_path / "multi_rw")
        client = MockHf3fsClient(
            path=path,
            size=bytes_per_page * n * 2,
            bytes_per_page=bytes_per_page,
            entries=8,
        )
        tensors_write = [
            torch.full((bytes_per_page // 4,), float(i), dtype=torch.float32)
            for i in range(n)
        ]
        offsets = [i * bytes_per_page for i in range(n)]
        event = _make_cuda_event()

        results = client.batch_write(offsets, tensors_write, event)
        assert all(r == bytes_per_page for r in results)

        tensors_read = [
            torch.zeros(bytes_per_page // 4, dtype=torch.float32) for _ in range(n)
        ]
        results = client.batch_read(offsets, tensors_read)
        assert all(r == bytes_per_page for r in results)

        for i, (tw, tr) in enumerate(zip(tensors_write, tensors_read)):
            assert torch.allclose(tw, tr), f"Tensor {i} mismatch after round-trip"
        client.close()

    def test_flush_and_close_no_error(self, tmp_path):
        """flush() and close() should not raise exceptions."""
        path = str(tmp_path / "flush_close")
        client = MockHf3fsClient(path=path, size=1024, bytes_per_page=128, entries=4)
        client.flush()
        client.close()


# ===========================================================================
# TestHF3FSKVConnectorStats
# ===========================================================================


class TestHF3FSKVConnectorStats:
    """Tests for HF3FSKVConnectorStats metric collection and aggregation."""

    def test_initial_is_empty(self, hf3fs_stats):
        """Fresh stats object should report is_empty() == True."""
        assert hf3fs_stats.is_empty() is True

    @pytest.mark.parametrize(
        "task_type, duration_key",
        [
            ("Saved", "save_duration"),
            ("Loaded", "load_duration"),
        ],
        ids=["save", "load"],
    )
    def test_record_success_duration(self, hf3fs_stats, task_type, duration_key):
        """Recording a successful task should update duration list and total count."""
        hf3fs_stats.record_success_task_duration(task_type, 0.5)
        assert not hf3fs_stats.is_empty()
        assert len(hf3fs_stats.data[duration_key]) == 1
        assert hf3fs_stats.data[duration_key][0] == pytest.approx(0.5)
        assert hf3fs_stats.data["num_transfer_task"] == 1

    @pytest.mark.parametrize(
        "task_type, failed_key",
        [
            ("Saved", "num_failed_save"),
            ("Loaded", "num_failed_load"),
        ],
        ids=["save", "load"],
    )
    def test_record_failed_task(self, hf3fs_stats, task_type, failed_key):
        """Recording a failed task should increment the corresponding counter."""
        hf3fs_stats.record_failed_task_count(task_type)
        assert hf3fs_stats.data[failed_key] == 1
        assert hf3fs_stats.data["num_transfer_task"] == 1

    def test_aggregate_two_stats(self):
        """aggregate() should merge save/load duration lists and sum counters."""
        stats1 = HF3FSKVConnectorStats()
        stats1.record_success_task_duration("Saved", 0.1)
        stats1.record_success_task_duration("Loaded", 0.2)

        stats2 = HF3FSKVConnectorStats()
        stats2.record_success_task_duration("Saved", 0.3)
        stats2.record_failed_task_count("Loaded")

        stats1.aggregate(stats2)
        assert stats1.data["save_duration"] == pytest.approx([0.1, 0.3])
        assert stats1.data["load_duration"] == pytest.approx([0.2])
        assert stats1.data["num_failed_load"] == 1
        assert stats1.data["num_transfer_task"] == 4

    def test_reduce_with_data(self):
        """reduce() computes correct averages when data is present."""
        stats = HF3FSKVConnectorStats()
        stats.record_success_task_duration("Saved", 1.0)
        stats.record_success_task_duration("Saved", 3.0)
        result = stats.reduce()
        assert result["Num save task success"] == pytest.approx(2.0, rel=0.01)
        assert result["Num save task failed"] == pytest.approx(0.0, rel=0.01)
        assert result["Avg save duration (ms)"] == pytest.approx(2000.0, rel=0.01)
        # Reduced values must be plain Python scalars so CLI logging renders
        # them without numpy reprs (eg np.float64(...)).
        assert all(not isinstance(v, np.generic) for v in result.values())

    def test_clone_and_reset(self, hf3fs_stats):
        """clone_and_reset() returns a copy with data and resets the original."""
        hf3fs_stats.record_success_task_duration("Saved", 0.7)
        hf3fs_stats.record_success_task_duration("Loaded", 0.4)

        clone = hf3fs_stats.clone_and_reset()
        assert clone.data["num_transfer_task"] == 2
        assert hf3fs_stats.is_empty()
