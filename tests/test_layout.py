"""Pure layout math, property-tested: shards, bands, the planners, URIs and the sort permutation."""

from __future__ import annotations

import math
from itertools import pairwise

import pandas as pd
from hypothesis import example, given, settings
from hypothesis import strategies as st

from annizarr._core._layout import band_plan, dense_shards, major_bands, plan_bands
from annizarr._core._sorting import compute_sort
from annizarr._storage._uri import bucket_prefix, canonical_location, is_remote, scheme, store_name

_DIM = st.integers(min_value=1, max_value=500)
_CHUNK = st.integers(min_value=1, max_value=600)
_SEGMENT = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_."), min_size=1, max_size=20
)
_REMOTE_SCHEMES = ("s3", "gs", "gcs")
_FAST = settings(max_examples=100, deadline=None)


@given(row_chunk=_CHUNK, col_chunk=_CHUNK, n_rows=_DIM, n_cols=_DIM, factor=st.integers(min_value=-5, max_value=8))
@_FAST
def test_dense_shards_invariants(row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int) -> None:
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, factor)
    assert layout.chunks == (row_chunk, col_chunk)
    if factor <= 1:
        assert layout.shards is None and layout.block == layout.chunks
        return
    assert layout.shards is not None and layout.block == layout.shards
    shard_row, shard_col = layout.shards
    assert shard_row % row_chunk == 0 and shard_col % col_chunk == 0  # zarr's own requirement
    # capped at the chunk-grid extent, with equality exactly when the factor is capped
    n_row_chunks, n_col_chunks = math.ceil(n_rows / row_chunk), math.ceil(n_cols / col_chunk)
    assert shard_row <= row_chunk * n_row_chunks and (shard_row == row_chunk * n_row_chunks) == (factor >= n_row_chunks)
    assert shard_col <= col_chunk * n_col_chunks and (shard_col == col_chunk * n_col_chunks) == (factor >= n_col_chunks)
    assert band_plan(n_rows, shard_row)[-1][1] == n_rows and band_plan(n_cols, shard_col)[-1][1] == n_cols


@given(
    n_rows=st.integers(min_value=-1000, max_value=200_000), band_rows=st.integers(min_value=-1000, max_value=200_000)
)
@_FAST
def test_band_plan_invariants(n_rows: int, band_rows: int) -> None:
    bands = band_plan(n_rows, band_rows)
    if n_rows <= 0 or band_rows <= 0:
        assert bands == ()
        return
    assert bands[0][0] == 0 and bands[-1][1] == n_rows
    assert all(end == start for (_, end), (start, _) in pairwise(bands))
    assert all(0 < end - start <= band_rows for start, end in bands)


@given(
    total=st.integers(min_value=0, max_value=5000),
    unit=st.integers(min_value=1, max_value=400),
    bytes_per_item=st.integers(min_value=1, max_value=64),
    cpus=st.integers(min_value=1, max_value=16),
    grain=st.one_of(st.none(), st.integers(min_value=1, max_value=400)),
    budget=st.integers(min_value=1, max_value=20_000),
)
@settings(max_examples=150, deadline=None)
def test_plan_bands_invariants(
    total: int, unit: int, bytes_per_item: int, cpus: int, grain: int | None, budget: int
) -> None:
    bands = plan_bands(total, unit=unit, bytes_per_item=bytes_per_item, cpus=cpus, grain=grain, budget=budget)
    if total == 0:
        assert bands == ()
        return
    assert bands[0][0] == 0 and bands[-1][1] == total
    assert all(a[1] == b[0] for a, b in pairwise(bands))
    sizes = [s1 - s0 for s0, s1 in bands]
    # every band but the ragged last is a whole number of write units, never below one unit
    assert all(size % unit == 0 and size >= unit for size in sizes[:-1])
    # a band exceeds the budget only when a single unit already does
    assert all(size * bytes_per_item <= budget for size in sizes) or unit * bytes_per_item > budget
    if total >= cpus * unit:
        assert len(bands) >= cpus  # every worker gets a band whenever the units allow


def test_plan_bands_aligns_to_the_source_grain_only_when_it_fits() -> None:
    aligned = plan_bands(5000, unit=4, bytes_per_item=1, cpus=1, grain=6, budget=1000)
    assert (aligned[0][1] - aligned[0][0]) % math.lcm(4, 6) == 0
    # lcm(4, 7) = 28 at 10 bytes each exceeds a 50-byte budget, so bands fall back to the unit alone
    plain = plan_bands(500, unit=4, bytes_per_item=10, cpus=1, grain=7, budget=50)
    assert (plain[0][1] - plain[0][0]) % 4 == 0 and (plain[0][1] - plain[0][0]) % 7 != 0


@given(
    row_nnz=st.lists(st.integers(min_value=0, max_value=50), min_size=0, max_size=200),
    max_nnz=st.integers(min_value=1, max_value=2000),
    cpus=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=150, deadline=None)
def test_major_bands_invariants(row_nnz: list[int], max_nnz: int, cpus: int) -> None:
    indptr = [0]
    for n in row_nnz:
        indptr.append(indptr[-1] + n)
    bands = major_bands(indptr, max_nnz=max_nnz, cpus=cpus)
    n_rows = len(row_nnz)
    if n_rows == 0:
        assert bands == ()
        return
    assert bands[0][0] == 0 and bands[-1][1] == n_rows
    assert all(a[1] == b[0] for a, b in pairwise(bands)) and all(r1 > r0 for r0, r1 in bands)
    # a band's nonzeros fit the budget unless a single row already exceeds it
    assert all(indptr[r1] - indptr[r0] <= max_nnz or r1 - r0 == 1 for r0, r1 in bands)
    nnz = indptr[-1]
    if nnz >= cpus and all(n <= nnz // cpus for n in row_nnz):
        assert len(bands) >= min(cpus, n_rows)


@st.composite
def _remote_uri(draw: st.DrawFn) -> tuple[str, str, str]:
    prefix = "/".join(draw(st.lists(_SEGMENT, min_size=0, max_size=4)))
    return draw(st.sampled_from(_REMOTE_SCHEMES)), draw(_SEGMENT), prefix


@given(sch=st.sampled_from((*_REMOTE_SCHEMES, "file", "http", "https", "")), bucket=_SEGMENT, remote=_remote_uri())
@_FAST
def test_uri_scheme_bucket_prefix_and_canonical_location(sch: str, bucket: str, remote: tuple[str, str, str]) -> None:
    prefix = remote[2]
    uri = f"{sch}://{bucket}/{prefix}" if prefix else f"{sch}://{bucket}/"
    assert is_remote(uri) == (scheme(uri) in set(_REMOTE_SCHEMES))
    if scheme(uri) in _REMOTE_SCHEMES:
        assert bucket_prefix(uri) == (bucket, prefix or None)
    remote_uri = "{}://{}/{}".format(*remote) if prefix else f"{remote[0]}://{remote[1]}/"
    assert canonical_location(canonical_location(remote_uri)) == canonical_location(remote_uri)


@st.composite
def _realistic_path(draw: st.DrawFn) -> tuple[str, str]:
    dirs = draw(st.lists(_SEGMENT, min_size=0, max_size=4))
    tail = draw(_SEGMENT)
    return "/".join((*dirs, tail)) + "/" * draw(st.integers(min_value=0, max_value=4)), tail


@given(data=_realistic_path())
@example(data=("/", "/"))
@example(data=("///", "///"))
@_FAST
def test_store_name_returns_tail_and_strips_trailing_slashes(data: tuple[str, str]) -> None:
    path, tail = data
    assert store_name(path) == (path if set(path) == {"/"} else tail)


@st.composite
def _obs_frame(draw: st.DrawFn) -> tuple[pd.DataFrame, tuple[str, ...]]:
    n_keys = draw(st.integers(min_value=1, max_value=3))
    names = draw(st.lists(st.sampled_from(("batch", "donor", "sample")), min_size=n_keys, max_size=n_keys, unique=True))
    n_rows = draw(st.integers(min_value=1, max_value=200))
    values = st.sampled_from(("a", "b", "c", "d", "e"))
    return pd.DataFrame({n: draw(st.lists(values, min_size=n_rows, max_size=n_rows)) for n in names}), tuple(names)


@given(obs_and_keys=_obs_frame())
@_FAST
def test_compute_sort_groups_are_contiguous_ordered_and_bijective(
    obs_and_keys: tuple[pd.DataFrame, tuple[str, ...]],
) -> None:
    obs, sort_by = obs_and_keys
    perm, ranges = compute_sort(obs, sort_by)
    assert sorted(perm.tolist()) == list(range(len(obs)))
    sorted_keys = list(obs.iloc[perm][list(sort_by)].itertuples(index=False, name=None))
    prev_end, prev_key = 0, None
    for row in ranges.itertuples(index=False):
        start, end = int(row.start), int(row.end)
        key = tuple(getattr(row, c) for c in sort_by)
        assert start == prev_end and start < end  # contiguous, non-empty
        assert all(k == key for k in sorted_keys[start:end])
        assert prev_key is None or prev_key < key  # ascending by the key tuple
        prev_end, prev_key = end, key
    assert prev_end == len(obs)
