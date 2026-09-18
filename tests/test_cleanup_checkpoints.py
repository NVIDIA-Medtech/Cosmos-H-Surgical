from pathlib import Path

import pytest

cleanup_checkpoints = pytest.importorskip("tools.cleanup_checkpoints")
build_plan = cleanup_checkpoints.build_plan
delete_candidates = cleanup_checkpoints.delete_candidates
main = cleanup_checkpoints.main
parse_iter_dir = cleanup_checkpoints.parse_iter_dir

COMPONENTS = ("model", "optim_net", "scheduler_net", "trainer")


def _write_checkpoint(root: Path, iteration: int, shards: int = 2) -> Path:
    checkpoint = root / f"iter_{iteration:09d}"
    for component in COMPONENTS:
        component_dir = checkpoint / component
        component_dir.mkdir(parents=True)
        (component_dir / ".metadata").write_bytes(b"metadata")
        for rank in range(shards):
            content = b"" if rank == 0 and shards > 1 else f"rank-{rank}".encode()
            (component_dir / f"__{rank}_0.distcp").write_bytes(content)
    return checkpoint


def _write_pointer(root: Path, iteration: int) -> None:
    (root / "latest_checkpoint.txt").write_text(f"iter_{iteration:09d}\n")


def test_parse_iter_dir_uses_decimal_and_rejects_non_directories(tmp_path: Path) -> None:
    parsed_path = tmp_path / "iter_000000010"
    parsed_path.mkdir()
    (tmp_path / "iter_000000020").write_text("not a directory")

    parsed = parse_iter_dir(parsed_path)

    assert parsed is not None
    assert parsed.iteration == 10
    assert parse_iter_dir(tmp_path / "iter_1.5") is None
    assert parse_iter_dir(tmp_path / "iter_000000020") is None


def test_plan_preserves_policy_and_allows_in_progress_highest(tmp_path: Path) -> None:
    for iteration in (50, 100, 150, 200, 250):
        _write_checkpoint(tmp_path, iteration)
    (tmp_path / "iter_000000300").mkdir()
    _write_pointer(tmp_path, 250)

    plan = build_plan(
        tmp_path.resolve(),
        keep_every=200,
        keep_latest=1,
        keep_before_latest=50,
        keep_exact=(50,),
    )

    assert [item.iteration for item in plan.kept] == [50, 200, 250, 300]
    assert [item.iteration for item in plan.candidates] == [100, 150]
    assert plan.protected_before_latest == (tmp_path / "iter_000000200").resolve()


def test_plan_refuses_candidate_with_mismatched_shard_shape(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, 50, shards=1)
    _write_checkpoint(tmp_path, 100, shards=2)
    _write_pointer(tmp_path, 100)

    with pytest.raises(RuntimeError, match="different checkpoint shape"):
        build_plan(tmp_path.resolve(), 100, 1, None)


def test_delete_refuses_state_change_after_dry_run(tmp_path: Path) -> None:
    old_checkpoint = _write_checkpoint(tmp_path, 50)
    _write_checkpoint(tmp_path, 100)
    _write_pointer(tmp_path, 100)
    plan = build_plan(tmp_path.resolve(), 100, 1, None)
    _write_checkpoint(tmp_path, 150)

    with pytest.raises(RuntimeError, match="state changed"):
        delete_candidates(tmp_path.resolve(), plan, 100, 1, None)
    assert old_checkpoint.is_dir()


def test_delete_removes_only_validated_candidates(tmp_path: Path) -> None:
    old_checkpoint = _write_checkpoint(tmp_path, 50)
    latest_checkpoint = _write_checkpoint(tmp_path, 100)
    _write_pointer(tmp_path, 100)
    plan = build_plan(tmp_path.resolve(), 100, 1, None)

    delete_candidates(tmp_path.resolve(), plan, 100, 1, None)

    assert not old_checkpoint.exists()
    assert latest_checkpoint.is_dir()
    assert (tmp_path / "latest_checkpoint.txt").read_text().strip() == latest_checkpoint.name


def test_main_defaults_to_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old_checkpoint = _write_checkpoint(tmp_path, 50)
    _write_checkpoint(tmp_path, 100)
    _write_pointer(tmp_path, 100)

    assert (
        main(
            [
                "--checkpoint-dir",
                str(tmp_path),
                "--keep-every",
                "100",
                "--keep-latest",
                "1",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "DELETE_CANDIDATE iter_000000050" in output
    assert "DELETE_MODE dry_run" in output
    assert old_checkpoint.is_dir()
