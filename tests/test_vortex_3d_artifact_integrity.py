"""Deterministic namespace substitutions; synthetic snapshots and VTUs only."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from nek_post import vortex_3d_artifacts as artifacts
from nek_post import vortex_3d_series as series
from nek_post import vortex_3d_workflow as single
from test_vortex_3d_series import frames, run, root, frame_paths, completed_stage, hashes


@pytest.fixture(autouse=True)
def ample_memory(monkeypatch):
    monkeypatch.setattr(single, 'available_memory_bytes', lambda: 10**15)


def substitute(path, preserved, kind):
    """Retain the original bytes; replace only the synthetic directory entry."""
    path.rename(preserved)
    if kind == 'original_link':
        path.symlink_to(preserved)
    elif kind == 'copy_link':
        other = preserved.with_name('other-'+preserved.name)
        shutil.copy2(preserved, other)
        path.symlink_to(other)
    elif kind == 'dangling':
        path.symlink_to(preserved.with_name('absent-'+preserved.name))
    elif kind == 'directory':
        path.mkdir()
    elif kind == 'fifo':
        os.mkfifo(path)
    elif kind == 'empty':
        path.touch()
    elif kind == 'regular_copy':
        shutil.copy2(preserved, path)
    elif kind != 'missing':
        raise AssertionError(kind)


def setup_recovery(tmp_path, monkeypatch):
    case, output = frames(tmp_path)
    run(case, output, indices=(3,))
    stage = completed_stage(case, output, monkeypatch)
    return case, output, stage


@pytest.mark.parametrize('mode', ['recover', 'reuse'])
@pytest.mark.parametrize('artifact', ['vtu', 'metadata'])
@pytest.mark.parametrize('kind', ['original_link', 'copy_link', 'dangling', 'missing',
                                 'directory', 'fifo', 'empty', 'regular_copy'])
def test_artifact_substitution_after_validation_fails_without_publication(
        tmp_path, monkeypatch, mode, artifact, kind):
    if mode == 'recover':
        case, output, stage = setup_recovery(tmp_path, monkeypatch)
        target = stage/Path(frame_paths(case, output)[artifact]).name
    else:
        case, output = frames(tmp_path)
        run(case, output, indices=(1,3))
        target = frame_paths(case, output)[artifact]
    controls = series._control_files(root(case, output))
    protected = hashes([*case.iterdir(), *frame_paths(case, output, 3).values(),
                        controls['pvd'], controls['manifest']])
    original = target.read_bytes()
    old_stat = target.stat()
    preserved = tmp_path/('preserved'+target.suffix)
    called = []
    def progress(message):
        if json.loads(message)['message'] == 'Frame started':
            called.append(True)
            substitute(target, preserved, kind)
            if kind == 'original_link':
                # This is precisely the original stat()-following reproduction.
                current = target.stat()
                assert (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == (
                    old_stat.st_dev, old_stat.st_ino, old_stat.st_size, old_stat.st_mtime_ns)
    with pytest.raises(RuntimeError, match='changed'):
        run(case, output, indices=(1,), resume=True, recover_staging=mode == 'recover', progress=progress)
    assert called == [True]
    assert preserved.read_bytes() == original
    assert hashes(Path(path) for path in protected) == protected
    assert json.loads(controls['summary'].read_text())['status'] == 'failed'
    if mode == 'recover':
        assert stage.is_dir()
        assert not frame_paths(case, output)['vtu'].parent.exists()


@pytest.mark.parametrize('artifact', ['vtu', 'metadata'])
def test_recovery_rechecks_after_other_integrity_work_at_publication_boundary(tmp_path, monkeypatch, artifact):
    case, output, stage = setup_recovery(tmp_path, monkeypatch)
    target = stage/Path(frame_paths(case, output)[artifact]).name
    original = series._unchanged
    started = False
    changed = []
    def progress(message):
        nonlocal started
        if json.loads(message)['message'] == 'Frame started':
            started = True
    def check(*args, **kwargs):
        result = original(*args, **kwargs)
        if started and not changed:
            substitute(target, tmp_path/('preserved'+target.suffix), 'original_link')
            changed.append(True)
        return result
    monkeypatch.setattr(series, '_unchanged', check)
    with pytest.raises(RuntimeError, match='staging changed'):
        run(case, output, indices=(1,), resume=True, recover_staging=True, progress=progress)
    assert changed == [True]
    assert not frame_paths(case, output)['vtu'].parent.exists()


@pytest.mark.parametrize('artifact', ['vtu', 'metadata'])
def test_newly_generated_staging_rechecks_both_files_before_publication(tmp_path, monkeypatch, artifact):
    case, output = frames(tmp_path)
    run(case, output, indices=(3,))
    controls = series._control_files(root(case, output))
    before = hashes([controls['pvd'], controls['manifest'], *frame_paths(case, output, 3).values()])
    destination = single._safe_destination
    changed = []
    def after_path_checks(*args, **kwargs):
        result = destination(*args, **kwargs)
        for stage in root(case, output).glob('.vortex-3d-f00001-*'):
            if list(stage.glob('*.json')) and not changed:
                target = stage/Path(frame_paths(case, output)[artifact]).name
                substitute(target, tmp_path/('preserved'+target.suffix), 'original_link')
                changed.append(True)
        return result
    monkeypatch.setattr(single, '_safe_destination', after_path_checks)
    with pytest.raises(ValueError, match='regular artifact'):
        run(case, output, indices=(1,))
    assert changed == [True]
    assert not frame_paths(case, output)['vtu'].parent.exists()
    assert hashes(Path(path) for path in before) == before


@pytest.mark.parametrize('artifact', ['vtu', 'metadata'])
def test_pvd_publication_rechecks_reused_artifacts_after_xml_readback(tmp_path, monkeypatch, artifact):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    controls = series._control_files(root(case, output))
    before = hashes([controls['pvd'], controls['manifest']])
    target = frame_paths(case, output)[artifact]
    original = series._pvd_entries
    changed = []
    def check(path, records):
        result = original(path, records)
        if Path(path).suffix == '.tmp' and not changed:
            substitute(target, tmp_path/('preserved'+target.suffix), 'original_link')
            changed.append(True)
        return result
    monkeypatch.setattr(series, '_pvd_entries', check)
    with pytest.raises(RuntimeError, match='output changed'):
        run(case, output, indices=(1,2), resume=True)
    assert changed == [True]
    assert hashes(Path(path) for path in before) == before
    assert frame_paths(case, output, 2)['vtu'].is_file()  # Valid orphan remains recoverable.
    assert json.loads(controls['summary'].read_text())['status'] == 'failed'


@pytest.mark.parametrize('artifact', ['vtu', 'metadata'])
def test_last_progress_callback_cannot_make_an_invalid_resume_complete(tmp_path, monkeypatch, artifact):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    target = frame_paths(case, output)[artifact]
    def progress(message):
        if json.loads(message)['message'] == 'Series completed':
            substitute(target, tmp_path/('preserved'+target.suffix), 'original_link')
    with pytest.raises(RuntimeError, match='output changed'):
        run(case, output, indices=(1,), resume=True, progress=progress)
    assert json.loads(series._control_files(root(case, output))['summary'].read_text())['status'] == 'failed'


@pytest.mark.parametrize('kind', ['original_link', 'copy_link', 'regular_copy', 'fifo', 'directory', 'missing'])
def test_no_follow_open_rejects_swaps_between_lstat_and_open(tmp_path, monkeypatch, kind):
    target = tmp_path/'artifact.json'
    target.write_bytes(b'{"valid":true}')
    original = os.open
    opened = []
    def open_after_swap(path, flags, *args, **kwargs):
        assert Path(path) == target
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
        substitute(target, tmp_path/'preserved.json', kind)
        descriptor = original(path, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor
    monkeypatch.setattr(artifacts.os, 'open', open_after_swap)
    with pytest.raises((ValueError, OSError)):
        with artifacts.open_regular_file(target):
            pytest.fail('substituted directory entry was accepted')
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize('kind', ['original_link', 'regular_copy'])
def test_open_descriptor_remains_pinned_and_detects_later_namespace_changes(tmp_path, kind):
    target = tmp_path/'artifact.json'
    payload = b'{"valid":true}'
    target.write_bytes(payload)
    with pytest.raises(ValueError):
        with artifacts.open_regular_file(target) as stream:
            descriptor = stream.fileno()
            substitute(target, tmp_path/'preserved.json', kind)
            assert stream.read() == payload
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_regular_file_controls_keep_json_hashes_and_close_descriptors(tmp_path):
    target = tmp_path/'artifact.json'
    payload = b'{"valid":true}'
    target.write_bytes(payload)
    identity = artifacts.regular_file_identity(target)
    assert stat.S_ISREG(identity[2])
    assert series._stat(target) == identity
    assert series._json(target) == {'valid': True}
    assert artifacts.regular_file_hash(target, expected=identity) == sha256(payload).hexdigest()
    with artifacts.open_regular_file(target, expected=identity) as stream:
        descriptor = stream.fileno()
        assert stream.read() == payload
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize('reader', ['json', 'binary_hash', 'payload'])
def test_expected_identity_rejects_same_content_regular_replacement_before_read(tmp_path, monkeypatch, reader):
    target = tmp_path/'artifact.json'
    target.write_bytes(b'{"valid":true}')
    expected = artifacts.regular_file_identity(target)
    substitute(target, tmp_path/'preserved.json', 'regular_copy')
    monkeypatch.setattr(artifacts, '_validate_saved_grid', lambda *a: pytest.fail('decoded changed inode'))
    with pytest.raises(ValueError):
        if reader == 'json':
            series._json(target, expected=expected)
        elif reader == 'binary_hash':
            artifacts.binary_vtu_fingerprint(target, expected=expected)
        else:
            artifacts.validate_saved_grid(target, {}, expected=expected)


def test_vtk_decoder_uses_the_pinned_inode_and_rejects_a_changed_name(tmp_path, monkeypatch):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    paths = frame_paths(case, output)
    metadata = json.loads(paths['metadata'].read_text())
    decoder = artifacts._validate_saved_grid
    decoded = []
    def decode(path, report):
        assert Path(path).parent == Path('/proc/self/fd')
        substitute(paths['vtu'], tmp_path/'preserved.vtu', 'original_link')
        decoder(path, report)
        decoded.append(True)
    monkeypatch.setattr(artifacts, '_validate_saved_grid', decode)
    with pytest.raises(ValueError):
        artifacts.validate_saved_grid(paths['vtu'], metadata)
    assert decoded == [True]


@pytest.mark.parametrize('artifact', ['vtu', 'metadata'])
def test_uncooperative_change_inside_rename_gap_fails_without_pvd_or_completion(tmp_path, monkeypatch, artifact):
    """No atomic OS primitive locks directory contents and renames them together."""
    case, output, stage = setup_recovery(tmp_path, monkeypatch)
    target = stage/Path(frame_paths(case, output)[artifact]).name
    controls = series._control_files(root(case, output))
    before = hashes([controls['pvd'], controls['manifest']])
    publish = series._publish_directory
    def race(source, destination):
        assert Path(source) == stage
        substitute(target, tmp_path/('preserved'+target.suffix), 'original_link')
        publish(source, destination)
    monkeypatch.setattr(series, '_publish_directory', race)
    with pytest.raises(ValueError, match='regular artifact'):
        run(case, output, indices=(1,), resume=True, recover_staging=True)
    assert hashes(Path(path) for path in before) == before
    assert json.loads(controls['summary'].read_text())['status'] == 'failed'
    # The kernel rename already happened: retain evidence, never delete or
    # replace this invalid final directory, never reference it in the PVD.
    assert frame_paths(case, output)['vtu'].parent.is_dir()
