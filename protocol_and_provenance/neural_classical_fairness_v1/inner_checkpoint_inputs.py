"""Resolve a sealed inner result ZIP or its extracted contents without fitting."""
from contextlib import contextmanager
import hashlib
from pathlib import Path, PurePosixPath
import zipfile


def file_binding(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):
            h.update(block)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def inventory_from_zip(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        assert len(names)==len(set(names)), 'DUPLICATE_ARCHIVE_MEMBERS'
        entries={}
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts=PurePosixPath(info.filename).parts
            assert parts and not PurePosixPath(info.filename).is_absolute() and '..' not in parts
            with archive.open(info) as stream:
                h=hashlib.sha256()
                while block:=stream.read(4*1024*1024):
                    h.update(block)
            entries[info.filename]=dict(bytes=info.file_size,sha256=h.hexdigest())
    prefixes={PurePosixPath(n).parts[0] for n in entries}
    assert len(prefixes)==1, 'EXPECTED_SINGLE_ARCHIVE_ROOT'
    return dict(name=path.name,archive=file_binding(path),root_prefix=prefixes.pop(),members=entries)


class AcceptedInnerInput:
    def __init__(self, path, spec, mode):
        self.path=Path(path).resolve()
        self.spec=spec
        self.name=spec['name']
        self.mode=mode
        self.archive_binding=spec['archive']
        if mode=='zip':
            assert file_binding(self.path)==self.archive_binding, 'ARCHIVE_NOT_ACCEPTED'
        elif mode=='extracted':
            for name,binding in spec['members'].items():
                assert file_binding(self.member_path(name))==binding, 'EXTRACTED_MEMBER_HASH:'+name
        else:
            raise AssertionError('INPUT_MODE')

    def member_path(self, name):
        assert name in self.spec['members'], 'UNBOUND_MEMBER'
        parts=PurePosixPath(name).parts
        assert parts[0]==self.spec['root_prefix'] and '..' not in parts, 'MEMBER_PATH'
        path=self.path.joinpath(*parts[1:]).resolve()
        assert path.is_relative_to(self.path) and path!=self.path, 'MEMBER_PATH_ESCAPE'
        return path

    def namelist(self):
        return list(self.spec['members'])

    def read(self, name):
        raw=self.member_path(name).read_bytes()
        expected=self.spec['members'][name]
        assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'], 'MEMBER_CHANGED:'+name
        return raw

    @contextmanager
    def reader(self):
        if self.mode=='zip':
            assert file_binding(self.path)==self.archive_binding, 'ARCHIVE_CHANGED'
            with zipfile.ZipFile(self.path) as archive:
                yield archive
        else:
            yield self

    def receipt(self):
        return dict(mode=self.mode,path=str(self.path),original_archive=self.archive_binding,
                    verified_members=len(self.spec['members']),zip_byte_identity=self.mode=='zip',
                    extracted_member_content_identity=self.mode=='extracted')


def resolve_inner_input(input_root, spec, discovery):
    """Refuse multiple possible inputs; record discovery even when resolution fails."""
    input_root=Path(input_root).resolve()
    assert input_root.is_dir(), 'INPUT_ROOT_MISSING'
    zips=sorted(p for p in input_root.rglob('*.zip') if p.is_file() and
        (p.name==spec['name'] or p.stat().st_size==spec['archive']['bytes']))
    markers=sorted(n for n in spec['members'] if n.endswith('/trajectory/history.json'))
    assert markers, 'NO_HISTORY_MARKER'
    relative=PurePosixPath(markers[0]).parts[1:]
    roots=set()
    for marker in input_root.rglob('history.json'):
        resolved=marker.resolve()
        if resolved.parts[-len(relative):]!=tuple(relative):
            continue
        root=resolved
        for _ in relative:
            root=root.parent
        assert root.is_relative_to(input_root), 'INPUT_ROOT_ESCAPE'
        roots.add(root)
    discovery.update(zip_candidates=[str(p) for p in zips],
        extracted_candidates=[str(p) for p in sorted(roots)],
        candidate_count=len(zips)+len(roots))
    assert len(zips)+len(roots)==1, 'INNER_INPUT_MISSING_OR_AMBIGUOUS: inspect input_discovery.json'
    if zips:
        return AcceptedInnerInput(zips[0],spec,'zip')
    return AcceptedInnerInput(next(iter(roots)),spec,'extracted')
