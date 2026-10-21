"""Check released file hashes; no model execution or external data reads."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
for item in m['files']:
 p=(root/item['path']).resolve()
 assert p.is_relative_to(root) and p.is_file(),item['path']
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==item['bytes'] and d==item['sha256'],item['path']
print(json.dumps(dict(status='PUBLIC_RELEASE_HASHES_PASS',files=len(m['files']))))
