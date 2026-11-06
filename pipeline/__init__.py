"""The generator: DocPipeline (generator.py) with its checks (checking.py), run.py (dataset, website), pools.py
(text, photos). The other projects it uses are the unchanged copies in originals/, put first on Python's path."""
import sys
from pathlib import Path

_ORIGINALS = str(Path(__file__).resolve().parents[1] / "originals")
if _ORIGINALS not in sys.path:
    sys.path.insert(0, _ORIGINALS)
