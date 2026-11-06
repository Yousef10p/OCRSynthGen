# originals/

Code from other projects, copied here unchanged so the generator needs nothing but this repository.
`pipeline/__init__.py` puts this folder first on Python's path, so `import synthdoc`, `import ocrsmith` and the
others load these copies. Fixes we need are applied in memory in `pipeline/checking.py`; the files here stay
as their authors wrote them. Each folder keeps its project's licence.

| Folder | Project | Version | Licence |
|---|---|---|---|
| `synthdoc/` | [MatterWaveAI/synthdoc](https://github.com/MatterWaveAI/synthdoc), `src/synthdoc` | commit 1c4506d | MIT |
| `synthdoc_arabic/` | SynthDoc's Arabic extension (`synthdoc-arabic-test/arabic_ext`) | 0.1.0 | |
| `ocrsmith/` | [atlasia-ma/OCRSmith](https://github.com/atlasia-ma/OCRSmith), `src/ocrsmith` | commit 7ebaef8 (v1.3.0) | MIT |
| `ocrsmith_assets/` | OCRSmith's `assets/`: its 13 Arabic fonts and paper backgrounds | commit 7ebaef8 | MIT (fonts: their own licences) |
| `scan_simulator/` | [s1mb1o/scan-simulator](https://github.com/s1mb1o/scan-simulator) | commit 5befedf | MIT |
| `augraphy/` | [sparkfish/augraphy](https://github.com/sparkfish/augraphy) | 8.2.6 | MIT |
| `surya/inference/` | [datalab-to/surya](https://github.com/datalab-to/surya): `compact.py` (`encode()` / `decode()`), `prompts.py` (the prompts, allowed tags and labels), `schema.py` (what `prompts.py` imports) | commit 9d5aadf | Apache-2.0 |
