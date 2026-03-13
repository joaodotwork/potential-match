# CLAUDE.md

## Project Overview

**potential-match** — GPU-accelerated image recognition for a digital zine. Explores a ProQuest Vogue Archive of "How To" articles (1893–2012) through permuted title fragments, using visual clustering, tagging, and semantic search.

Collaboration with Linda van Deursen.

## Setup & Running

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

There are no tests in this project.

## Architecture

The project uses relative imports via `python -m src.cli.main` (no `__init__.py` files exist).

### GPU Acceleration (`src/core/accelerator.py`)

Inherited from metalcut. Uses OpenCV's OpenCL backend which maps to Metal on macOS. Images are converted to `cv2.UMat` for GPU-resident processing. Falls back to CPU (`np.ndarray`) automatically on failure. Both `UMat` and `ndarray` paths must be handled throughout the codebase.

### Source Material

**Image archive:** 313 high-res TIFF scans (~15MB each, 3.9GB total) from ProQuest Vogue Archive. Mix of photographs and illustrations spanning 120 years.

**Metadata spreadsheet** (313 rows): title, year, abstract, authors, subject terms (fashion vocabulary — "Tulip Skirt", "Stiletto", "Bias-Cut"), companies (designers/brands), document type.

**Permutation spreadsheet** (`vogue-how-to-perm-no-dup.xls`): 272 titles decomposed into 3-column fragments:
- 118 openers: "How To Be", "How To Wear", "How To Marry"...
- 237 middles: "A Billionaire", "Lingerie", "Couture"...
- 100 closers: "...Day and Night!", "for Life", ": Architecture"...
- Full combinatorial space: ~2.8M possible permutations

### Key Design Considerations

- All image processing code must handle both `cv2.UMat` (GPU) and `np.ndarray` (CPU) types
- Permuted titles are the navigation interface — not labels, not categories
- Subject terms from metadata are already rich; use them as a bridge between visual clusters and title fragments
- Images are archival scans (TIFF) — expect mixed quality, varied composition (editorial photography, illustrations, text-heavy pages)
