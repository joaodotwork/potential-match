# CLAUDE.md

## Project Overview

**potential-match** — GPU-accelerated image recognition for a digital zine. Explores an image archive through 64,000 permuted text titles using CLIP embeddings for semantic matching, tagging, clustering, and search.

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

Inherited from metalcut. Uses OpenCV's OpenCL backend which maps to Metal on macOS. Frames/images are converted to `cv2.UMat` for GPU-resident processing. Falls back to CPU (`np.ndarray`) automatically on failure. Both `UMat` and `ndarray` paths must be handled throughout the codebase.

### Key Design Considerations

- All image processing code must handle both `cv2.UMat` (GPU) and `np.ndarray` (CPU) types
- Image archive volume and format TBD — design for flexibility
- 64,000 titles are permutations of an original list — structure TBD
