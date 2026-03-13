# potential-match

GPU-accelerated image recognition for a digital zine — tagging, clustering, and semantic search across an image archive navigated by 64,000 permuted titles.

## Concept

An image archive explored through permuted text titles. CLIP embeddings bridge the gap between title permutations and visual content, enabling semantic navigation where titles "find" their images.

## Architecture

```mermaid
flowchart TD
    Archive["Image Archive"] --> GPU["MetalAccelerator
    GPU / CPU fallback"]
    Titles["64,000 Permuted Titles"] --> Embed["CLIP Text Encoder"]

    GPU --> ImgEmbed["CLIP Image Encoder"]
    ImgEmbed --> VecDB["Vector Store"]
    Embed --> VecDB

    VecDB --> Tag["Tagging
    zero-shot classification"]
    VecDB --> Cluster["Clustering
    visual grouping"]
    VecDB --> Search["Semantic Search
    title → image matching"]

    Tag --> Zine["Digital Zine"]
    Cluster --> Zine
    Search --> Zine
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Origin

GPU acceleration module forked from [metalcut](https://github.com/joaodotwork/metalcut).
