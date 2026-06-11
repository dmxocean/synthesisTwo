# Core Document Assembly

The central orchestration layer for synthesizing full A4 documents: this module manages the integration of diverse visual layers into a cohesive and historically accurate document representation

## Assembly Logic

The assembly process follows a structured multi-layer approach:
*   **Background Initialization**: loads an epoch-specific paper texture harvested from the original archives
*   **Layout Application**: retrieves a blueprint corresponding to the selected historical period
*   **Layer Composition**: iteratively renders printed text, handwriting blocks, and noise artifacts onto the base texture
*   **Ink Simulation**: applies color-space transformations to simulate realistic charcoal and ink-on-paper effects
*   **Coordinate Mapping**: translates all local region coordinates into a global A4 page space to ensure high-precision ground truth generation

## Main Components

*   **`assembler.py`**: primary coordinator responsible for compositing multiple layers into a single A4 image
*   **Metadata Integration**: logic for generating synchronized semantic masks and COCO-compatible annotations
