CARD-SDK
========

A unified SDK for loading, visualizing, and experimenting with the **CARD Dataset**.  
This repo provides both quick visualization utilities (via `Rerun <https://rerun.io>`_) and a flexible PyTorch dataloader for training & evaluation.

---

⚡ Quick Visualization
----------------------

You have two ways to quickly launch the visualizer:

1. Run the script directly::

      python scripts/vis.py <sequence_path> --send_agg_depth

2. Or, after installation, use the CLI entrypoint::

      card-viz -h

This opens the Rerun visualizer and shows the dataset with options like ``--send-agg-depth`` or ``--send-lidar-intensities``.

Example::

   python scripts/vis.py /media/T7/CARD/sequences/seq01 --send_agg_depth

---

🚀 Installation
---------------

Clone the repository and install it in editable/development mode::

   git clone https://github.com/xx/CARD-SDK.git
   cd CARD-SDK
   pip install -e .

This makes the ``card_dataset`` package and the ``card-viz`` command available system-wide.

---

🔧 Using the Data Loader
------------------------

The SDK includes a **unified PyTorch dataset** for CARD.

Example::

   from card_dataset import UnifiedCardDataset

   ds = UnifiedCardDataset(
       base_root="/media/T7/CARD/",
       split_file="train_split.txt",
       context_offsets=(-1, 0, +1),   # load ±1 context frames
       stereo=True,                   # load stereo pairs
   )

   sample = ds[0]
   print(sample.keys())
   # -> ['meta', 'cam_1', 'cam_0', ...]

Each frame dictionary includes:

- **RGB** (normalized tensor)
- **Sparse LiDAR depth**
- **Aggregated dense depth**
- **Camera intrinsics** (K, distortion coeffs)
- **Poses & relative poses**
- **Context frames** if requested
- **Optional YOLO labels**

---

🎥 Visualization Scripts
------------------------

All visualization/testing utilities are located in ``scripts/``.

1. Rerun Visualizer
~~~~~~~~~~~~~~~~~~~

Explore sequences interactively with `Rerun <https://rerun.io>`_::

   python -m scripts.vis \
      "<-seq-path>" \
      [--send-agg-depth] [--send-dense-depth] [--send-lidar-intensities] [--camera-subsample <N>]

or (if repo is installed)::

   card-viz "<-seq-path>" [options]

Where ``<seq-path>`` is the path to a CARD sequence (e.g., ``/path/to/full_data/x/seq01``).
Options:

- ``--send-agg-depth`` → visualize aggregated dense ground-truth depth
- ``--send-dense-depth`` → log predicted dense depth (if available)
- ``--send-lidar-intensities`` → visualize LiDAR intensity channels
- ``--camera-subsample <N>`` → log every N-th image (faster previews)

This will open a Rerun session with:

- Left/right camera streams
- Depth overlays
- Optional LiDAR & dense depth
- Full trajectory + sensor poses

.. image:: docs/example_rerun.png
   :alt: Example Rerun Screenshot
   :align: center

2. Loader Sanity Check
~~~~~~~~~~~~~~~~~~~~~~

A simpler visualization is provided to **check intrinsics, projections, and pose correctness**::

python -m scripts.viz_dataloader --base_root <base-root-path> --split_file <split-path> --idx 11230

This script will:

- Show the ``agg_depth`` overlay on the center RGB
- Re-project the depth into a **neighbor frame** (using relative pose)
- Verify consistency visually

.. image:: docs/example_projection.png
   :alt: Projection Correctness
   :align: center

---

📦 Features
-----------

- 🔗 **Unified Dataset Loader** → Returns images, sparse/agg depth, LiDAR, poses, context images  
- 🎥 **Context Support** → Load ±N frames (with relative poses)  
- 🖼️ **Visualizer** → Rerun-based, real-time, interactive exploration  
- 📊 **Depth Checks** → Scripts to validate projections & intrinsics  
- ⚡ **Torch-ready** → Plug-and-play ``torch.utils.data.Dataset``  

---

📂 Repo Structure
-----------------

.. code-block:: text

   CARD-SDK/
   │
   ├── card_dataset/
   │   ├── __init__.py          # Package entry
   │   ├── dataloader.py        # Unified PyTorch dataset
   │   ├── torch_augs.py        # Torch-only augmentations
   │   └── ...
   │
   ├── scripts/
   │   ├── viz_dataloader.py    # Rerun visualizer
   │   ├── vis.py               # Thin Rerun launcher
   │   └── ...
   ├── src/
   │   ├── cli.py               # CLI entrypoint
   │   └── rerun_vis.py       # Rerun visualizer logic
   └── README.rst

---

🖼️ Example Outputs
-------------------

Rerun Visualizer
~~~~~~~~~~~~~~~~

.. image:: docs/example_rerun.png
   :alt: Example Rerun
   :align: center

Projection Correctness
~~~~~~~~~~~~~~~~~~~~~~

.. image:: docs/example_projection.png
   :alt: Projection Correctness
   :align: center

---

