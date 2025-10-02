CARD-SDK
========

A unified SDK for loading, visualizing, and experimenting with the **CARD Dataset**.  
This repo provides both quick visualization utilities (via `Rerun <https://rerun.io>`_) and a flexible PyTorch dataloader for training & evaluation.

---

🚀 Quick Start
--------------

1. Visualization with Rerun
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to explore the dataset is with the built-in **Rerun visualizer**.

Run::

   python scripts/vis.py <path_to_seq> [options]

Options:

- ``--send_agg_depth`` → visualize aggregated dense **ground-truth depth**
- ``--send_dense_depth`` → log exported **predicted dense depth** (from any depth module)
- ``--send_lidar_intensities`` → visualize LiDAR intensity channels
- ``--camera_subsample <N>`` → log every N-th image (for faster previews)

💡 Combine options to quickly inspect different modalities.

Example::

   python scripts/vis.py /media/T7/CARD/sequences/seq01 --send_agg_depth --camera_subsample 10

This will show:

- Left/right cameras
- Aggregated ground-truth depth (projected into the images)
- Optional LiDAR / predicted depths

.. image:: path/to/example_rerun.png
   :alt: Example Rerun Screenshot
   :align: center

---

2. Using the Data Loader
~~~~~~~~~~~~~~~~~~~~~~~~~

The SDK includes a **unified PyTorch dataset** for CARD.

Install with::

   pip install -e .[torch]

Usage example::

   from card_dataset import UnifiedCardDataset

   ds = UnifiedCardDataset(
       root="/media/T7/CARD/",
       split="train",
       include_context=True,   # load +/- context frames
       include_poses=True,     # return relative poses
   )

   sample = ds[0]
   print(sample.keys())
   # -> ['image_left', 'image_right', 'agg_depth', 'pose', 'pose_context', ...]

---

3. Checking Loader Correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A minimal visualization is provided to check intrinsics, projections, and pose correctness:

Run::

   python card_dataset/viz_sample.py <path_to_seq>

This script will:

- Show the **agg_depth** projected into one image
- Re-project the depth into a **neighbor frame** (using relative pose)
- Verify consistency visually

.. image:: path/to/example_projection.png
   :alt: Projection Correctness
   :align: center

---

📦 Features
-----------

- 🔗 **Unified Dataset Loader** → Returns camera images, aggregated depth, LiDAR, poses, context images  
- 🎥 **Context Support** → Load ±N frames (with relative poses)  
- 🖼️ **Visualizer** → Rerun-based, real-time, highly interactive  
- 📊 **Depth Checks** → Simple scripts to validate projections & camera intrinsics  
- ⚡ **Torch-ready** → Plug-and-play ``torch.utils.data.Dataset``  

---

📂 Repo Structure
-----------------

.. code-block:: text

   card-sdk/
   │
   ├── card_dataset/
   │   ├── dataloader.py      # Unified PyTorch dataset
   │   ├── viz_sample.py      # Sample visualization for depth/pose checks
   │   └── ...
   │
   ├── scripts/
   │   ├── vis.py             # Main Rerun visualizer
   │   └── ...
   │
   └── README.rst

---

🖼️ Example Outputs
-------------------

Rerun Visualizer
~~~~~~~~~~~~~~~~

.. image:: path/to/example_rerun.png
   :alt: Example Rerun
   :align: center

Projection Correctness
~~~~~~~~~~~~~~~~~~~~~~

.. image:: path/to/example_projection.png
   :alt: Projection correctness
   :align: center

---

🤝 Contributing
---------------

Feel free to open issues / PRs for improvements or integration with new depth modules.

---

📜 License
----------

MIT License. See LICENSE for details.
