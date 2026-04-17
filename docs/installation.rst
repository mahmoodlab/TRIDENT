Installation
============

Create a fresh environment (Python 3.10 or 3.11)
-------------------------------------------------

.. code-block:: bash

   conda create -n "trident" python=3.11
   conda activate trident

Clone the repository:

.. code-block:: bash

   git clone https://github.com/mahmoodlab/trident.git && cd trident

Install the package locally:

.. code-block:: bash

   pip install -e .

When to use ``pip install -e .``
--------------------------------

- Use ``pip install -e .`` for the default pipeline (segmentation, patching, feature extraction).
- Use extras only when you need optional model families or converter-specific dependencies.

Optional install profiles:

.. code-block:: bash

   pip install -e ".[patch-encoders]"
   pip install -e ".[slide-encoders]"
   pip install -e ".[convert]"
   pip install -e ".[full]"

When to use each profile:

- ``.[patch-encoders]``: only if you need extra patch encoder dependencies (e.g., CONCH/MUSK/CTransPath).
- ``.[slide-encoders]``: only if you need extra slide encoder dependencies (e.g., PRISM/GigaPath/Madeleine).
- ``.[convert]``: only if you run ``trident convert`` workflows.
- ``.[full]``: install everything optional in one environment.

Platform notes (common install issues)
--------------------------------------

- **Linux**: TRIDENT relies on OpenSlide. If you see import errors for ``openslide``, install system packages first (e.g., ``libopenslide0`` / ``openslide-tools`` depending on your distro).
- **macOS**: OpenSlide is usually easiest via Homebrew (``brew install openslide``).
- **Windows**: OpenSlide support is more fragile; consider WSL2 or use formats supported by non-OpenSlide readers where possible.

GPU vs CPU
----------

- **Segmentation**:
  - ``hest`` / ``grandqc`` are GPU workflows in practice.
  - ``otsu`` is CPU-only and requires **no model weights**.
- **Feature extraction** (patch or slide): GPU strongly recommended.

Hugging Face (gated models + offline clusters)
----------------------------------------------

- If a model is **gated** on Hugging Face, make sure you have access and you are logged in (or set ``HF_TOKEN``).
- If you are **offline**, TRIDENT will raise a clear error telling you what to download.
  Put local checkpoints in:

  - ``trident/segmentation_models/local_ckpts.json``
  - ``trident/patch_encoder_models/local_ckpts.json``
  - ``trident/slide_encoder_models/local_ckpts.json``

Model/cache directory
---------------------

Downloaded weights are cached under:

- ``$TRIDENT_HOME`` if set, otherwise
- ``$XDG_CACHE_HOME/trident`` (defaults to ``~/.cache/trident``)

Preflight checks:

.. code-block:: bash

   trident-doctor --profile base
   trident-doctor --profile patch-encoders --check-gated
   trident-doctor --profile slide-encoders
   trident-doctor --profile convert

When to use each doctor profile:

- ``base``: minimum sanity check before any run.
- ``patch-encoders`` / ``slide-encoders``: run before those specific feature extraction workflows.
- ``convert``: run before TIFF conversion workflows.

.. warning::
   Some pretrained models require additional dependencies. TRIDENT will guide you via error messages when needed.