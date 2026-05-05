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
   pip install -e ".[czi]"
   pip install -e ".[omezarr]"
   pip install -e ".[full]"

When to use each profile:

- ``.[patch-encoders]``: extra patch encoder dependencies (e.g., CONCH / MUSK / CTransPath).
- ``.[slide-encoders]``: extra slide encoder dependencies (e.g., PRISM / GigaPath / Madeleine).
- ``.[convert]``: required for ``trident convert`` workflows (BioFormats-backed conversion to pyramidal TIFF).
- ``.[czi]``: enables the Zeiss CZI reader (``--reader_type czi``).
- ``.[omezarr]``: enables the OME-Zarr / NGFF reader (``--reader_type omezarr``).
- ``.[full]``: install all model-related optional extras (does not include ``omezarr``).

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

Preflight checks
----------------

Before launching a long job, validate your install with ``trident-doctor``:

.. code-block:: bash

   trident-doctor --profile base
   trident-doctor --profile patch-encoders --check-gated
   trident-doctor --profile slide-encoders
   trident-doctor --profile convert
   trident-doctor --profile full --check-gated

When to use each profile:

- ``base``: minimum sanity check before any run.
- ``patch-encoders`` / ``slide-encoders``: run before those specific feature extraction workflows.
- ``convert``: run before ``trident convert`` workflows.
- ``full``: validates everything in one go.
- Add ``--check-gated`` to actively probe Hugging Face access for gated repos (network required).

The doctor exits with a non-zero status if any check fails, so you can wire it into CI / job-launch scripts:

.. code-block:: bash

   trident-doctor --profile patch-encoders --format json > doctor.json && python run_batch_of_slides.py ...

.. warning::
   Some pretrained models require additional dependencies. TRIDENT will guide you via error messages when needed.