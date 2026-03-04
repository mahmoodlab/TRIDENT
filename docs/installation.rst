Installation
============

Create a fresh environment (Python 3.10 or 3.11):

.. code-block:: bash

   conda create -n "trident" python=3.11
   conda activate trident

Clone the repository:

.. code-block:: bash

   git clone https://github.com/mahmoodlab/trident.git && cd trident

Install the package locally:

.. code-block:: bash

   pip install -e .

When to use this command:

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