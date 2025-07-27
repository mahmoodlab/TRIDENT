API Reference
=============

This section documents the **public API** of TRIDENT. 

.. contents::
   :local:
   :depth: 2


Trident
-------

Core of TRIDENT with `Processor` and WSI building.

.. automodule:: trident
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Segmentation Models
-------------------

Semantic segmentation models for tissue vs. background detection and filtering.

.. automodule:: trident.segmentation_models
   :members:
   :undoc-members:


Patch Encoders
--------------

Factory for loading patch-level encoder models.

.. list-table:: 
   :header-rows: 1
   :widths: 18 10 40 32

   * - Patch Encoder
     - Dim
     - Args
     - Link
   * - **UNI**
     - 1024
     - ``--patch_encoder uni_v1 --patch_size 256 --mag 20``
     - `MahmoodLab/UNI <https://huggingface.co/MahmoodLab/UNI>`__
   * - **UNI2-h**
     - 1536
     - ``--patch_encoder uni_v2 --patch_size 256 --mag 20``
     - `MahmoodLab/UNI2-h <https://huggingface.co/MahmoodLab/UNI2-h>`__
   * - **CONCH**
     - 512
     - ``--patch_encoder conch_v1 --patch_size 512 --mag 20``
     - `MahmoodLab/CONCH <https://huggingface.co/MahmoodLab/CONCH>`__
   * - **CONCHv1.5**
     - 768
     - ``--patch_encoder conch_v15 --patch_size 512 --mag 20``
     - `MahmoodLab/conchv1_5 <https://huggingface.co/MahmoodLab/conchv1_5>`__
   * - **Virchow**
     - 2560
     - ``--patch_encoder virchow --patch_size 224 --mag 20``
     - `paige-ai/Virchow <https://huggingface.co/paige-ai/Virchow>`__
   * - **Virchow2**
     - 2560
     - ``--patch_encoder virchow2 --patch_size 224 --mag 20``
     - `paige-ai/Virchow2 <https://huggingface.co/paige-ai/Virchow2>`__
   * - **Phikon**
     - 768
     - ``--patch_encoder phikon --patch_size 224 --mag 20``
     - `owkin/phikon <https://huggingface.co/owkin/phikon>`__
   * - **Phikon-v2**
     - 1024
     - ``--patch_encoder phikon_v2 --patch_size 224 --mag 20``
     - `owkin/phikon-v2 <https://huggingface.co/owkin/phikon-v2/>`__
   * - **Prov-Gigapath**
     - 1536
     - ``--patch_encoder gigapath --patch_size 256 --mag 20``
     - `prov-gigapath <https://huggingface.co/prov-gigapath/prov-gigapath>`__
   * - **H-Optimus-0**
     - 1536
     - ``--patch_encoder hoptimus0 --patch_size 224 --mag 20``
     - `bioptimus/H-optimus-0 <https://huggingface.co/bioptimus/H-optimus-0>`__
   * - **H-Optimus-1**
     - 1536
     - ``--patch_encoder hoptimus1 --patch_size 224 --mag 20``
     - `bioptimus/H-optimus-1 <https://huggingface.co/bioptimus/H-optimus-1>`__
   * - **MUSK**
     - 1024
     - ``--patch_encoder musk --patch_size 384 --mag 20``
     - `xiangjx/musk <https://huggingface.co/xiangjx/musk>`__
   * - **Midnight-12k**
     - 3072
     - ``--patch_encoder midnight12k --patch_size 224 --mag 20``
     - `kaiko-ai/midnight <https://huggingface.co/kaiko-ai/midnight>`__
   * - **Kaiko**
     - 384/768/1024
     - ``--patch_encoder kaiko-vit* --patch_size 256 --mag 20``
     - `Kaiko Collection <https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795>`__
   * - **Lunit**
     - 384
     - ``--patch_encoder lunit-vits8 --patch_size 224 --mag 20``
     - `1aurent/lunit <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`__
   * - **Hibou**
     - 1024
     - ``--patch_encoder hibou_l --patch_size 224 --mag 20``
     - `histai/hibou-L <https://huggingface.co/histai/hibou-L>`__
   * - **CTransPath-CHIEF**
     - 768
     - ``--patch_encoder ctranspath --patch_size 256 --mag 10``
     - —
   * - **ResNet50**
     - 1024
     - ``--patch_encoder resnet50 --patch_size 256 --mag 20``
     - —

.. automodule:: trident.patch_encoder_models
   :members:
   :undoc-members:


Slide Encoders
--------------

Factory for slide-level encoder models.

.. list-table:: 
   :header-rows: 1
   :widths: 20 20 40 32

   * - Slide Encoder
     - Patch Encoder
     - Args
     - Link
   * - **Threads**
     - conch_v15
     - ``--slide_encoder threads --patch_size 512 --mag 20``
     - *(Coming Soon!)*
   * - **Titan**
     - conch_v15
     - ``--slide_encoder titan --patch_size 512 --mag 20``
     - `MahmoodLab/TITAN <https://huggingface.co/MahmoodLab/TITAN>`__
   * - **PRISM**
     - virchow
     - ``--slide_encoder prism --patch_size 224 --mag 20``
     - `paige-ai/Prism <https://huggingface.co/paige-ai/Prism>`__
   * - **CHIEF**
     - ctranspath
     - ``--slide_encoder chief --patch_size 256 --mag 10``
     - `CHIEF <https://github.com/hms-dbmi/CHIEF>`__
   * - **GigaPath**
     - gigapath
     - ``--slide_encoder gigapath --patch_size 256 --mag 20``
     - `prov-gigapath <https://huggingface.co/prov-gigapath/prov-gigapath>`__
   * - **Madeleine**
     - conch_v1
     - ``--slide_encoder madeleine --patch_size 256 --mag 10``
     - `MahmoodLab/madeleine <https://huggingface.co/MahmoodLab/madeleine>`__
   * - **Feather**
     - conch_v15
     - ``--slide_encoder feather --patch_size 512 --mag 20``
     - `MahmoodLab/feather <https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k>`__

.. automodule:: trident.slide_encoder_models
   :members:
   :undoc-members:
