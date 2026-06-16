"""MambaMPD: a Mamba-driven segmentation framework for marine pollution detection.

Assembles the full model described in the paper (Fig. 2):

    Input -> FAA (WT block) -> stem -> VSS encoder
          -> SE-ResDecoder (UnetrUpBlock + SE) with deep supervision,
             whose skip connections are refined by Edge-Guided Attention (EGA).

Compared with the original research code, this implementation:

* registers FAA, EGA and the SE blocks as proper sub-modules (they are created
  once in ``__init__`` and trained, instead of being re-instantiated inside
  ``forward`` with fresh random weights);
* is device-agnostic (no hard-coded ``.cuda()`` calls); and
* exposes a clean ``deep_supervision`` switch matching the paper's training
  objective.


* The specific implementation method of the model will be updated simultaneously after the paper is officially published.
"""

