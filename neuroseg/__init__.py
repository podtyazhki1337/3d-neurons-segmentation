"""neuroseg: inference, evaluation and data access for the XSoma3D label-free 3D neuron segmentation benchmark.

    from neuroseg import load_model, read_volume
    model = load_model("nnunet", "rat_dgc")          # weights from the Hugging Face Hub
    labels = model.predict(read_volume("stack.tif"))  # (Z, Y, X) instance labels

Each method needs its own environment (see environments/); importing neuroseg itself needs only numpy and tifffile.
"""
from .io import read_volume, write_labels  # noqa: F401
from .methods import load_model  # noqa: F401
from .registry import DATASETS, METHODS  # noqa: F401

__version__ = "1.0.0"
