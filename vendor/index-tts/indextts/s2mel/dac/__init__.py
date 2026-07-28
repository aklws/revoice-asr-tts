__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

try:
    import audiotools
except ModuleNotFoundError:
    audiotools = None
else:
    audiotools.ml.BaseModel.INTERN += ["dac.**"]
    audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn

if audiotools is not None:
    from . import model
    from . import utils
    from .model import DAC
    from .model import DACFile
else:
    DAC = None
    DACFile = None
