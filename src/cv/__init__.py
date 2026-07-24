"""Computer vision modules for optical flow analysis.

NOTE: this module previously also imported a `FeatureExtractor` from a
`.feature_extractor` submodule that did not exist anywhere in this repo
(an orphaned import — every real usage in the codebase is
`CNNFeatureExtractor` from `dl_misalignment.models.cnn_feature_extractor`,
an unrelated module). Removed rather than reconstructed, since nothing
outside this file ever referenced `cv.FeatureExtractor`.
"""

from .flow_analyzer import OpticalFlowAnalyzer

__all__ = ["OpticalFlowAnalyzer"]
