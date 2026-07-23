"""
Structural interfaces (Protocols) for the Camera-Location-Aware Hazard
Rules engine.

orchestrator.py and detection_pipeline.py depend on these interfaces
rather than importing concrete implementations directly. This decouples
both modules from two pre-existing, unrelated broken dependency chains in
this codebase:

  - hazard_detection.container_analyzer.ContainerAnalyzer transitively
    imports cv.flow_analyzer, which imports a src.models.core.FlowResult
    module that does not exist anywhere in this codebase.
  - hazard_detection.frame_sampler.FrameSampler transitively imports
    acquisition.frame_acquisition, which imports a
    src.models.core.SynchronizedFrameBatch module that likewise does not
    exist anywhere in this codebase.

Both breaks trace back to the same missing src.models.core module — a gap
in a different, earlier spec's implementation, not something introduced
by or fixable within this spec. Depending on these Protocols means the
rule engine and DetectionPipeline's opt-in orchestrator wiring can still
be imported, constructed, and unit tested (with a mock, stub, or any
class satisfying the Protocol) even while those chains remain broken. The
real ContainerAnalyzer/FrameSampler classes still satisfy these Protocols
and can be passed in wherever their own import chains are fixed.
"""

from typing import List, Optional, Protocol

from hazard_detection.models import Detection, FrameSequence, HazardEvent


class ContainerAnalyzerProtocol(Protocol):
    """
    Structural interface for a container hazard analyzer.

    Satisfied by hazard_detection.container_analyzer.ContainerAnalyzer,
    but orchestrator.py/detection_pipeline.py never import that class
    directly — only this Protocol, so an unrelated bug in
    ContainerAnalyzer's own import chain (or any dependency it pulls in)
    cannot prevent the rule engine or pipeline from being imported,
    constructed, or unit tested.
    """

    def analyze(
        self,
        detections_per_frame: List[List[Detection]],
        frames: FrameSequence,
    ) -> List[HazardEvent]:
        """Analyze container detections across a frame sequence and
        return the resulting HazardEvent list."""
        ...


class FrameSamplerProtocol(Protocol):
    """
    Structural interface for a frame sampler.

    Satisfied by hazard_detection.frame_sampler.FrameSampler, but
    detection_pipeline.py never imports that class directly — only this
    Protocol, for the same reason as ContainerAnalyzerProtocol above (see
    module docstring): FrameSampler's own import chain is broken by a
    missing src.models.core.SynchronizedFrameBatch module.
    """

    def sample(self, camera_id: str) -> Optional[FrameSequence]:
        """Capture a frame sequence from the specified camera, or None on
        failure/timeout."""
        ...

    def release(self) -> None:
        """Release any currently held frame data from memory."""
        ...
