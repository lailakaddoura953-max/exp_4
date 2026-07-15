"""
Optical Flow Analyzer using Farneback dense flow

This module computes dense optical flow to track motion between frames and
segments the scene into dynamic/static regions.

Properties validated:
- Property 8: Flow Spatial Dimension Preservation (flow matches frame size)
- Property 4: Universal Confidence Bounds (confidence in [0.0, 1.0])
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from src.models.core import FlowResult


@dataclass
class FlowConfig:
    """Configuration for optical flow computation"""
    pyr_scale: float = 0.5  # Pyramid scale factor
    levels: int = 3  # Number of pyramid levels
    winsize: int = 15  # Window size
    iterations: int = 3  # Number of iterations at each level
    poly_n: int = 5  # Polynomial expansion size
    poly_sigma: float = 1.2  # Gaussian sigma for polynomial expansion
    flags: int = 0  # Operation flags
    
    # Dynamic region segmentation parameters
    flow_threshold: float = 1.0  # Minimum flow magnitude for dynamic regions
    confidence_threshold: float = 0.5  # Minimum confidence for valid flow
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (0.0 < self.pyr_scale < 1.0):
            raise ValueError(f"pyr_scale must be in (0.0, 1.0), got {self.pyr_scale}")
        if self.levels < 1:
            raise ValueError(f"levels must be >= 1, got {self.levels}")
        if self.winsize < 3:
            raise ValueError(f"winsize must be >= 3, got {self.winsize}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.poly_n < 3:
            raise ValueError(f"poly_n must be >= 3, got {self.poly_n}")
        if self.flow_threshold < 0:
            raise ValueError(f"flow_threshold must be non-negative, got {self.flow_threshold}")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}")


class OpticalFlowAnalyzer:
    """
    Compute dense optical flow between consecutive frames
    
    This class uses the Farneback method to compute dense optical flow,
    providing flow vectors and confidence estimates for each pixel.
    """
    
    def __init__(self, config: Optional[FlowConfig] = None):
        """
        Initialize the optical flow analyzer
        
        Args:
            config: Flow computation configuration (uses defaults if None)
        """
        self.config = config or FlowConfig()
        self._flow_count = 0  # Track number of flow computations
    
    def compute_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> FlowResult:
        """
        Compute dense optical flow between two consecutive frames
        
        Args:
            prev_frame: Previous frame (BGR or grayscale)
            curr_frame: Current frame (BGR or grayscale)
        
        Returns:
            FlowResult with flow vectors, confidence, and statistics
        
        Raises:
            ValueError: If frames are invalid or have different dimensions
            
        Properties validated:
            - Property 8: Flow spatial dimensions match frame dimensions
            - Property 4: Confidence values in [0.0, 1.0]
        """
        # Validate inputs
        if prev_frame is None or prev_frame.size == 0:
            raise ValueError("Previous frame cannot be None or empty")
        if curr_frame is None or curr_frame.size == 0:
            raise ValueError("Current frame cannot be None or empty")
        
        # Convert to grayscale if needed
        prev_gray = self._ensure_grayscale(prev_frame)
        curr_gray = self._ensure_grayscale(curr_frame)
        
        # Check dimensions match
        if prev_gray.shape != curr_gray.shape:
            raise ValueError(
                f"Frame dimensions must match: prev {prev_gray.shape} vs curr {curr_gray.shape}"
            )
        
        frame_shape = (prev_gray.shape[0], prev_gray.shape[1])
        
        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=self.config.pyr_scale,
            levels=self.config.levels,
            winsize=self.config.winsize,
            iterations=self.config.iterations,
            poly_n=self.config.poly_n,
            poly_sigma=self.config.poly_sigma,
            flags=self.config.flags
        )
        
        # Calculate flow magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate confidence based on magnitude consistency
        # Higher magnitude with lower variance indicates more confident flow
        confidence = self._calculate_confidence(magnitude, flow)
        
        # Calculate statistics
        mean_magnitude = float(np.mean(magnitude))
        mean_direction = float(np.mean(angle))
        
        # Create FlowResult (automatically validates Properties 4 and 8)
        flow_result = FlowResult(
            flow_vectors=flow,
            confidence=confidence,
            mean_magnitude=mean_magnitude,
            mean_direction=mean_direction,
            frame_shape=frame_shape
        )
        
        self._flow_count += 1
        return flow_result
    
    def segment_dynamic_regions(
        self,
        flow_result: FlowResult
    ) -> np.ndarray:
        """
        Segment scene into dynamic (moving) and static regions
        
        Args:
            flow_result: Computed optical flow
        
        Returns:
            Binary mask where True indicates dynamic regions (moving objects)
        """
        # Calculate flow magnitude
        magnitude, _ = cv2.cartToPolar(
            flow_result.flow_vectors[..., 0],
            flow_result.flow_vectors[..., 1]
        )
        
        # Dynamic regions: high flow magnitude with high confidence
        dynamic_mask = (
            (magnitude > self.config.flow_threshold) &
            (flow_result.confidence > self.config.confidence_threshold)
        )
        
        return dynamic_mask.astype(np.uint8)
    
    def filter_outliers(
        self,
        flow_result: FlowResult,
        method: str = "median"
    ) -> FlowResult:
        """
        Filter outliers from optical flow
        
        Args:
            flow_result: Flow result to filter
            method: Filtering method ("median" or "bilateral")
        
        Returns:
            Filtered FlowResult
        """
        if method == "median":
            # Apply median filter to flow components
            filtered_u = cv2.medianBlur(flow_result.flow_vectors[..., 0], 5)
            filtered_v = cv2.medianBlur(flow_result.flow_vectors[..., 1], 5)
        elif method == "bilateral":
            # Apply bilateral filter for edge-preserving smoothing
            filtered_u = cv2.bilateralFilter(
                flow_result.flow_vectors[..., 0], 9, 75, 75
            )
            filtered_v = cv2.bilateralFilter(
                flow_result.flow_vectors[..., 1], 9, 75, 75
            )
        else:
            raise ValueError(f"Unknown filtering method: {method}")
        
        # Stack filtered components
        filtered_flow = np.stack([filtered_u, filtered_v], axis=-1)
        
        # Recalculate statistics
        magnitude, angle = cv2.cartToPolar(filtered_u, filtered_v)
        mean_magnitude = float(np.mean(magnitude))
        mean_direction = float(np.mean(angle))
        
        # Recalculate confidence
        confidence = self._calculate_confidence(magnitude, filtered_flow)
        
        return FlowResult(
            flow_vectors=filtered_flow,
            confidence=confidence,
            mean_magnitude=mean_magnitude,
            mean_direction=mean_direction,
            frame_shape=flow_result.frame_shape
        )
    
    def get_flow_consistency_score(
        self,
        flow_result: FlowResult,
        reference_direction: Optional[float] = None
    ) -> float:
        """
        Calculate flow consistency score
        
        Args:
            flow_result: Flow result to analyze
            reference_direction: Optional reference direction (radians)
        
        Returns:
            Consistency score in [0.0, 1.0], where 1.0 is perfectly consistent
        """
        magnitude, angle = cv2.cartToPolar(
            flow_result.flow_vectors[..., 0],
            flow_result.flow_vectors[..., 1]
        )
        
        # Weight by confidence
        valid_flow = flow_result.confidence > self.config.confidence_threshold
        
        if not np.any(valid_flow):
            return 0.0
        
        if reference_direction is not None:
            # Measure consistency with reference direction
            angle_diff = np.abs(angle[valid_flow] - reference_direction)
            # Normalize to [0, pi]
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            consistency = 1.0 - (np.mean(angle_diff) / np.pi)
        else:
            # Measure internal consistency (variance of directions)
            angle_variance = np.var(angle[valid_flow])
            # Normalize: low variance = high consistency
            consistency = 1.0 / (1.0 + angle_variance)
        
        return float(np.clip(consistency, 0.0, 1.0))
    
    def _ensure_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed"""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def _calculate_confidence(
        self,
        magnitude: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confidence for flow estimates
        
        Confidence is based on:
        1. Flow magnitude (stronger flow = more confident)
        2. Local consistency (similar flow in neighborhood = more confident)
        
        Returns confidence in [0.0, 1.0]
        """
        # Normalize magnitude to [0, 1]
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            normalized_magnitude = np.clip(magnitude / max_magnitude, 0.0, 1.0)
        else:
            normalized_magnitude = np.zeros_like(magnitude, dtype=np.float32)
        
        # Calculate local variance for consistency measure
        kernel_size = 5
        mean_u = cv2.blur(flow[..., 0].astype(np.float32), (kernel_size, kernel_size))
        mean_v = cv2.blur(flow[..., 1].astype(np.float32), (kernel_size, kernel_size))
        
        var_u = cv2.blur((flow[..., 0].astype(np.float32) - mean_u) ** 2, (kernel_size, kernel_size))
        var_v = cv2.blur((flow[..., 1].astype(np.float32) - mean_v) ** 2, (kernel_size, kernel_size))
        
        local_variance = np.sqrt(var_u + var_v)
        max_var = np.max(local_variance)
        if max_var > 0:
            consistency = np.clip(1.0 - (local_variance / max_var), 0.0, 1.0)
        else:
            consistency = np.ones_like(local_variance, dtype=np.float32)
        
        # Combine magnitude and consistency
        confidence = (normalized_magnitude + consistency) / 2.0
        
        # Final clip to ensure values are strictly in [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
        
        return confidence
    
    @property
    def flows_computed(self) -> int:
        """Get count of flow computations"""
        return self._flow_count
    
    def reset(self):
        """Reset the analyzer state"""
        self._flow_count = 0
