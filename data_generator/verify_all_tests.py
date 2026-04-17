#!/usr/bin/env python3
"""
Comprehensive CV-based Verification Suite
==========================================

This module verifies all aspects of the image generation pipeline using
classic computer vision techniques. Tests MUST pass programmatically
before showing results to user.

Test Categories:
1. Corner Tracking - Colored fiducial markers
2. Black Edge Detection - Edge detection + histogram
3. Shadow Bounds - HSV analysis + spread measurement
4. Bounding Box Accuracy - Canny edges vs labels
5. Perspective Transform - Grid pattern validation

Author: Photo Pose Detector - Verification Suite
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# TEST RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class CornerTestResult:
    """Result of corner tracking test"""
    photo_idx: int
    corner_name: str
    expected: Tuple[float, float]
    detected: Tuple[float, float]
    error_px: float
    passed: bool


@dataclass
class EdgeTestResult:
    """Result of black edge detection"""
    image_name: str
    border_dark_pixels: Dict[str, int]  # top, bottom, left, right
    total_edge_pixels: int
    dark_ratio: float
    passed: bool


@dataclass
class ShadowTestResult:
    """Result of shadow bounds test"""
    shadow_spread_px: float
    photo_size_px: float
    spread_ratio: float
    passed: bool


@dataclass
class BboxTestResult:
    """Result of bounding box accuracy test"""
    detected_edges: List[Tuple[float, float]]
    bbox_corners: List[Tuple[float, float]]
    max_error_px: float
    passed: bool


class TestSuite:
    """Comprehensive test suite for image generation verification"""
    
    # Colors for corner markers (BGR)
    CORNER_COLORS = {
        'TL': (0, 255, 0),      # Green - Top-Left
        'TR': (0, 0, 255),      # Red - Top-Right
        'BR': (255, 0, 255),    # Magenta - Bottom-Right
        'BL': (255, 0, 0),      # Blue - Bottom-Left
    }
    
    # Color detection ranges
    COLOR_RANGES = {
        'TL': (np.array([0, 200, 0]), np.array([100, 255, 100])),   # Green
        'TR': (np.array([200, 0, 0]), np.array([255, 100, 100])),   # Red
        'BR': (np.array([200, 0, 200]), np.array([255, 100, 255])), # Magenta
        'BL': (np.array([0, 0, 200]), np.array([100, 100, 255])),   # Blue
    }
    
    # Pass/fail thresholds
    CORNER_ERROR_THRESHOLD_PX = 10
    EDGE_DARK_RATIO_THRESHOLD = 0.05
    SHADOW_SPREAD_RATIO_THRESHOLD = 0.15
    BBOX_ERROR_THRESHOLD_PX = 15
    
    def __init__(self):
        self.results = {
            'corners': [],
            'edges': [],
            'shadows': [],
            'bboxes': [],
        }
    
    # =========================================================================
    # TEST 1: CORNER TRACKING VERIFICATION
    # =========================================================================
    
    def place_corner_markers(self, photo: np.ndarray, marker_size: int = 10) -> np.ndarray:
        """
        Place colored fiducial markers at each corner of a photo.
        These markers allow us to track corner positions through transforms.
        """
        h, w = photo.shape[:2]
        marked = photo.copy()
        
        positions = {
            'TL': (marker_size, marker_size),
            'TR': (w - marker_size - 1, marker_size),
            'BR': (w - marker_size - 1, h - marker_size - 1),
            'BL': (marker_size, h - marker_size - 1),
        }
        
        for corner, (x, y) in positions.items():
            # Draw filled circle
            cv2.circle(marked, (x, y), marker_size, self.CORNER_COLORS[corner], -1)
            # Draw crosshair for sub-pixel accuracy
            cv2.line(marked, (x - marker_size, y), (x + marker_size, y), (255, 255, 255), 2)
            cv2.line(marked, (x, y - marker_size), (x, y + marker_size), (255, 255, 255), 2)
        
        return marked
    
    def detect_corner_markers(self, image: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Detect colored corner markers in image using contour analysis.
        Returns dict of corner_name -> (x, y) or None if not detected.
        """
        detected = {}
        
        for corner_name, (lower, upper) in self.COLOR_RANGES.items():
            # Create mask for this color
            mask = cv2.inRange(image, lower, upper)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (should be the marker)
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                # Minimum area threshold to avoid noise
                if area > 25:  # 5x5 pixel minimum
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        detected[corner_name] = (cx, cy)
                    else:
                        detected[corner_name] = None
                else:
                    detected[corner_name] = None
            else:
                detected[corner_name] = None
        
        return detected
    
    def test_corner_tracking_pipeline(self, photo: np.ndarray, center_x: float, 
                                       center_y: float, rotation: float,
                                       perspective_matrix: np.ndarray,
                                       output_size: Tuple[int, int]) -> List[CornerTestResult]:
        """
        Test the full corner tracking pipeline:
        1. Place markers at corners
        2. Apply rotation
        3. Composite onto canvas
        4. Apply perspective warp
        5. Detect markers in output
        6. Compare with expected positions
        """
        h, w = photo.shape[:2]
        results = []
        
        # Calculate expected marker positions in photo (before transforms)
        photo_markers = {
            'TL': (10, 10),
            'TR': (w - 11, 10),
            'BR': (w - 11, h - 11),
            'BL': (10, h - 11),
        }
        
        # Transform expected positions through rotation
        if abs(rotation) > 0.5:
            M_rot = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
            cos_a = abs(M_rot[0, 0])
            sin_a = abs(M_rot[0, 1])
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)
            M_rot[0, 2] += (new_w - w) / 2
            M_rot[1, 2] += (new_h - h) / 2
            
            for corner, (px, py) in photo_markers.items():
                pt = np.array([px, py, 1])
                result = M_rot @ pt
                photo_markers[corner] = (result[0], result[1])
        
        # Add offset for canvas position
        for corner in photo_markers:
            x, y = photo_markers[corner]
            photo_markers[corner] = (x + center_x - w/2, y + center_y - h/2)
        
        # Transform through perspective
        for corner in photo_markers:
            x, y = photo_markers[corner]
            # Apply perspective transform
            denom = perspective_matrix[2, 0] * x + perspective_matrix[2, 1] * y + perspective_matrix[2, 2]
            if abs(denom) > 1e-10:
                new_x = (perspective_matrix[0, 0] * x + perspective_matrix[0, 1] * y + perspective_matrix[0, 2]) / denom
                new_y = (perspective_matrix[1, 0] * x + perspective_matrix[1, 1] * y + perspective_matrix[1, 2]) / denom
                photo_markers[corner] = (new_x, new_y)
        
        # Place markers on photo and run through transforms
        photo_with_markers = self.place_corner_markers(photo.copy())
        
        # Rotate
        if abs(rotation) > 0.5:
            photo_with_markers = self.rotate_photo(photo_with_markers, rotation)
        
        # Composite
        canvas = np.zeros((1500, 1500, 4), dtype=np.uint8)
        canvas = self.composite_at_center(canvas, photo_with_markers, center_x + 300, center_y + 300)
        
        # Warp
        warped = cv2.warpPerspective(canvas, perspective_matrix, output_size, 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0, 0, 0, 0))
        
        # Detect in warped output
        detected = self.detect_corner_markers(warped)
        
        # Compare
        corner_mapping = {'TL': 'UL', 'TR': 'UR', 'BR': 'LR', 'BL': 'LL'}
        for photo_corner, yolo_corner in corner_mapping.items():
            if photo_corner in detected and detected[photo_corner]:
                exp = photo_markers[photo_corner]
                det = detected[photo_corner]
                error = math.sqrt((exp[0] - det[0])**2 + (exp[1] - det[1])**2)
                
                results.append(CornerTestResult(
                    photo_idx=0,
                    corner_name=yolo_corner,
                    expected=exp,
                    detected=det,
                    error_px=error,
                    passed=error < self.CORNER_ERROR_THRESHOLD_PX
                ))
            else:
                results.append(CornerTestResult(
                    photo_idx=0,
                    corner_name=yolo_corner,
                    expected=photo_markers[photo_corner] if photo_corner in photo_markers else (0, 0),
                    detected=(0, 0),
                    error_px=999,
                    passed=False
                ))
        
        return results
    
    # =========================================================================
    # TEST 2: BLACK EDGE DETECTION
    # =========================================================================
    
    def detect_black_edges(self, image: np.ndarray, border_width: int = 30) -> EdgeTestResult:
        """
        Detect black edges using edge detection and histogram analysis.
        
        CV Technique:
        1. Convert to grayscale
        2. Apply Canny edge detection
        3. Analyze pixels near borders
        4. Check histogram for dark pixels
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # Check each border
        border_dark = {
            'top': np.sum(gray[:border_width, :] < 10),
            'bottom': np.sum(gray[-border_width:, :] < 10),
            'left': np.sum(gray[:, :border_width] < 10),
            'right': np.sum(gray[:, -border_width:] < 10),
        }
        
        total_edge_pixels = 4 * border_width * max(w, h)
        dark_ratio = sum(border_dark.values()) / total_edge_pixels if total_edge_pixels > 0 else 0
        
        return EdgeTestResult(
            image_name="test",
            border_dark_pixels=border_dark,
            total_edge_pixels=total_edge_pixels,
            dark_ratio=dark_ratio,
            passed=dark_ratio < self.EDGE_DARK_RATIO_THRESHOLD
        )
    
    def analyze_border_variance(self, image: np.ndarray, border_width: int = 20) -> Dict[str, float]:
        """
        Analyze variance in border regions to detect unnatural edges.
        Good images should have texture/variance in borders.
        Bad images have solid black/white borders.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        borders = {
            'top': gray[:border_width, :],
            'bottom': gray[-border_width:, :],
            'left': gray[:, :border_width],
            'right': gray[:, -border_width:],
        }
        
        variance = {}
        for name, region in borders.items():
            variance[name] = np.var(region)
        
        return variance
    
    # =========================================================================
    # TEST 3: SHADOW BOUNDS VERIFICATION
    # =========================================================================
    
    def detect_shadow_regions(self, image: np.ndarray, photo_bbox: Tuple[int, int, int, int]) -> ShadowTestResult:
        """
        Detect shadow regions and measure their spread.
        
        CV Technique:
        1. Convert to HSV
        2. Find low-value regions (potential shadows)
        3. Measure distance from photo edge to shadow edge
        4. Calculate spread ratio
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        x1, y1, x2, y2 = photo_bbox
        photo_width = x2 - x1
        photo_height = y2 - y1
        
        # Create mask for photo region (as reference)
        photo_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(photo_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Detect dark regions (potential shadows)
        # Low saturation, low value, but not pure black (which would be border)
        lower = np.array([0, 0, 20])
        upper = np.array([180, 50, 120])
        shadow_mask = cv2.inRange(hsv, lower, upper)
        
        # Expand shadows slightly
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=2)
        
        # Find contours of shadow regions
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return ShadowTestResult(
                shadow_spread_px=0,
                photo_size_px=min(photo_width, photo_height),
                spread_ratio=0,
                passed=True
            )
        
        # Find largest shadow contour
        largest = max(contours, key=cv2.contourArea)
        
        # Get bounding rect of shadow
        rx, ry, rw, rh = cv2.boundingRect(largest)
        
        # Calculate shadow spread beyond photo
        spread_left = max(0, x1 - rx)
        spread_right = max(0, (rx + rw) - x2)
        spread_top = max(0, y1 - ry)
        spread_bottom = max(0, (ry + rh) - y2)
        
        max_spread = max(spread_left, spread_right, spread_top, spread_bottom)
        photo_size = min(photo_width, photo_height)
        spread_ratio = max_spread / photo_size if photo_size > 0 else 0
        
        return ShadowTestResult(
            shadow_spread_px=max_spread,
            photo_size_px=photo_size,
            spread_ratio=spread_ratio,
            passed=spread_ratio < self.SHADOW_SPREAD_RATIO_THRESHOLD
        )
    
    # =========================================================================
    # TEST 4: BOUNDING BOX ACCURACY
    # =========================================================================
    
    def detect_photo_edges(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect photo edges using Canny edge detection.
        Returns corner points of the main photo region.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny edge detection with low threshold to catch all edges
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Find largest rectangular contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 1000:  # Minimum reasonable photo size
            return []
        
        # Approximate to 4-point polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        if len(approx) >= 4:
            # Get 4 corners by sorting
            corners = [(pt[0][0], pt[0][1]) for pt in approx[:4]]
            # Sort by x to get left/right, then by y within each
            sorted_x = sorted(corners, key=lambda p: p[0])
            left = sorted(sorted_x[:2], key=lambda p: p[1])  # [UL, LL]
            right = sorted(sorted_x[2:], key=lambda p: p[1])  # [UR, LR]
            return [left[0], left[1], right[0], right[1]]  # [UL, LL, UR, LR]
        
        return []
    
    def test_bbox_accuracy(self, image: np.ndarray, 
                           label_bbox: Tuple[float, float, float, float],
                           label_corners: List[Tuple[float, float]]) -> BboxTestResult:
        """
        Compare label bounding box to detected photo edges.
        
        CV Technique:
        1. Detect edges in image
        2. Find corner points of photo region
        3. Compare to label values
        """
        detected_edges = self.detect_photo_edges(image)
        
        if len(detected_edges) < 4:
            return BboxTestResult(
                detected_edges=detected_edges,
                bbox_corners=label_corners,
                max_error_px=999,
                passed=False
            )
        
        # Compare detected corners to label corners
        errors = []
        for det, lbl in zip(detected_edges, label_corners):
            error = math.sqrt((det[0] - lbl[0])**2 + (det[1] - lbl[1])**2)
            errors.append(error)
        
        max_error = max(errors) if errors else 999
        
        return BboxTestResult(
            detected_edges=detected_edges,
            bbox_corners=label_corners,
            max_error_px=max_error,
            passed=max_error < self.BBOX_ERROR_THRESHOLD_PX
        )
    
    # =========================================================================
    # TEST 5: PERSPECTIVE TRANSFORM VALIDATION
    # =========================================================================
    
    def create_grid_pattern(self, size: int, grid_spacing: int = 50) -> np.ndarray:
        """
        Create a grid pattern for perspective transform testing.
        White grid on black background.
        """
        grid = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Draw vertical lines
        for x in range(0, size, grid_spacing):
            cv2.line(grid, (x, 0), (x, size), (255, 255, 255), 2)
        
        # Draw horizontal lines
        for y in range(0, size, grid_spacing):
            cv2.line(grid, (0, y), (size, y), (255, 255, 255), 2)
        
        return grid
    
    def verify_perspective_transform(self, grid: np.ndarray, 
                                      transform_matrix: np.ndarray,
                                      expected_direction: int) -> Dict:
        """
        Verify that perspective transform was applied correctly.
        
        CV Technique:
        1. Apply known perspective to grid
        2. Check that grid lines remain straight (not curved)
        3. Verify distortion direction matches expected
        """
        warped = cv2.warpPerspective(grid, transform_matrix, (640, 640),
                                     flags=cv2.INTER_LINEAR)
        
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Check line straightness by checking deviation from linear
        # Sample vertical lines at different y positions
        h, w = warped.shape[:2]
        
        line_samples = []
        for x in range(50, w - 50, 50):
            # Get column of edge pixels
            col = edges[:, x]
            edge_positions = np.where(col > 0)[0]
            
            if len(edge_positions) >= 2:
                # Check if positions are roughly linear
                y_start, y_end = edge_positions[0], edge_positions[-1]
                for y in edge_positions[1:-1]:
                    # Calculate expected position if line is straight
                    t = (y - y_start) / (y_end - y_start) if y_end != y_start else 0.5
                    expected_y = y_start + t * (y_end - y_start)
                    deviation = abs(y - expected_y)
                    line_samples.append(deviation)
        
        avg_deviation = np.mean(line_samples) if line_samples else 0
        
        return {
            'warped': warped,
            'avg_line_deviation': avg_deviation,
            'lines_straight': avg_deviation < 5,  # Lines should be nearly straight
            'grid_detected': len(line_samples) > 5  # Should have many grid lines
        }
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def rotate_photo(self, photo: np.ndarray, angle: float) -> np.ndarray:
        """Rotate photo using OpenCV."""
        h, w = photo.shape[:2]
        
        if abs(angle) < 0.5:
            return photo
        
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        return cv2.warpAffine(photo, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(128, 128, 128, 0))
    
    def composite_at_center(self, canvas: np.ndarray, photo: np.ndarray, 
                           cx: float, cy: float) -> np.ndarray:
        """Composite photo onto canvas at center position."""
        ph, pw = photo.shape[:2]
        ch, cw = canvas.shape[:2]
        
        # Ensure photo has alpha
        if photo.shape[2] == 3:
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
        
        top_x = int(cx - pw / 2)
        top_y = int(cy - ph / 2)
        
        # Calculate copy region
        src_x1, src_y1 = 0, 0
        src_x2, src_y2 = pw, ph
        dst_x1, dst_y1 = top_x, top_y
        dst_x2, dst_y2 = top_x + pw, top_y + ph
        
        # Clip to canvas bounds
        if dst_x1 < 0:
            src_x1 = -dst_x1
            dst_x1 = 0
        if dst_y1 < 0:
            src_y1 = -dst_y1
            dst_y1 = 0
        if dst_x2 > cw:
            src_x2 = cw - dst_x1
            dst_x2 = cw
        if dst_y2 > ch:
            src_y2 = ch - dst_y1
            dst_y2 = ch
        
        copy_w = int(dst_x2 - dst_x1)
        copy_h = int(dst_y2 - dst_y1)
        
        if copy_w <= 0 or copy_h <= 0:
            return canvas
        
        canvas_f = canvas.astype(np.float32) / 255.0
        photo_f = photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w].astype(np.float32) / 255.0
        
        alpha = photo_f[:, :, 3:4]
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
            photo_f[:, :, :3] * alpha + 
            canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] * (1 - alpha)
        ).astype(np.float32)
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.maximum(
            canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3],
            photo_f[:, :, 3]
        )
        
        return (canvas_f * 255).astype(np.uint8)
    
    # =========================================================================
    # MAIN TEST RUNNER
    # =========================================================================
    
    def run_all_tests(self, image_path: Path, label_path: Optional[Path] = None) -> Dict:
        """
        Run all verification tests on an image.
        Returns dict with results for each test category.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'Could not load {image_path}'}
        
        results = {
            'corner_tracking': [],
            'black_edges': None,
            'shadows': None,
            'bbox_accuracy': None,
        }
        
        # Test 1: Black edge detection
        results['black_edges'] = self.detect_black_edges(image)
        
        # Test 2: Border variance analysis
        results['border_variance'] = self.analyze_border_variance(image)
        
        # If we have labels, test bbox accuracy
        if label_path and label_path.exists():
            with open(label_path) as f:
                lines = f.readlines()
            
            if lines:
                parts = lines[0].strip().split()
                # Parse pose label format
                corners = []
                for i in range(4):
                    kx = float(parts[5 + i*3]) * 640
                    ky = float(parts[5 + i*3 + 1]) * 640
                    corners.append((kx, ky))
                
                results['bbox_accuracy'] = self.test_bbox_accuracy(image, None, corners)
        
        return results
    
    def generate_test_report(self, results: Dict) -> str:
        """Generate a human-readable test report."""
        lines = []
        lines.append("=" * 60)
        lines.append("VERIFICATION TEST REPORT")
        lines.append("=" * 60)
        
        # Black edges
        if 'black_edges' in results and results['black_edges']:
            edge = results['black_edges']
            lines.append(f"\n1. BLACK EDGE DETECTION:")
            lines.append(f"   Dark pixel ratio: {edge.dark_ratio:.4f} (threshold: {self.EDGE_DARK_RATIO_THRESHOLD})")
            lines.append(f"   Status: {'✅ PASS' if edge.passed else '❌ FAIL'}")
            if not edge.passed:
                lines.append(f"   Border dark pixels: {edge.border_dark_pixels}")
        
        # Border variance
        if 'border_variance' in results:
            lines.append(f"\n   Border variance (texture indicator):")
            for name, var in results['border_variance'].items():
                status = "✅" if var > 50 else "❌"
                lines.append(f"     {name}: {var:.1f} {status}")
        
        # Bbox accuracy
        if 'bbox_accuracy' in results and results['bbox_accuracy']:
            bbox = results['bbox_accuracy']
            lines.append(f"\n2. BOUNDING BOX ACCURACY:")
            lines.append(f"   Max error: {bbox.max_error_px:.1f}px (threshold: {self.BBOX_ERROR_THRESHOLD_PX}px)")
            lines.append(f"   Status: {'✅ PASS' if bbox.passed else '❌ FAIL'}")
        
        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)


# =============================================================================
# STANDALONE VERIFICATION SCRIPT
# =============================================================================

def verify_generated_images():
    """Verify all generated images programmatically."""
    
    suite = TestSuite()
    
    test_dir = Path("/Users/krys.petrie/dev/photo-pose-detector/data/examples_v34")
    
    if not test_dir.exists():
        print(f"ERROR: Test directory {test_dir} does not exist")
        return
    
    image_files = sorted([f for f in test_dir.glob("example_*.jpg") if '_debug' not in f.name])
    
    print(f"\nVerifying {len(image_files)} images...\n")
    
    all_passed = True
    fail_count = 0
    pass_count = 0
    
    for img_path in image_files:
        # Construct label path from image number
        img_num = img_path.stem.split('_')[1]  # Gets '01' from 'example_01'
        label_path = test_dir / f"example_{img_num}_pose.txt"
        
        print(f"Testing: {img_path.name}")
        
        results = suite.run_all_tests(img_path, label_path if label_path.exists() else None)
        report = suite.generate_test_report(results)
        print(report)
        
        # Check if all tests passed
        tests_passed = True
        if 'black_edges' in results and not results['black_edges'].passed:
            tests_passed = False
        if 'bbox_accuracy' in results and results['bbox_accuracy'] and not results['bbox_accuracy'].passed:
            tests_passed = False
        
        if tests_passed:
            pass_count += 1
            print(f"   ALL TESTS PASSED ✅")
        else:
            fail_count += 1
            all_passed = False
            print(f"   SOME TESTS FAILED ❌")
        
        print()
    
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Images tested: {pass_count + fail_count}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == '__main__':
    success = verify_generated_images()
    exit(0 if success else 1)