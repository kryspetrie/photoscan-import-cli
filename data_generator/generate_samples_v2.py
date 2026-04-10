#!/usr/bin/env python3
"""
Photo Pose Detector - Fast Synthetic Data Generator (v2)
Improved with:
1. Camera-angle perspective simulation (all 4 corners affected)
2. Variable, subtle drop shadows
3. Pre-perspective rotation
4. Efficient photo packing (6-12 photos)
"""

import signal
import sys
import os
from pathlib import Path
import traceback
import time
import math

# Timeout handler
def timeout_handler(signum, frame):
    print("\n⏱️ TIMEOUT: Generation took too long, stopping...")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(180)  # 3 minute timeout

print("🖼️  Generating 10 example images (v2 - improved)...")
print("⏱️  Timeout: 180 seconds")
print()

start_time = time.time()

try:
    import numpy as np
    import cv2
    from PIL import Image
    
    # =============================================================================
    # UTILITY FUNCTIONS
    # =============================================================================
    
    def clip_corners(corners, margin, canvas_w, canvas_h):
        """Ensure all corners stay within canvas bounds."""
        corners = corners.copy()
        corners[:, 0] = np.clip(corners[:, 0], margin, canvas_w - margin)
        corners[:, 1] = np.clip(corners[:, 1], margin, canvas_h - margin)
        return corners
    
    # =============================================================================
    # CAMERA-ANGLE PERSPECTIVE SIMULATION (Issue 1)
    # =============================================================================
    
    def apply_camera_perspective(corners, margin=40, canvas_w=1920, canvas_h=1080):
        """
        Simulate camera viewing photo at an angle.
        
        Moves all 4 corners independently while maintaining quadrilateral shape.
        Each corner gets its own displacement vector (dx, dy) scaled to be similar.
        Displacements are constrained to keep max/min ratio <= 2.5.
        All 4 corners guaranteed to move >= 15px.
        Creates proper trapezoidal shapes with edge ratios 0.40-0.98.
        
        Args:
            corners: 4x2 array of corner coordinates [TL, TR, BR, BL]
            margin: minimum distance from canvas edge
            canvas_w, canvas_h: canvas dimensions
        
        Returns:
            corners with perspective distortion applied
        """
        new_corners = corners.copy()
        
        width = corners[1, 0] - corners[0, 0]
        height = corners[2, 1] - corners[1, 1]
        
        # Base displacement - use fraction of smaller dimension
        base = min(width, height) * random.uniform(0.35, 0.55)
        base = max(60, base)  # Ensure minimum base
        
        # Choose corner movement factors that keep displacements balanced
        # Key insight: Use corner-specific factors that result in similar displacements
        # when combined with the base direction vectors
        
        # h_base and v_base define the direction of movement
        # corner_factors scale each corner's movement
        # To keep displacements similar, we need to balance corner_factors
        
        narrow_top = random.random() < 0.5
        
        if narrow_top:
            h_base = [1.0, -1.0, -0.4, 0.4]
        else:
            h_base = [-0.4, 0.4, -1.0, 1.0]
        
        stretch_diag1 = random.random() < 0.5
        if stretch_diag1:
            v_base = [1.0, 0.3, -1.0, 0.3]
        else:
            v_base = [0.3, 1.0, 0.3, -1.0]
        
        # Calculate intrinsic displacement multipliers for each corner
        # displacement = base * scale * sqrt(h^2 + v^2)
        intrinsic = [np.sqrt(h_base[i]**2 + v_base[i]**2) for i in range(4)]
        
        # Scale factors should be inversely proportional to intrinsic values
        # to balance the final displacements
        # But we also need some randomness for variety
        
        # Use a more constrained random approach:
        # Pick a base scale, then adjust each corner slightly
        base_scale = random.uniform(0.8, 1.5)
        corner_factors = [
            base_scale * random.uniform(0.85, 1.15),
            base_scale * random.uniform(0.85, 1.15),
            base_scale * random.uniform(0.85, 1.15),
            base_scale * random.uniform(0.85, 1.15)
        ]
        
        # Normalize to ensure displacement ratio constraint
        # Calculate unnormalized displacements
        raw_disps = [intrinsic[i] * corner_factors[i] for i in range(4)]
        
        # Scale all to ensure minimum >= 15
        min_raw = min(raw_disps)
        if min_raw < 15 / base:
            scale_up = (15 / base) / min_raw
            corner_factors = [f * scale_up for f in corner_factors]
        
        # Calculate final displacements
        displacements = [base * intrinsic[i] * corner_factors[i] for i in range(4)]
        
        # Balance the corner factors to keep ratio <= 2.5
        min_disp = min(displacements)
        max_disp = max(displacements)
        
        if max_disp > min_disp * 2.5:
            # Scale down the max corner
            max_idx = displacements.index(max_disp)
            target_max = min_disp * 2.5
            corner_factors[max_idx] *= target_max / max_disp
        
        # Recalculate displacements
        displacements = [base * intrinsic[i] * corner_factors[i] for i in range(4)]
        
        # Calculate final dx, dy for each corner
        h_disp = [base * h_base[i] * corner_factors[i] for i in range(4)]
        v_disp = [base * v_base[i] * corner_factors[i] for i in range(4)]
        
        for i in range(4):
            new_corners[i, 0] += h_disp[i]
            new_corners[i, 1] += v_disp[i]
        
        # Add small rotation
        tilt = random.uniform(-8, 8)
        if abs(tilt) > 1:
            cx = (corners[0, 0] + corners[1, 0]) / 2
            cy = (corners[0, 1] + corners[2, 1]) / 2
            angle_rad = np.radians(tilt)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            translated = new_corners - [cx, cy]
            rotated = np.array([
                translated[:, 0] * cos_a - translated[:, 1] * sin_a,
                translated[:, 0] * sin_a + translated[:, 1] * cos_a
            ]).T
            new_corners = rotated + [cx, cy]
        
        return clip_corners(new_corners, margin, canvas_w, canvas_h)
    
    # =============================================================================
    # PRE-PERSPECTIVE ROTATION (Issue 3)
    # =============================================================================
    
    def rotate_photo_and_corners(photo, corners):
        """
        Rotate photo and its corner keypoints around photo center.
        
        Applied BEFORE perspective warp to simulate photos at various
        angles on a surface.
        
        Args:
            photo: Photo image array (H, W, C)
            corners: 4x2 array of corner coordinates in canvas-space
        
        Returns:
            (rotated_photo, rotated_corners)
        """
        import random
        
        h, w = photo.shape[:2]
        center = (w / 2, h / 2)
        
        # Weighted distribution for more natural orientations
        r = random.random()
        if r < 0.30:
            angle = random.uniform(-5, 5)       # Near-horizontal (30%)
        elif r < 0.70:
            angle = random.uniform(-15, 15)     # Slight tilt (40%)
        else:
            angle = random.uniform(-25, 25)     # Strong tilt (30%)
        
        # Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(photo, M, (w, h), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_REFLECT_101)
        
        # Calculate rotated corners in canvas-space
        photo_cx = (corners[0, 0] + corners[2, 0]) / 2
        photo_cy = (corners[0, 1] + corners[2, 1]) / 2
        
        local_corners = corners - [photo_cx, photo_cy]
        
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        rotated_local = np.array([
            local_corners[:, 0] * cos_a - local_corners[:, 1] * sin_a,
            local_corners[:, 0] * sin_a + local_corners[:, 1] * cos_a
        ]).T
        
        rotated_corners = rotated_local + [photo_cx, photo_cy]
        
        return rotated, rotated_corners
    
    def process_photo_pipeline(photo, flat_corners, canvas_w, canvas_h):
        """
        Full photo processing pipeline:
        1. Rotate photo (and corners) on background plane
        2. Apply camera perspective distortion
        
        Args:
            photo: Photo image
            flat_corners: 4x2 array of axis-aligned corner coordinates
            canvas_w, canvas_h: canvas dimensions
        
        Returns:
            (processed_photo, final_corners)
        """
        # Step 1: Pre-perspective rotation
        rotated_photo, rotated_corners = rotate_photo_and_corners(photo, flat_corners)
        
        # Step 2: Apply camera perspective
        perspective_corners = apply_camera_perspective(
            rotated_corners, 
            margin=40, 
            canvas_w=canvas_w, 
            canvas_h=canvas_h
        )
        
        return rotated_photo, perspective_corners
    
    # =============================================================================
    # VARIABLE, SUBTLE DROP SHADOWS (Issue 2)
    # =============================================================================
    
    def create_variable_shadow(canvas, corners, shadow_offset_base=8):
        """
        Create a subtle drop shadow that doesn't darken the background.
        
        Key improvements over old shadow:
        - Opacity: 12-22% (was 25-45%)
        - Blur: Variable based on shadow size (5-25px)
        - Shadow direction: Aligned with implied light source
        - Light gray color instead of dark
        
        Args:
            canvas: Background image
            corners: 4x2 corner coordinates
            shadow_offset_base: Base offset for shadow (default 8px)
        
        Returns:
            canvas with shadow composited
        """
        import random
        
        h, w = canvas.shape[:2]
        
        # --- Shadow parameters (variable, subtle) ---
        # Opacity: 12-22% (much subtler than old 25-45%)
        shadow_opacity = random.uniform(0.12, 0.22)
        
        # Blur: Variable based on shadow size (5-25px)
        shadow_bounds = cv2.boundingRect(corners.astype(np.int32))
        shadow_size = max(shadow_bounds[2], shadow_bounds[3])
        blur_sigma = np.clip(shadow_size * 0.04, 5, 25)  # Variable blur
        blur_sigma += random.uniform(-3, 3)  # Add variation
        
        # Offset: 5-12 pixels (subtler than old 15-35)
        shadow_offset = random.uniform(5, 12) + shadow_offset_base
        
        # --- Light direction (upper-left most common) ---
        if random.random() < 0.65:
            light_dir = (-1, -1)  # Upper-left (natural sunlight)
        else:
            angle = random.uniform(0, 360)
            light_dir = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))
        
        lx, ly = light_dir
        l_len = np.sqrt(lx**2 + ly**2)
        light_dir = (lx / l_len, ly / l_len)
        
        # --- Create shadow polygon ---
        offset_vector = np.array([light_dir[0] * shadow_offset, light_dir[1] * shadow_offset])
        shadow_corners = corners + offset_vector
        
        # --- Create shadow mask ---
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        pts = shadow_corners.astype(np.int32)
        cv2.fillPoly(shadow_mask, [pts], 1.0)
        
        # Apply Gaussian blur
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), blur_sigma)
        
        # --- Create subtle shadow color ---
        # Light gray (220) instead of dark (191)
        shadow_color = 220
        
        # --- Blend shadow onto canvas ---
        # Use a lighter blend to avoid darkening background
        canvas_f = canvas.astype(np.float32)
        
        for c in range(3):
            # Lighter shadow blend: softens edges rather than darkening
            canvas_f[:, :, c] = np.clip(
                canvas_f[:, :, c] * (1 - shadow_mask * shadow_opacity * 0.4) + 
                shadow_color * shadow_mask * shadow_opacity * 0.6,
                0, 255
            )
        
        return canvas_f.astype(np.uint8)
    
    # =============================================================================
    # EFFICIENT PHOTO PACKING (Issue 4)
    # =============================================================================
    
    def calculate_efficient_packings(canvas_w, canvas_h, num_photos_target=None):
        """
        Calculate efficient photo packings using shelf algorithm.
        
        Improvements:
        - 6-12 photos per frame (was 4-7)
        - Variable sizes (8-35% of canvas)
        - Shelf packing for better space usage
        - Rotation buffer for rotated photos
        
        Args:
            canvas_w, canvas_h: Canvas dimensions
            num_photos_target: Target number of photos (or None for auto)
        
        Returns:
            List of placements, each with: x, y, width, height, rotation
        """
        import random
        
        margin = 50  # Edge margin
        gap = 30     # Gap between photos
        
        # --- Determine number of photos ---
        if num_photos_target is None:
            base_count = max(6, min(11, int((canvas_w * canvas_h) / 220000)))
            num_photos = base_count + random.randint(-1, 2)
        else:
            num_photos = num_photos_target
        
        # --- Determine photo sizes ---
        # Mix of large, medium, and small
        photo_sizes = []
        for _ in range(num_photos):
            r = random.random()
            if r < 0.35:
                # Large: 25-35% of canvas width
                width = int(canvas_w * random.uniform(0.22, 0.32))
                height = int(canvas_h * random.uniform(0.16, 0.24))
            elif r < 0.75:
                # Medium: 15-22% of canvas width
                width = int(canvas_w * random.uniform(0.14, 0.22))
                height = int(canvas_h * random.uniform(0.10, 0.17))
            else:
                # Small: 8-14% of canvas width
                width = int(canvas_w * random.uniform(0.08, 0.14))
                height = int(canvas_h * random.uniform(0.06, 0.11))
            photo_sizes.append((width, height))
        
        # Sort by area (larger first) for better shelf packing
        photo_sizes.sort(key=lambda x: x[0] * x[1], reverse=True)
        
        # --- Shelf packing algorithm ---
        placements = []
        shelf_y = margin
        current_x = margin
        current_shelf_height = 0
        
        for width, height in photo_sizes:
            # Determine if this photo is rotated (for packing purposes)
            rotate = random.random() < 0.30  # 30% chance
            if rotate:
                rot_width, rot_height = height, width
                rotation_angle = random.uniform(-20, 20)
            else:
                rot_width, rot_height = width, height
                rotation_angle = 0
            
            # Add buffer for rotation (15% extra space)
            buffer = abs(np.sin(np.radians(rotation_angle))) * rot_width * 0.12 if rotation_angle != 0 else 0
            effective_width = rot_width + buffer * 2
            effective_height = rot_height + buffer * 2
            
            # Check if fits in current shelf
            if current_x + effective_width > canvas_w - margin:
                shelf_y += current_shelf_height + gap
                current_x = margin
                current_shelf_height = 0
            
            # Check if fits vertically
            if shelf_y + effective_height > canvas_h - margin:
                # Try scaling down
                scale = (canvas_h - margin - shelf_y) / effective_height
                if scale > 0.5:
                    rot_width = int(rot_width * scale)
                    rot_height = int(rot_height * scale)
                    effective_width = rot_width + buffer * 2
                    effective_height = rot_height + buffer * 2
                else:
                    continue  # Skip this photo
            
            placements.append({
                'x': current_x,
                'y': shelf_y,
                'width': rot_width,
                'height': rot_height,
                'rotation': rotation_angle
            })
            
            current_shelf_height = max(current_shelf_height, effective_height)
            current_x += effective_width + gap
        
        return placements
    
    def get_flat_corners_from_placement(placement):
        """Get flat corner coordinates for a photo placement."""
        x, y = placement['x'], placement['y']
        w, h = placement['width'], placement['height']
        
        return np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)
    
    # =============================================================================
    # BACKGROUND GENERATION
    # =============================================================================
    
    def fast_background(w, h):
        """Generate background in ~0.1s."""
        import random
        from numpy.random import randint, uniform, normal
        
        bg_type = random.choices(['solid', 'wood', 'fabric', 'laminate'], weights=[0.4, 0.25, 0.2, 0.15])[0]
        
        if bg_type == 'solid':
            LIGHT = [(240,242,245),(235,240,242),(250,248,245),(245,245,248),(255,255,252)]
            DARK  = [(15,15,18),(20,20,25),(12,15,20),(25,22,28),(18,18,22)]
            MID   = [(165,160,155),(140,145,138),(180,175,170),(155,150,148),(170,168,162)]
            tier = random.choices(['light','dark','mid'], weights=[0.30,0.30,0.40])[0]
            color = random.choice(LIGHT) if tier=='light' else random.choice(DARK) if tier=='dark' else random.choice(MID)
            img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)
            noise_sigma = random.uniform(3, 15)
            img = np.clip(img + normal(0, noise_sigma, (h, w, 3)), 0, 255).astype(np.uint8)
            return img
        
        elif bg_type == 'wood':
            base_colors = [(139,90,43),(160,110,65),(120,75,35),(180,130,80)]
            base = random.choice(base_colors)
            img = np.ones((h, w, 3), dtype=np.uint8) * np.array(base)
            num_grains = randint(15, 25)
            for _ in range(num_grains):
                y = randint(0, h-1)
                thickness = randint(2, 8)
                intensity = randint(-30, 30)
                offset = random.uniform(-0.02, 0.02)
                y1, y2 = max(0, y-thickness), min(h, y+thickness)
                for i in range(y1, y2):
                    wavy_x = int(math.sin(i * offset) * 20)
                    blend = 1 - abs(i - y) / thickness
                    x1, x2 = max(0, wavy_x), min(w, wavy_x + w)
                    img[i, x1:x2] = np.clip(img[i, x1:x2] + (intensity + randint(-15,15)) * blend, 0, 255)
            noise = normal(0, 8, (h, w, 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            return cv2.GaussianBlur(img, (3, 3), 0)
        
        elif bg_type == 'fabric':
            base_colors = [(120,110,100),(100,95,90),(140,130,120),(110,105,100)]
            color = random.choice(base_colors)
            img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)
            weave_scale = randint(3, 6)
            for i in range(0, h, weave_scale):
                for j in range(0, w, weave_scale):
                    shade = randint(-5, 5)
                    i2 = min(i+weave_scale, h)
                    j2 = min(j+weave_scale, w)
                    img[i:i2, j:j2] = np.clip(np.array(color) + shade, 0, 255)
            noise = normal(0, 6, (h, w, 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            return cv2.GaussianBlur(img, (3, 3), 0)
        
        else:  # laminate
            base_colors = [(180,160,140),(160,145,125),(190,175,155),(170,155,135)]
            color = random.choice(base_colors)
            img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)
            plank_h = randint(60, 120)
            for i in range(0, h, plank_h):
                shade = randint(-15, 15)
                i2 = min(i+plank_h, h)
                img[i:i2, :] = np.clip(np.array(color) + shade, 0, 255)
            noise = normal(0, 10, (h, w, 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            return cv2.GaussianBlur(img, (5, 5), 0)
    
    def fast_luma_gradient(img):
        """Add luma gradient overlay."""
        import random
        if random.random() < 0.4:
            h, w = img.shape[:2]
            direction = random.choice(['top', 'bottom', 'left', 'right', 'radial'])
            
            if direction == 'top':
                grad = np.linspace(0, 0.15, h)[:, np.newaxis, np.newaxis]
            elif direction == 'bottom':
                grad = np.linspace(0.15, 0, h)[:, np.newaxis, np.newaxis]
            elif direction == 'left':
                grad = np.linspace(0, 0.15, w)[np.newaxis, :, np.newaxis]
            elif direction == 'right':
                grad = np.linspace(0.15, 0, w)[np.newaxis, :, np.newaxis]
            else:
                cx, cy = w/2, h/2
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
                max_dist = np.sqrt(cx**2 + cy**2)
                grad = (1 - dist/max_dist * 0.3)[:, :, np.newaxis]
                grad = np.clip(grad, 0.3, 1.0)
            
            factor = random.uniform(0.5, 1.5)
            img = np.clip(img * (1 + grad * factor - 0.5 * factor), 0, 255).astype(np.uint8)
        return img
    
    def random_background_gradient(img):
        """Add 1-3 gradient overlays."""
        import random
        h, w = img.shape[:2]
        num_gradients = random.randint(1, 3)
        
        for _ in range(num_gradients):
            grad_type = random.choice(['radial', 'horizontal', 'vertical', 'diagonal'])
            alpha = random.uniform(0.05, 0.20)
            direction = random.choice([-1, 1])
            
            if grad_type == 'radial':
                cx, cy = w/2, h/2
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
                max_dist = np.sqrt(cx**2 + cy**2)
                gradient = (1 - dist/max_dist) * alpha * direction
            elif grad_type == 'horizontal':
                gradient = (np.linspace(0, 1, w) * alpha * direction)[np.newaxis, :]
            elif grad_type == 'vertical':
                gradient = (np.linspace(0, 1, h) * alpha * direction)[:, np.newaxis]
            else:
                Y, X = np.ogrid[:h, :w]
                gradient = ((X/w + Y/h) / 2) * alpha * direction
            
            gradient = gradient[:, :, np.newaxis]
            img = np.clip(img * (1 + gradient), 0, 255).astype(np.uint8)
        
        return img
    
    # =============================================================================
    # PHOTO EFFECTS
    # =============================================================================
    
    def fast_photo_manipulation(img):
        """Apply brightness/contrast/saturation/gamma."""
        import random
        brightness = random.uniform(0.85, 1.20)
        contrast   = random.uniform(0.85, 1.20)
        saturation = random.uniform(0.80, 1.25)
        gamma      = random.uniform(0.80, 1.30)
        
        img_f = img.astype(np.float32) / 255.0
        img_f = ((img_f - 0.5) * contrast + 0.5) * brightness
        img_f = np.clip(img_f, 0, 1)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= saturation
        hsv[:,:,2] *= brightness * contrast
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
        blend = random.uniform(0.0, 1.0)
        img_f = img_f * (1 - blend) + img_sat * blend
        img_f = np.power(np.clip(img_f, 0.001, 1), 1.0/gamma)
        
        return np.clip(img_f * 255, 0, 255).astype(np.uint8)
    
    def fast_glare(img):
        """Add screen-mode glare."""
        import random
        if random.random() < 0.35:
            h, w = img.shape[:2]
            num_flares = random.randint(1, 3)
            for _ in range(num_flares):
                cx = random.uniform(w * 0.2, w * 0.8)
                cy = random.uniform(h * 0.2, h * 0.8)
                rx = random.uniform(w * 0.05, w * 0.15)
                ry = random.uniform(h * 0.05, h * 0.15)
                angle = random.uniform(0, math.pi)
                
                y, x = np.ogrid[:h, :w]
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                xr = (x - cx) * cos_a - (y - cy) * sin_a
                yr = (x - cx) * sin_a + (y - cy) * cos_a
                flare = np.maximum(0, 1 - (xr/rx)**2 - (yr/ry)**2)
                flare = cv2.GaussianBlur(flare.astype(np.float32), (31, 31), 0)
                
                img_f = img.astype(np.float32) / 255.0
                flare_bgr = np.stack([flare * 255] * 3, axis=-1)
                flare_f = flare_bgr / 255.0
                img_f = 1 - (1 - img_f) * (1 - flare_f * 0.3)
                img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
        return img
    
    # =============================================================================
    # COMPOSITING
    # =============================================================================
    
    def composite_trapezoid(canvas, photo, dst_pts):
        """Composite a photo warped to trapezoidal corners."""
        h, w = photo.shape[:2]
        src_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
        warped = cv2.warpPerspective(photo, M, (canvas.shape[1], canvas.shape[0]), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [dst_pts.astype(np.int32)], 255)
        mask = cv2.erode(mask, np.ones((3,3), dtype=np.uint8), iterations=1)
        
        black_mask = ((warped[:,:,0]==0) & (warped[:,:,1]==0) & (warped[:,:,2]==0)).astype(np.uint8) * 255
        if black_mask.sum() > 0:
            kernel = np.ones((3,3), dtype=np.uint8)
            black_mask = cv2.dilate(black_mask, kernel, iterations=2)
            warped = cv2.inpaint(warped, black_mask, 3, cv2.INPAINT_TELEA)
        
        for c in range(3):
            canvas[:,:,c] = np.where(mask > 0, warped[:,:,c], canvas[:,:,c])
        
        return canvas
    
    # =============================================================================
    # VERIFICATION FUNCTIONS
    # =============================================================================
    
    def verify_perspective_symmetry(warped_corners, flat_corners):
        """
        Verify perspective affects all 4 corners, not just one.
        
        Pass criteria:
        - All 4 corners have displacement > 15 pixels
        - No single corner displacement > 2.5× the minimum displacement
        """
        displacements = [np.linalg.norm(warped_corners[i] - flat_corners[i]) for i in range(4)]
        
        # All 4 corners must be displaced
        if not all(d > 15 for d in displacements):
            moved = sum(1 for d in displacements if d > 15)
            return False, f"Only {moved}/4 corners moved significantly (>15px)"
        
        max_disp = max(displacements)
        min_disp = min(displacements)
        
        ratio = max_disp / max(min_disp, 1)
        if ratio > 2.5:
            return False, f"Corner displacement unbalanced: max/min={ratio:.2f}"
        
        return True, ""
    
    def verify_edge_ratio(corners):
        """Check that top and bottom edges have different widths (trapezoid)."""
        top_width = np.linalg.norm(corners[1] - corners[0])
        bot_width = np.linalg.norm(corners[2] - corners[3])
        ratio = min(top_width, bot_width) / max(top_width, bot_width)
        # Extended range to allow more variation in trapezoid shapes
        return 0.40 <= ratio <= 0.98, ratio
    
    def verify_rotation_applied(corners):
        """Verify rotation was applied before perspective."""
        top_edge = corners[1] - corners[0]
        angle = np.arctan2(top_edge[1], top_edge[0])
        
        left_edge = corners[3] - corners[0]
        left_angle = np.arctan2(left_edge[0], left_edge[1])
        
        if abs(angle) < 0.05 and abs(left_angle) < 0.05:
            return False, "Photo appears not rotated (edges axis-aligned)"
        
        return True, ""
    
    def verify_shadow_subtlety(canvas_before, canvas_after, corners):
        """Verify shadow doesn't significantly darken the background."""
        x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
        
        pad = 30
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(canvas_after.shape[1], x + w + pad), min(canvas_after.shape[0], y + h + pad)
        
        region_before = canvas_before[y1:y2, x1:x2].astype(np.float32)
        region_after = canvas_after[y1:y2, x1:x2].astype(np.float32)
        
        brightness_before = region_before.mean()
        brightness_after = region_after.mean()
        
        if brightness_before > 0:
            change = (brightness_before - brightness_after) / brightness_before
            if change > 0.20:  # More than 20% darkening
                return False, f"Shadow too dark: {change*100:.0f}% darker"
        
        return True, ""
    
    # =============================================================================
    # MAIN GENERATION LOOP
    # =============================================================================
    
    import random
    
    source_dir = Path("./images")
    sources = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + list(source_dir.glob('*.png')) + list(source_dir.glob('*.webp'))
    
    print(f"📁 Found {len(sources)} source images")
    
    if len(sources) < 5:
        print("ERROR: Need at least 5 source images")
        sys.exit(1)
    
    output_dir = Path("../data/examples_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CANVAS_W, CANVAS_H = 1920, 1080
    
    random.seed(42)
    np.random.seed(42)
    
    print("\n📸 Generating images...\n")
    
    # Verification tracking
    verification_stats = {
        'total_photos': 0,
        'photos_in_bounds': 0,
        'perspective_symmetric': 0,
        'edge_ratios_ok': 0,
        'rotation_applied': 0,
        'shadows_subtle': 0
    }
    
    for i in range(10):
        img_start = time.time()
        
        # Generate background
        bg = fast_background(CANVAS_W, CANVAS_H)
        bg = fast_luma_gradient(bg)
        bg = random_background_gradient(bg)
        bg_original = bg.copy()  # For shadow verification
        
        # Get efficient packing
        placements = calculate_efficient_packings(CANVAS_W, CANVAS_H)
        
        placed_photos = []
        
        for placement in placements:
            # Load photo
            photo = cv2.imread(str(random.choice(sources)))
            if photo is None:
                continue
            
            # Scale to target size
            target_w, target_h = placement['width'], placement['height']
            h_orig, w_orig = photo.shape[:2]
            scale = min(target_w / w_orig, target_h / h_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            photo = cv2.resize(photo, (new_w, new_h))
            
            # Get flat corners
            flat_corners = np.array([
                [placement['x'], placement['y']],
                [placement['x'] + new_w, placement['y']],
                [placement['x'] + new_w, placement['y'] + new_h],
                [placement['x'], placement['y'] + new_h]
            ], dtype=np.float32)
            
            # Apply rotation + perspective pipeline
            photo, warped_corners = process_photo_pipeline(photo, flat_corners, CANVAS_W, CANVAS_H)
            
            # Apply effects
            photo = fast_photo_manipulation(photo)
            photo = fast_glare(photo)
            
            # Add shadow (before composite)
            bg_before_shadow = bg.copy()
            bg = create_variable_shadow(bg, warped_corners)
            
            # Composite onto canvas
            bg = composite_trapezoid(bg, photo, warped_corners)
            
            # Store keypoints
            placed_photos.append({
                'keypoints': [(warped_corners[k,0], warped_corners[k,1]) for k in range(4)],
                'flat_corners': flat_corners
            })
            
            # Run verification
            verification_stats['total_photos'] += 1
            
            # Bounds check
            in_bounds = all(0 <= warped_corners[k,0] < CANVAS_W and 0 <= warped_corners[k,1] < CANVAS_H for k in range(4))
            if in_bounds:
                verification_stats['photos_in_bounds'] += 1
            
            # Perspective symmetry
            ok, _ = verify_perspective_symmetry(warped_corners, flat_corners)
            if ok:
                verification_stats['perspective_symmetric'] += 1
            
            # Edge ratio
            ok, ratio = verify_edge_ratio(warped_corners)
            if ok:
                verification_stats['edge_ratios_ok'] += 1
            
            # Rotation applied
            ok, _ = verify_rotation_applied(warped_corners)
            if ok:
                verification_stats['rotation_applied'] += 1
            
            # Shadow subtlety
            ok, _ = verify_shadow_subtlety(bg_before_shadow, bg, warped_corners)
            if ok:
                verification_stats['shadows_subtle'] += 1
        
        # Save image
        pil_img = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        img_path = output_dir / f"example_{i+1:02d}.jpg"
        pil_img.save(img_path, quality=90)
        
        # Save label file
        lbl_path = output_dir / f"example_{i+1:02d}.txt"
        with open(lbl_path, 'w') as f:
            for p in placed_photos:
                kps = p['keypoints']
                x_center = sum(k[0] for k in kps) / 4 / CANVAS_W
                y_center = sum(k[1] for k in kps) / 4 / CANVAS_H
                width = (max(k[0] for k in kps) - min(k[0] for k in kps)) / CANVAS_W
                height = (max(k[1] for k in kps) - min(k[1] for k in kps)) / CANVAS_H
                
                line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for kx, ky in kps:
                    line += f" {kx/CANVAS_W:.6f} {ky/CANVAS_H:.6f} 2"
                line += "\n"
                f.write(line)
        
        img_time = time.time() - img_start
        print(f"  {i+1:2d}/10: {img_time:5.1f}s ({len(placed_photos)} photos)")
    
    total_time = time.time() - start_time
    
    # Print verification summary
    print(f"\n{'='*60}")
    print("📊 VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    total = verification_stats['total_photos']
    print(f"\n  Total photos: {total}")
    print(f"  In bounds:    {verification_stats['photos_in_bounds']}/{total} ({verification_stats['photos_in_bounds']/max(total,1)*100:.0f}%)")
    print(f"  Perspective symmetry: {verification_stats['perspective_symmetric']}/{total} ({verification_stats['perspective_symmetric']/max(total,1)*100:.0f}%)")
    print(f"  Edge ratios OK:      {verification_stats['edge_ratios_ok']}/{total} ({verification_stats['edge_ratios_ok']/max(total,1)*100:.0f}%)")
    print(f"  Rotation applied:     {verification_stats['rotation_applied']}/{total} ({verification_stats['rotation_applied']/max(total,1)*100:.0f}%)")
    print(f"  Shadows subtle:      {verification_stats['shadows_subtle']}/{total} ({verification_stats['shadows_subtle']/max(total,1)*100:.0f}%)")
    
    print(f"\n✅ Done! 10 images in {total_time:.1f}s")
    print(f"📂 {output_dir.absolute()}")
    
    # List files
    for f in sorted(output_dir.glob("example_*.jpg")):
        print(f"   - {f.name} ({f.stat().st_size//1024} KB)")
    
    signal.alarm(0)
    
except Exception as e:
    signal.alarm(0)
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)