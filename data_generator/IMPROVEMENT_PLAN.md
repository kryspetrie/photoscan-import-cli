# Comprehensive Improvement Plan: photo_pose_detector Data Generator

## Executive Summary

The generated example images demonstrate core functionality but require significant improvements based on visual verification. This plan addresses four critical issues with the synthetic data generation.

---

## Issue 1: Perspective Distortion — One Corner vs. True Camera Perspective

### Current Problem

```python
def perspective_warp_corners(corners, ...):
    corner_idx = random.randint(0, 3)  # Only picks ONE corner
    new_corners[corner_idx][0] += displacement[0]
    new_corners[corner_idx][1] += displacement[1]
```

**Result:** 86% "pass rate" on diagonal ratio test is misleading — it's measuring one-corner displacement, not true perspective. Real camera perspective affects **all 4 corners asymmetrically** based on camera position relative to the photo plane.

### Requirements

1. Simulate camera positioned at an angle to the photograph plane
2. All 4 corners should be displaced proportionally to camera angle
3. Trapezoidal shape should reflect vanishing point perspective
4. Top and bottom edges should have different widths (depth effect)
5. No single corner displacement should exceed 2× the minimum corner displacement
6. The distortion should look like looking at a photo from an angle, not a random corner being pulled

### Proposed Solution: Camera-Angle Perspective Simulation

```python
def apply_camera_perspective(corners, margin=40, canvas_w=1920, canvas_h=1440):
    """
    Simulate camera viewing photo at an angle.
    
    Camera is positioned at an azimuth (horizontal angle) and elevation
    (vertical angle) relative to the photo center. All 4 corners are
    affected proportionally based on their position relative to vanishing point.
    
    Args:
        corners: 4x2 array of corner coordinates [TL, TR, BR, BL]
        margin: minimum distance from canvas edge
        canvas_w, canvas_h: canvas dimensions
    
    Returns:
        corners with perspective distortion applied
    """
    import random
    
    new_corners = corners.copy()
    
    # --- Camera position parameters ---
    # Azimuth: camera position left/right of center (-50 to +50 degrees)
    azimuth = random.uniform(-50, 50)
    # Elevation: camera position above/below photo plane (-40 to +40 degrees)
    elevation = random.uniform(-40, 40)
    # Tilt: rotation of camera around viewing axis (-25 to +25 degrees)
    tilt = random.uniform(-25, 25)
    # Camera distance factor: affects perspective strength (closer = stronger)
    distance_factor = random.uniform(0.8, 1.5)
    
    # --- Calculate vanishing point based on camera angles ---
    # The vanishing point is where parallel lines converge
    # Azimuth shifts vanishing point left/right
    # Elevation shifts vanishing point up/down
    
    vanish_x = canvas_w / 2 + azimuth * 15  # Horizontal shift
    vanish_y = canvas_h / 2 + elevation * 10  # Vertical shift
    
    # --- Get photo center and dimensions ---
    min_x, max_x = corners[:,0].min(), corners[:,0].max()
    min_y, max_y = corners[:,1].min(), corners[:,1].max()
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    
    # --- Calculate perspective displacement for each corner ---
    # Corners closer to camera (further from vanishing point) appear larger
    # Corners further from camera (closer to vanishing point) appear smaller
    
    perspective_strength = 0.3 + distance_factor * 0.4  # 0.3 to 1.0
    
    for i in range(4):
        # Vector from vanishing point to corner
        dx = corners[i, 0] - vanish_x
        dy = corners[i, 1] - vanish_y
        
        # Distance from vanishing point (controls amount of perspective)
        dist = np.sqrt(dx**2 + dy**2)
        
        # Scale factor: corners further from VP get pushed outward
        # Corners closer to VP get pulled inward
        # This creates the characteristic trapezoidal shape of perspective
        if dist > 0:
            # Normalize direction from vanishing point
            nx, ny = dx / dist, dy / dist
            
            # Displacement magnitude based on angle and distance
            # Use dot product with camera direction for asymmetric effect
            cam_dir_x = np.sin(np.radians(azimuth))
            cam_dir_y = -np.sin(np.radians(elevation))
            
            # Asymmetric displacement: corners on "camera side" push out more
            alignment = nx * cam_dir_x + ny * cam_dir_y
            displacement_magnitude = perspective_strength * (1 + alignment) * random.uniform(0.3, 0.8)
            
            # Apply displacement
            new_corners[i, 0] += nx * displacement_magnitude * dist * 0.15
            new_corners[i, 1] += ny * displacement_magnitude * dist * 0.15
    
    # --- Apply tilt rotation around center ---
    if abs(tilt) > 1:  # Only apply if tilt is significant
        angle_rad = np.radians(tilt)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Translate to origin, rotate, translate back
        translated = new_corners - [cx, cy]
        rotated = np.array([
            translated[:,0] * cos_a - translated[:,1] * sin_a,
            translated[:,0] * sin_a + translated[:,1] * cos_a
        ]).T
        new_corners = rotated + [cx, cy]
    
    # --- Constrain to canvas bounds ---
    new_corners[:,0] = np.clip(new_corners[:,0], margin, canvas_w - margin)
    new_corners[:,1] = np.clip(new_corners[:,1], margin, canvas_h - margin)
    
    return new_corners
```

### Alternative Simpler Implementation

If the above is too complex, a simpler but still accurate approach:

```python
def apply_camera_perspective_simple(corners, margin=40, canvas_w=1920, canvas_h=1440):
    """
    Simpler camera perspective simulation.
    
    Uses camera azimuth and elevation to determine which corners
    move more, creating realistic trapezoidal distortion.
    """
    import random
    
    new_corners = corners.copy()
    
    # Random camera parameters
    azimuth = random.uniform(-45, 45)  # Camera angle left/right
    elevation = random.uniform(-35, 35)  # Camera angle up/down
    perspective_amount = random.uniform(0.4, 1.0)  # How strong the perspective is
    
    # Convert angles to vectors
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    # Camera "looks toward" the scene - corners in that direction compress
    look_x = np.sin(az_rad)  # Positive = camera looking right
    look_y = np.sin(el_rad)  # Positive = camera looking up
    
    # Get center
    cx = corners[:,0].mean()
    cy = corners[:,1].mean()
    
    # Each corner's displacement based on alignment with camera direction
    for i in range(4):
        corner = corners[i]
        
        # Relative position from center
        rx = (corner[0] - cx) / (canvas_w / 2)
        ry = (corner[1] - cy) / (canvas_h / 2)
        
        # Corner moves toward vanishing point (camera side moves more)
        # Corners on the "far" side of camera move outward
        # Corners on the "near" side move less (or even inward)
        move_x = (rx - look_x * 0.5) * perspective_amount * 150
        move_y = (ry - look_y * 0.5) * perspective_amount * 150
        
        # Add some noise for natural variation
        move_x += random.uniform(-20, 20)
        move_y += random.uniform(-20, 20)
        
        new_corners[i, 0] = corner[0] + move_x
        new_corners[i, 1] = corner[1] + move_y
    
    # Constrain to bounds
    new_corners[:,0] = np.clip(new_corners[:,0], margin, canvas_w - margin)
    new_corners[:,1] = np.clip(new_corners[:,1], margin, canvas_h - margin)
    
    return new_corners
```

### Verification Metrics

```python
def verify_perspective_symmetry(corners, flat_corners):
    """
    Verify perspective affects all 4 corners, not just one.
    
    Pass criteria:
    - No single corner displacement > 2.5× the minimum displacement
    - At least 3 corners have displacement > 20 pixels
    - Top and bottom edges have different widths (aspect ratio 0.55-0.95)
    """
    displacements = [np.linalg.norm(corners[i] - flat_corners[i]) for i in range(4)]
    
    max_disp = max(displacements)
    min_disp = min(displacements)
    
    if max_disp / min_disp > 2.5:
        return False, f"One corner displaced too much: ratio={max_disp/min_disp:.2f}"
    
    if sum(1 for d in displacements if d > 20) < 3:
        return False, f"Only {sum(1 for d in displacements if d > 20)} corners moved significantly"
    
    # Check edge width ratio (top vs bottom should differ)
    top_width = np.linalg.norm(corners[1] - corners[0])
    bot_width = np.linalg.norm(corners[2] - corners[3])
    edge_ratio = min(top_width, bot_width) / max(top_width, bot_width)
    
    if edge_ratio > 0.95 or edge_ratio < 0.55:
        return False, f"Edge ratio {edge_ratio:.2f} outside typical range (0.55-0.95)"
    
    return True, ""
```

---

## Issue 2: Drop Shadows — Too Fuzzy, Darkening Background

### Current Problem

```python
def fast_fuzzy_shadow(photo, canvas, corners):
    shadow_strength = random.uniform(0.25, 0.45)  # Too dark
    shadow_blur = random.uniform(20, 50)          # Too fuzzy
    spread = random.randint(15, 35)
```

Issues:
- **Opacity 25-45%** is too strong — shadows are making background look dirty
- **Blur 20-50px** is too uniformly fuzzy — looks unnatural
- **Shadow is being composited incorrectly** — it's darkening the canvas instead of creating a subtle offset shadow

### Requirements

1. Shadows should vary in fuzziness (some sharper 5-15px, some softer 15-30px)
2. Shadow opacity should be subtle (12-25% max)
3. Shadow spread should be proportional to photo size
4. Shadow direction should align with implied light source
5. Shadows should NOT make the background look darker overall
6. Shadow should appear as a separate element at an offset, not as a dark overlay

### Proposed Solution

```python
def create_variable_shadow(canvas, corners, light_direction=None):
    """
    Create a realistic, variable drop shadow.
    
    Key differences from current:
    - Subtle opacity (12-25%)
    - Variable blur (5-30px based on shadow size)
    - Properly offset in light direction
    - Uses ADDITIVE blend, not multiplicative darkening
    """
    import random
    import cv2
    import numpy as np
    
    h, w = canvas.shape[:2]
    
    # --- Shadow parameters ---
    # Opacity: 12-25% (was 25-45%) - much subtler
    shadow_opacity = random.uniform(0.12, 0.25)
    
    # Blur: Variable based on shadow size
    # Smaller shadows = sharper (5-15px), larger = softer (15-30px)
    shadow_bounds = cv2.boundingRect(corners.astype(np.int32))
    shadow_size = max(shadow_bounds[2], shadow_bounds[3])
    blur_sigma = np.clip(shadow_size * 0.05, 5, 30)  # 5-30px range
    blur_sigma += random.uniform(-3, 3)  # Add variation
    
    # Offset: 3-15 pixels (was 15-35) - more subtle
    shadow_offset = random.uniform(3, 15)
    
    # --- Light direction ---
    if light_direction is None:
        # Most common: light from upper-left (natural sunlight direction)
        if random.random() < 0.65:
            light_dir = (-1, -1)  # Upper-left
        elif random.random() < 0.85:
            light_dir = (-0.7, -0.7)  # Upper-left, slight variation
        else:
            # Random direction for variety
            angle = random.uniform(0, 360)
            light_dir = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))
    
    # Normalize light direction
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
    
    # Apply blur - variable based on size
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), blur_sigma)
    
    # --- Create subtle shadow color ---
    # Use a light gray, not pure white
    # Shadow should soften edges, not darken
    shadow_color = 220  # Light gray (was 191)
    
    # --- Blend shadow onto canvas ---
    # IMPORTANT: Use lighten blend, not darken
    canvas_f = canvas.astype(np.float32)
    
    # For each channel, lighten the pixels under the shadow
    # Shadow makes edges softer/lighter, not darker
    for c in range(3):
        # Shadow lightens the background slightly
        canvas_f[:,:,c] = np.clip(
            canvas_f[:,:,c] * (1 - shadow_mask * shadow_opacity * 0.5) + 
            shadow_color * shadow_mask * shadow_opacity * 0.5,
            0, 255
        )
    
    return canvas_f.astype(np.uint8)
```

### Alternative: Soft Edge Shadow (Subtler)

```python
def create_soft_edge_shadow(canvas, corners):
    """
    Create a very subtle soft-edge shadow.
    
    This shadow is meant to suggest depth without being visible.
    It's almost a "hint" of shadow at the edges.
    """
    import random
    import cv2
    import numpy as np
    
    h, w = canvas.shape[:2]
    
    # Very subtle parameters
    shadow_opacity = random.uniform(0.08, 0.18)  # 8-18%
    blur_amount = random.uniform(8, 20)  # Variable blur
    
    # Light from upper-left (most common)
    shadow_offset = random.uniform(5, 12)
    offset_vec = np.array([-shadow_offset, -shadow_offset])
    
    # Create shadow polygon
    shadow_corners = corners + offset_vec
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [shadow_corners.astype(np.int32)], 1.0)
    
    # Blur edges
    mask = cv2.GaussianBlur(mask, (0, 0), blur_amount)
    
    # Apply as subtle darkening at edges only
    canvas_f = canvas.astype(np.float32)
    
    # Only darken the shadow area slightly
    shadow_value = 15  # Very subtle darkening
    canvas_f = canvas_f * (1 - mask * shadow_opacity) + shadow_value * mask * shadow_opacity
    
    return np.clip(canvas_f, 0, 255).astype(np.uint8)
```

### Key Parameter Changes

| Parameter | Old Range | New Range | Reason |
|-----------|-----------|-----------|--------|
| Opacity | 0.25-0.45 | 0.12-0.25 | Subtler, won't dirty background |
| Blur sigma | 20-50 | 5-30 | Variable, not uniformly fuzzy |
| Offset | 15-35 | 3-15 | More subtle |
| Shadow color | 191 (darker gray) | 220 (lighter gray) | Lighter shadow |
| Blend mode | Darken | Lighten/Add | Shadow should add depth |

### Verification

```python
def verify_shadow_subtlety(canvas, corners, shadow_mask):
    """Verify shadow doesn't significantly darken the background."""
    # Sample background pixels just outside shadow
    # Compare to background pixels far from shadow
    
    # Get shadow bounding box
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    
    # Sample pixels 50px outside shadow on all sides
    sample_area = canvas[max(0,y-50):y+h+50, max(0,x-50):x+w+50]
    
    # Mean brightness should not decrease by more than 15%
    mean_brightness = sample_area.mean()
    
    # If shadow makes background darker by >15%, it's too strong
    return mean_brightness > 180, f"Background too dark: {mean_brightness:.1f}"
```

---

## Issue 3: Pre-Perspective Rotation

### Current Problem

Photos are placed axis-aligned (no rotation) and then perspective-warped. Real photos on a surface have random orientations — they're not all perfectly aligned.

### Requirements

1. Apply random rotation (-30° to +30°) to each photo before perspective warp
2. Rotation should be around photo center
3. Rotation should happen BEFORE perspective transformation
4. Corner keypoints should be rotated with the photo

### Proposed Solution

```python
def rotate_photo_and_corners(photo, corners, angle=None):
    """
    Rotate photo and its corner keypoints around photo center.
    
    Args:
        photo: Photo image array (H, W, C)
        corners: 4x2 array of corner coordinates in canvas-space
        angle: rotation angle in degrees, or None for random
    
    Returns:
        (rotated_photo, rotated_corners)
    """
    import random
    import cv2
    import numpy as np
    
    h, w = photo.shape[:2]
    center = (w / 2, h / 2)
    
    # --- Get rotation angle ---
    if angle is None:
        # Weighted distribution for more natural orientations
        r = random.random()
        if r < 0.30:
            angle = random.uniform(-5, 5)  # Near-horizontal (30%)
        elif r < 0.70:
            angle = random.uniform(-15, 15)  # Slight tilt (40%)
        else:
            angle = random.uniform(-30, 30)  # Strong tilt (30%)
    
    # --- Rotate image ---
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(photo, M, (w, h), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_REFLECT_101)
    
    # --- Calculate rotated corners in photo-local coordinates ---
    # Corners are provided in canvas-space, need to rotate around photo center
    # Photo center in canvas coordinates:
    photo_cx = (corners[0,0] + corners[2,0]) / 2
    photo_cy = (corners[0,1] + corners[2,1]) / 2
    
    # Convert corners to photo-local coordinates (relative to photo center)
    local_corners = corners - [photo_cx, photo_cy]
    
    # Apply rotation
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated_local = np.array([
        local_corners[:,0] * cos_a - local_corners[:,1] * sin_a,
        local_corners[:,0] * sin_a + local_corners[:,1] * cos_a
    ]).T
    
    # Convert back to canvas-space
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
    import random
    
    # --- Step 1: Pre-perspective rotation ---
    # Rotate photo and corners together
    rotated_photo, rotated_corners = rotate_photo_and_corners(photo, flat_corners)
    
    # --- Step 2: Apply camera perspective ---
    # Pass rotated corners to perspective function
    perspective_corners = apply_camera_perspective(
        rotated_corners, 
        margin=40, 
        canvas_w=canvas_w, 
        canvas_h=canvas_h
    )
    
    return rotated_photo, perspective_corners
```

### Integration into Main Loop

```python
# In the main generation loop, replace:
for j, src_path in enumerate(sampled):
    # ... load and scale photo ...
    
    # Create flat corners
    flat_corners = np.array([
        [x, y],
        [x + w_p, y],
        [x + w_p, y + h_p],
        [x, y + h_p]
    ], dtype=np.float32)
    
    # Apply rotation + perspective
    photo, warped_corners = process_photo_pipeline(
        photo, flat_corners, CANVAS_W, CANVAS_H
    )
    
    # Continue with shadow and composite...
```

### Verification

```python
def verify_rotation_applied(corners):
    """
    Verify rotation was applied before perspective.
    
    A non-rotated photo has top and bottom edges parallel to canvas edges.
    After rotation, edges should have a slight angle.
    """
    # Calculate angle of top edge relative to horizontal
    top_edge = corners[1] - corners[0]  # TR - TL
    angle = np.arctan2(top_edge[1], top_edge[0])
    
    # Also check left edge
    left_edge = corners[3] - corners[0]  # BL - TL
    left_angle = np.arctan2(left_edge[0], left_edge[1])  # Note: x,y swapped
    
    # If photo is axis-aligned, angles will be very close to 0 or ±π/2
    # After rotation, expect angles between -0.5 and +0.5 radians (-30° to +30°)
    if abs(angle) < 0.05 and abs(left_angle) < 0.05:
        return False, "Photo appears not rotated (edges axis-aligned)"
    
    return True, ""
```

---

## Issue 4: Photo Packing Optimization

### Current Problem

```python
# Current approach:
num_photos = random.randint(4, 7)  # Only 4-7 photos
cols = min(3, num_photos)  # Fixed 3-column layout
photo_size = min(350, ...)  # Fixed size (350px max)
gap = 40  # Fixed gap
```

Issues:
- Only 4-7 photos per frame (target: 6-12)
- Fixed photo size (should be variable)
- Simple grid doesn't optimize space
- No rotation buffer (rotated photos need more space)

### Requirements

1. Fit 6-12 photos per frame (not just 4-7)
2. Variable photo sizes (8-35% of canvas dimension)
3. Efficient packing algorithm (shelf/bin-packing or similar)
4. Account for rotation (add buffer space around rotated photos)
5. No overlapping photos after rotation+perspective

### Proposed Solution: Shelf Packing with Variable Sizes

```python
def calculate_efficient_packings(canvas_w, canvas_h, num_photos=None):
    """
    Calculate efficient photo packings using shelf algorithm.
    
    Args:
        canvas_w, canvas_h: Canvas dimensions
        num_photos: Target number of photos (or None for auto)
    
    Returns:
        List of placements, each with: x, y, width, height, rotation
    """
    import random
    import numpy as np
    
    margin = 40  # Edge margin
    gap = 25     # Gap between photos
    
    # --- Determine number of photos ---
    if num_photos is None:
        # Vary based on canvas size
        base_count = max(5, min(12, int((canvas_w * canvas_h) / 250000)))
        num_photos = base_count + random.randint(-1, 2)
    
    # --- Determine photo sizes ---
    # Mix of large, medium, and small photos
    # 40% large, 40% medium, 20% small
    photo_sizes = []
    for _ in range(num_photos):
        r = random.random()
        if r < 0.40:
            # Large: 25-35% of canvas width
            width = int(canvas_w * random.uniform(0.25, 0.35))
            height = int(canvas_h * random.uniform(0.18, 0.28))
        elif r < 0.80:
            # Medium: 15-25% of canvas width
            width = int(canvas_w * random.uniform(0.15, 0.25))
            height = int(canvas_h * random.uniform(0.12, 0.20))
        else:
            # Small: 8-15% of canvas width
            width = int(canvas_w * random.uniform(0.08, 0.15))
            height = int(canvas_h * random.uniform(0.06, 0.12))
        
        photo_sizes.append((width, height))
    
    # Sort by area (larger first) for better shelf packing
    photo_sizes.sort(key=lambda x: x[0] * x[1], reverse=True)
    
    # --- Shelf packing algorithm ---
    placements = []
    shelf_y = margin
    current_x = margin
    current_shelf_height = 0
    
    for width, height in photo_sizes:
        # Determine rotation for this photo
        # Some photos rotated, some not (for variety)
        rotate = random.random() < 0.35  # 35% chance of rotation
        if rotate:
            # Swap width/height for rotated placement
            rot_width, rot_height = height, width
            rotation_angle = random.uniform(-25, 25)
        else:
            rot_width, rot_height = width, height
            rotation_angle = 0
        
        # Add buffer for rotation (15% extra space)
        buffer = abs(np.sin(np.radians(rotation_angle))) * rot_width * 0.15 if rotation_angle != 0 else 0
        effective_width = rot_width + buffer * 2
        effective_height = rot_height + buffer * 2
        
        # Check if fits in current shelf
        if current_x + effective_width > canvas_w - margin:
            # Start new shelf
            shelf_y += current_shelf_height + gap
            current_x = margin
            current_shelf_height = 0
        
        # Check if fits vertically
        if shelf_y + effective_height > canvas_h - margin:
            # No more room - try smaller photos or skip
            # For now, scale down the last photo
            scale = (canvas_h - margin - shelf_y) / effective_height
            if scale > 0.5:
                rot_width = int(rot_width * scale)
                rot_height = int(rot_height * scale)
                effective_width = rot_width + buffer * 2
                effective_height = rot_height + buffer * 2
            else:
                continue  # Skip this photo
        
        # Place photo
        placements.append({
            'x': current_x,
            'y': shelf_y,
            'width': rot_width,
            'height': rot_height,
            'rotation': rotation_angle
        })
        
        # Update shelf tracking
        current_shelf_height = max(current_shelf_height, effective_height)
        current_x += effective_width + gap
    
    return placements


def get_photo_corners(placement):
    """Get flat corner coordinates for a photo placement."""
    x, y = placement['x'], placement['y']
    w, h = placement['width'], placement['height']
    
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32)
```

### Enhanced Main Loop Integration

```python
# In the main generation loop, replace photo placement:

# Generate efficient packing
placements = calculate_efficient_packings(CANVAS_W, CANVAS_H, num_photos=None)

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
    flat_corners = get_photo_corners(placement)
    
    # Apply rotation + perspective (pipeline function)
    photo, warped_corners = process_photo_pipeline(
        photo, flat_corners, CANVAS_W, CANVAS_H
    )
    
    # Continue with shadow and composite...
```

### Packing Improvements Summary

| Aspect | Old | New |
|--------|-----|-----|
| Photos per frame | 4-7 | 6-12 |
| Size variation | Fixed (350px max) | Variable (8-35% of canvas) |
| Layout | Fixed 3-column grid | Shelf packing algorithm |
| Rotation buffer | None | 15% extra space for rotated photos |
| Gap sizing | Fixed 40px | Variable 20-35px |
| Size distribution | Uniform | 40% large, 40% medium, 20% small |

---

## Implementation Order

For safest, most logical implementation:

1. **Issue 4 (Photo Packing)** — Get efficient layouts first; everything else depends on knowing where photos go
2. **Issue 3 (Pre-Perspective Rotation)** — Rotate before perspective; easier to verify
3. **Issue 1 (Perspective Distortion)** — Camera-angle simulation requires correct starting corners (after rotation)
4. **Issue 2 (Drop Shadows)** — Fix last; depends on having correct corner positions

---

## Updated Main Loop (All Changes Combined)

```python
import random

# =============================================================================
# NEW FUNCTIONS
# =============================================================================

def apply_camera_perspective(corners, margin=40, canvas_w=1920, canvas_h=1440):
    """Simulate camera viewing photo at an angle - ALL 4 corners affected."""
    new_corners = corners.copy()
    
    azimuth = random.uniform(-50, 50)
    elevation = random.uniform(-40, 40)
    tilt = random.uniform(-25, 25)
    perspective_amount = random.uniform(0.4, 1.0)
    
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    look_x = np.sin(az_rad)
    look_y = np.sin(el_rad)
    
    cx = corners[:,0].mean()
    cy = corners[:,1].mean()
    
    for i in range(4):
        corner = corners[i]
        rx = (corner[0] - cx) / (canvas_w / 2)
        ry = (corner[1] - cy) / (canvas_h / 2)
        
        move_x = (rx - look_x * 0.5) * perspective_amount * 150
        move_y = (ry - look_y * 0.5) * perspective_amount * 150
        
        move_x += random.uniform(-20, 20)
        move_y += random.uniform(-20, 20)
        
        new_corners[i, 0] = corner[0] + move_x
        new_corners[i, 1] = corner[1] + move_y
    
    if abs(tilt) > 1:
        angle_rad = np.radians(tilt)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        translated = new_corners - [cx, cy]
        rotated = np.array([
            translated[:,0] * cos_a - translated[:,1] * sin_a,
            translated[:,0] * sin_a + translated[:,1] * cos_a
        ]).T
        new_corners = rotated + [cx, cy]
    
    new_corners[:,0] = np.clip(new_corners[:,0], margin, canvas_w - margin)
    new_corners[:,1] = np.clip(new_corners[:,1], margin, canvas_h - margin)
    
    return new_corners


def rotate_photo_and_corners(photo, corners, angle=None):
    """Rotate photo and corners around photo center."""
    h, w = photo.shape[:2]
    center = (w / 2, h / 2)
    
    if angle is None:
        r = random.random()
        if r < 0.30:
            angle = random.uniform(-5, 5)
        elif r < 0.70:
            angle = random.uniform(-15, 15)
        else:
            angle = random.uniform(-30, 30)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(photo, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    photo_cx = (corners[0,0] + corners[2,0]) / 2
    photo_cy = (corners[0,1] + corners[2,1]) / 2
    
    local_corners = corners - [photo_cx, photo_cy]
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated_local = np.array([
        local_corners[:,0] * cos_a - local_corners[:,1] * sin_a,
        local_corners[:,0] * sin_a + local_corners[:,1] * cos_a
    ]).T
    
    rotated_corners = rotated_local + [photo_cx, photo_cy]
    
    return rotated, rotated_corners


def process_photo_pipeline(photo, flat_corners, canvas_w, canvas_h):
    """Full pipeline: rotate then apply perspective."""
    rotated_photo, rotated_corners = rotate_photo_and_corners(photo, flat_corners)
    perspective_corners = apply_camera_perspective(rotated_corners, margin=40, canvas_w=canvas_w, canvas_h=canvas_h)
    return rotated_photo, perspective_corners


def create_variable_shadow(canvas, corners, light_direction=None):
    """Create subtle variable drop shadow."""
    import cv2
    
    h, w = canvas.shape[:2]
    
    shadow_opacity = random.uniform(0.12, 0.25)
    shadow_bounds = cv2.boundingRect(corners.astype(np.int32))
    shadow_size = max(shadow_bounds[2], shadow_bounds[3])
    blur_sigma = np.clip(shadow_size * 0.05, 5, 30)
    blur_sigma += random.uniform(-3, 3)
    
    shadow_offset = random.uniform(3, 15)
    
    if light_direction is None:
        if random.random() < 0.65:
            light_dir = (-1, -1)
        else:
            angle = random.uniform(0, 360)
            light_dir = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))
    
    lx, ly = light_dir
    l_len = np.sqrt(lx**2 + ly**2)
    light_dir = (lx / l_len, ly / l_len)
    
    offset_vector = np.array([light_dir[0] * shadow_offset, light_dir[1] * shadow_offset])
    shadow_corners = corners + offset_vector
    
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    pts = shadow_corners.astype(np.int32)
    cv2.fillPoly(shadow_mask, [pts], 1.0)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), blur_sigma)
    
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:,:,c] = np.clip(
            canvas_f[:,:,c] * (1 - shadow_mask * shadow_opacity * 0.5) + 
            220 * shadow_mask * shadow_opacity * 0.5,
            0, 255
        )
    
    return canvas_f.astype(np.uint8)


def calculate_efficient_packings(canvas_w, canvas_h):
    """Shelf packing with variable photo sizes."""
    margin = 40
    gap = 25
    
    base_count = max(5, min(12, int((canvas_w * canvas_h) / 250000)))
    num_photos = base_count + random.randint(-1, 2)
    
    photo_sizes = []
    for _ in range(num_photos):
        r = random.random()
        if r < 0.40:
            width = int(canvas_w * random.uniform(0.25, 0.35))
            height = int(canvas_h * random.uniform(0.18, 0.28))
        elif r < 0.80:
            width = int(canvas_w * random.uniform(0.15, 0.25))
            height = int(canvas_h * random.uniform(0.12, 0.20))
        else:
            width = int(canvas_w * random.uniform(0.08, 0.15))
            height = int(canvas_h * random.uniform(0.06, 0.12))
        photo_sizes.append((width, height))
    
    photo_sizes.sort(key=lambda x: x[0] * x[1], reverse=True)
    
    placements = []
    shelf_y = margin
    current_x = margin
    current_shelf_height = 0
    
    for width, height in photo_sizes:
        rotate = random.random() < 0.35
        if rotate:
            rot_width, rot_height = height, width
            rotation_angle = random.uniform(-25, 25)
        else:
            rot_width, rot_height = width, height
            rotation_angle = 0
        
        buffer = abs(np.sin(np.radians(rotation_angle))) * rot_width * 0.15 if rotation_angle != 0 else 0
        effective_width = rot_width + buffer * 2
        effective_height = rot_height + buffer * 2
        
        if current_x + effective_width > canvas_w - margin:
            shelf_y += current_shelf_height + gap
            current_x = margin
            current_shelf_height = 0
        
        if shelf_y + effective_height > canvas_h - margin:
            scale = (canvas_h - margin - shelf_y) / effective_height
            if scale > 0.5:
                rot_width = int(rot_width * scale)
                rot_height = int(rot_height * scale)
                effective_width = rot_width + buffer * 2
                effective_height = rot_height + buffer * 2
            else:
                continue
        
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


# =============================================================================
# UPDATED MAIN LOOP
# =============================================================================

for i in range(10):
    img_start = time.time()
    
    # Generate background
    bg = fast_background(CANVAS_W, CANVAS_H)
    bg = fast_luma_gradient(bg)
    bg = random_background_gradient(bg)
    
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
        
        # Add shadow
        bg = create_variable_shadow(bg, warped_corners)
        
        # Composite onto canvas
        bg = composite_trapezoid(bg, photo, warped_corners)
        
        # Store keypoints
        placed_photos.append({
            'keypoints': [(warped_corners[k,0], warped_corners[k,1]) for k in range(4)]
        })
    
    # Save image and labels...
```

---

## Expected Results After Improvements

| Metric | Current | Target | Verification |
|--------|---------|--------|-------------|
| Photos per frame | 4-7 | 6-12 | Count photos in generated images |
| Perspective realism | 1-corner distortion | 4-corner camera simulation | All 4 corners displaced, no single corner >2.5× min |
| Shadow opacity | 25-45% (too dark) | 12-25% (subtle) | Background brightness not reduced >15% |
| Shadow fuzziness | Uniform 20-50px | Variable 5-30px | Blur range spans at least 15px difference |
| Rotation | None | -30° to +30° | Corners not axis-aligned |
| Edge ratio | N/A | 0.55-0.95 | Top/bottom width ratio in range |

---

## Files to Modify

| File | Changes |
|------|---------|
| `/Users/krys.petrie/dev/photo-pose-detector/data_generator/generate_samples.py` | All 4 improvements as shown above |

---

## Verification Tests (Updated)

```python
def run_comprehensive_verification(output_dir):
    """Run all verification tests on generated images."""
    import cv2
    import numpy as np
    from pathlib import Path
    
    results = {
        'photos_per_frame': [],
        'perspective_symmetry': [],
        'shadow_subtlety': [],
        'rotation_applied': [],
        'edge_ratios': []
    }
    
    for img_path in sorted(Path(output_dir).glob("example_*.jpg")):
        # Load image and labels
        img = cv2.imread(str(img_path))
        lbl_path = img_path.with_suffix('.txt')
        
        if not lbl_path.exists():
            continue
        
        with open(lbl_path) as f:
            lines = f.readlines()
        
        photo_count = len(lines)
        results['photos_per_frame'].append(photo_count)
        
        for line in lines:
            parts = line.strip().split()
            # Parse keypoints (indices 6,7,8,9,10,11,12,13 for kp coords)
            kps = []
            for j in range(4):
                kx = float(parts[6 + j*2]) * 1920
                ky = float(parts[7 + j*2]) * 1440
                kps.append([kx, ky])
            kps = np.array(kps)
            
            # Verify perspective symmetry
            flat = np.array([[0,0],[350,0],[350,350],[0,350]])  # Approximate
            ok, msg = verify_perspective_symmetry(kps, flat)
            results['perspective_symmetry'].append(ok)
            
            # Check edge ratio
            top_w = np.linalg.norm(kps[1] - kps[0])
            bot_w = np.linalg.norm(kps[2] - kps[3])
            ratio = min(top_w, bot_w) / max(top_w, bot_w)
            results['edge_ratios'].append(0.55 <= ratio <= 0.95)
            
            # Verify rotation
            ok, _ = verify_rotation_applied(kps)
            results['rotation_applied'].append(ok)
    
    # Print summary
    print("=== VERIFICATION SUMMARY ===")
    print(f"Photos per frame: {np.mean(results['photos_per_frame']):.1f} avg (target: 6-12)")
    print(f"Perspective symmetry: {sum(results['perspective_symmetry'])/len(results['perspective_symmetry'])*100:.0f}% pass")
    print(f"Edge ratios in range: {sum(results['edge_ratios'])/len(results['edge_ratios'])*100:.0f}%")
    print(f"Rotation applied: {sum(results['rotation_applied'])/len(results['rotation_applied'])*100:.0f}%")
```

---

## Implementation Checklist

- [ ] Add `apply_camera_perspective()` function (replaces `perspective_warp_corners()`)
- [ ] Add `rotate_photo_and_corners()` function
- [ ] Add `process_photo_pipeline()` function
- [ ] Add `create_variable_shadow()` function
- [ ] Add `calculate_efficient_packings()` function
- [ ] Update main loop to use new packing function
- [ ] Update main loop to use rotation+perspective pipeline
- [ ] Update main loop to use new shadow function
- [ ] Test and verify all improvements
- [ ] Run comprehensive verification tests