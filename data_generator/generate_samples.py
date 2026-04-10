#!/usr/bin/env python3
"""
Fast generation of 10 example images.
Uses vectorized operations and smaller canvas to stay within timeout.
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
signal.alarm(120)

print("🖼️  Generating 10 example images (FAST mode)...")
print("⏱️  Timeout: 120 seconds")
print()

start_time = time.time()

try:
    import numpy as np
    import cv2
    from PIL import Image
    
    # =============================================================================
    # FAST, VECTORIZED BACKGROUND GENERATOR
    # =============================================================================
    
    def fast_background(w, h):
        """Generate background in ~0.1s instead of ~930s using vectorized ops."""
        import random
        from numpy.random import randint, uniform, normal
        
        # Decide type
        bg_type = random.choices(['solid', 'wood', 'fabric', 'laminate'], weights=[0.4, 0.25, 0.2, 0.15])[0]
        
        if bg_type == 'solid':
            # Three-tier color palette
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
            # Vectorized grain lines
            num_grains = randint(15, 25)
            for _ in range(num_grains):
                y = randint(0, h-1)
                thickness = randint(2, 8)
                intensity = randint(-30, 30)
                offset = random.uniform(-0.02, 0.02)
                y1, y2 = max(0, y-thickness), min(h, y+thickness)
                # Apply grain across width with wavy offset
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
        """Add luma gradient overlay - vectorized."""
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
    
    def fast_photo_manipulation(img):
        """Apply brightness/contrast/saturation/gamma - vectorized."""
        import random
        brightness = random.uniform(0.85, 1.20)
        contrast   = random.uniform(0.85, 1.20)
        saturation = random.uniform(0.80, 1.25)
        gamma      = random.uniform(0.80, 1.30)
        
        img_f = img.astype(np.float32) / 255.0
        
        # Brightness & contrast
        img_f = ((img_f - 0.5) * contrast + 0.5) * brightness
        img_f = np.clip(img_f, 0, 1)
        
        # Saturation in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= saturation
        hsv[:,:,2] *= brightness * contrast
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
        # Blend original + saturated
        blend = random.uniform(0.0, 1.0)
        img_f = img_f * (1 - blend) + img_sat * blend
        
        # Gamma
        img_f = np.power(np.clip(img_f, 0.001, 1), 1.0/gamma)
        
        return np.clip(img_f * 255, 0, 255).astype(np.uint8)
    
    def fast_fuzzy_shadow(photo, canvas, corners):
        """Add fuzzy drop shadow."""
        import random
        shadow_strength = random.uniform(0.25, 0.45)
        shadow_blur = random.uniform(20, 50)
        
        # Create shadow mask
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        pts = corners.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Dilate mask for shadow spread
        spread = random.randint(15, 35)
        kernel = np.ones((3,3), dtype=np.uint8)
        shadow_mask = cv2.dilate(mask, kernel, iterations=spread)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0,0), shadow_blur)
        
        # Create shadow color
        shadow_val = int(255 * (1 - shadow_strength))
        shadow_color = np.array([shadow_val] * 3, dtype=np.uint8)
        
        # Composite shadow onto canvas
        for c in range(3):
            channel = canvas[:,:,c].astype(np.float32)
            shadow_channel = shadow_mask.astype(np.float32) / 255.0
            channel = channel * (1 - shadow_channel * shadow_strength) + shadow_val * shadow_channel * shadow_strength
            canvas[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return canvas
    
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
                
                # Screen blend mode: result = 1 - (1-A)*(1-B)
                img_f = img.astype(np.float32) / 255.0
                flare_bgr = np.stack([flare * 255] * 3, axis=-1)
                flare_f = flare_bgr / 255.0
                img_f = 1 - (1 - img_f) * (1 - flare_f * 0.3)
                img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
        return img
    
    def perspective_warp_corners(corners, margin=30, canvas_w=1920, canvas_h=1440):
        """Warp corners to create trapezoidal perspective.
        
        Displaces ONE corner by 50-200px while ensuring all corners
        stay within canvas bounds. Uses corner-specific constraints.
        
        Returns corners with guaranteed strong trapezoid distortion.
        """
        import random
        new_corners = corners.copy()
        
        # Choose which corner to displace (0=TL, 1=TR, 2=BR, 3=BL)
        corner_idx = random.randint(0, 3)
        
        # Base displacement - start with target range
        target_dx = random.uniform(50, 200)  # 50-200 pixels target
        target_dy = random.uniform(50, 200)  # 50-200 pixels target
        
        # Get corner positions
        c = corners[corner_idx]
        
        # Calculate safe displacement based on corner position and canvas bounds
        # Each corner can only move in its "free" directions
        if corner_idx == 0:  # TL corner - moves left/up (negative)
            safe_left = c[0] - margin
            safe_up = c[1] - margin
            dx = min(target_dx, safe_left * 0.9) if safe_left > 0 else target_dx * 0.3
            dy = min(target_dy, safe_up * 0.9) if safe_up > 0 else target_dy * 0.3
            displacement = (-abs(dx), -abs(dy))
        elif corner_idx == 1:  # TR corner - moves right/up (positive X, negative Y)
            safe_right = canvas_w - margin - c[0]
            safe_up = c[1] - margin
            dx = min(target_dx, safe_right * 0.9) if safe_right > 0 else target_dx * 0.3
            dy = min(target_dy, safe_up * 0.9) if safe_up > 0 else target_dy * 0.3
            displacement = (abs(dx), -abs(dy))
        elif corner_idx == 2:  # BR corner - moves right/down (positive)
            safe_right = canvas_w - margin - c[0]
            safe_down = canvas_h - margin - c[1]
            dx = min(target_dx, safe_right * 0.9) if safe_right > 0 else target_dx * 0.3
            dy = min(target_dy, safe_down * 0.9) if safe_down > 0 else target_dy * 0.3
            displacement = (abs(dx), abs(dy))
        else:  # BL corner - moves left/down (negative X, positive Y)
            safe_left = c[0] - margin
            safe_down = canvas_h - margin - c[1]
            dx = min(target_dx, safe_left * 0.9) if safe_left > 0 else target_dx * 0.3
            dy = min(target_dy, safe_down * 0.9) if safe_down > 0 else target_dy * 0.3
            displacement = (-abs(dx), abs(dy))
        
        # Apply displacement
        new_corners[corner_idx][0] += displacement[0]
        new_corners[corner_idx][1] += displacement[1]
        
        # CRITICAL: Clamp ALL corners to canvas bounds
        # This ensures no corner goes out of bounds during warping
        new_corners[:,0] = np.clip(new_corners[:,0], margin, canvas_w - margin)
        new_corners[:,1] = np.clip(new_corners[:,1], margin, canvas_h - margin)
        
        return new_corners
    
    def composite_trapezoid(canvas, photo, dst_pts):
        """Composite a photo warped to trapezoidal corners."""
        h, w = photo.shape[:2]
        src_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
        warped = cv2.warpPerspective(photo, M, (canvas.shape[1], canvas.shape[0]), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Create polygon mask
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [dst_pts.astype(np.int32)], 255)
        mask = cv2.erode(mask, np.ones((3,3), dtype=np.uint8), iterations=1)
        
        # Inpaint any black areas in warped image
        black_mask = ((warped[:,:,0]==0) & (warped[:,:,1]==0) & (warped[:,:,2]==0)).astype(np.uint8) * 255
        if black_mask.sum() > 0:
            kernel = np.ones((3,3), dtype=np.uint8)
            black_mask = cv2.dilate(black_mask, kernel, iterations=2)
            warped = cv2.inpaint(warped, black_mask, 3, cv2.INPAINT_TELEA)
        
        # Composite using mask
        for c in range(3):
            canvas[:,:,c] = np.where(mask > 0, warped[:,:,c], canvas[:,:,c])
        
        return canvas
    
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
            else:  # diagonal
                # Create 2D diagonal gradient from corner
                Y, X = np.ogrid[:h, :w]
                gradient = ((X/w + Y/h) / 2) * alpha * direction
            
            gradient = gradient[:, :, np.newaxis]
            img = np.clip(img * (1 + gradient), 0, 255).astype(np.uint8)
        
        return img
    
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
    
    output_dir = Path("../data/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use 1920x1440 canvas -> 1920x1440 output (1:1 scale for speed)
    CANVAS_W, CANVAS_H = 1920, 1440
    
    random.seed(42)
    np.random.seed(42)
    
    print("\n📸 Generating images...\n")
    
    for i in range(10):
        img_start = time.time()
        
        # Generate background
        bg = fast_background(CANVAS_W, CANVAS_H)
        bg = fast_luma_gradient(bg)
        bg = random_background_gradient(bg)
        
        # Place 4-7 photos
        num_photos = random.randint(4, 7)
        sampled = random.sample(sources, min(num_photos, len(sources)))
        
        placed_photos = []
        
        # Simple grid-ish placement
        cols = min(3, num_photos)
        rows = (num_photos + cols - 1) // cols
        cell_w = CANVAS_W // cols
        cell_h = CANVAS_H // rows
        
        photo_size = min(350, int(min(cell_w, cell_h) * 0.65))
        gap = 40
        
        for j, src_path in enumerate(sampled):
            col = j % cols
            row = j // cols
            
            # Load photo
            photo = cv2.imread(str(src_path))
            if photo is None:
                continue
            
            h, w = photo.shape[:2]
            scale = photo_size / max(w, h)
            photo = cv2.resize(photo, (int(w*scale), int(h*scale)))
            
            # Position in grid cell with some jitter
            jitter_x = random.randint(-30, 30)
            jitter_y = random.randint(-30, 30)
            x = col * cell_w + (cell_w - photo.shape[1]) // 2 + jitter_x
            y = row * cell_h + (cell_h - photo.shape[0]) // 2 + jitter_y
            
            # Create flat corners
            h_p, w_p = photo.shape[:2]
            flat_corners = np.array([
                [x, y],
                [x + w_p, y],
                [x + w_p, y + h_p],
                [x, y + h_p]
            ], dtype=np.float32)
            
            # Apply perspective warping
            warped_corners = perspective_warp_corners(flat_corners, canvas_w=CANVAS_W, canvas_h=CANVAS_H)
            
            # Apply effects
            photo = fast_photo_manipulation(photo)
            photo = fast_glare(photo)
            
            # Add shadow
            bg = fast_fuzzy_shadow(photo.copy(), bg, warped_corners)
            
            # Composite onto canvas
            bg = composite_trapezoid(bg, photo, warped_corners)
            
            # Store keypoints (warped corners)
            placed_photos.append({
                'keypoints': [(warped_corners[k,0], warped_corners[k,1]) for k in range(4)]
            })
        
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
