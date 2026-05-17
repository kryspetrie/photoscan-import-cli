# Kotlin Integration Guide

## Overview

This guide covers using the two ONNX models in a Kotlin/Android application:

1. **Detection model** — finds bounding boxes around photos in a scanned image
2. **Pose model** — finds the 4 corner keypoints of each detected photo

Both models share the same input format and preprocessing. The pipeline is:

```
Input Image
    │
    ├─→ Detection Model ─→ Bounding boxes (NMS raw output)
    │         │                → crop each region
    │         ▼
    └─→ Pose Model ─→ Corner keypoints per region
                         │
                         ▼ (optional, recommended)
              Corner Refinement ─→ Recover invisible corners
                         │
                         ▼
              Perspective-corrected crops
```

> **Important:** The detection model narrows down regions of interest before running the pose model. If you want to skip the detection step and run the pose model directly on the full image, you can — just note it may be slower for images with many objects.

### Corner Refinement

The pose model sometimes reports invisible corners (visibility ≈ 0) when a photo's edge is occluded by shadow, glare, or the scanner edge. **Corner refinement** crops around each approximate corner and runs the model again on just that small region, recovering invisible corners to visibility = 1.0.

The algorithm:
1. For each corner (UL, UR, LL, LR), crop a region around the approximate position
2. Run the pose model on that crop (640×640, auto-sized from photo bbox)
3. Find the named keypoint matching the target corner (e.g., find "UL" keypoint for UL corner)
4. If found with visibility ≥ 0.3, use its position — no classification needed
5. If not found, use the relevant bounding box corner as fallback
6. Validate: reject if the refined position moved more than 30% of crop size from the original

**Pose model (recommended):** Uses named keypoints directly. Fast (~120ms per corner). No extra model session needed (reuses the pose model). Single pass per corner with no expand retries.

**Detection model (fallback):** Uses bounding box corners with geometric classification. Slower due to expand retries when the bbox fills the crop. Less precise than keypoints. No extra model session needed (reuses the detection model).

All 4 corners × N photos are independent and can be processed in parallel — ideal for threading.

---

## Model I/O Reference

### Shared Input Format

Both models expect the same input:

| Field | Value |
|-------|-------|
| **Name** | `images` |
| **Shape** | `(batch, 3, height, width)` — dynamic dimensions |
| **Type** | `float32` |
| **Format** | RGB, CHW, normalized to `[0, 1]` |
| **Default size** | 640 × 640 (but any size works with dynamic shapes) |

Preprocessing steps:
1. Convert BGR → RGB (if coming from OpenCV/Android Bitmap)
2. Resize to 640×640 with letterboxing (or use dynamic shapes with the original aspect ratio)
3. Normalize pixel values to `[0, 1]` float
4. Transpose to CHW format
5. Add batch dimension

### Detection Model Output

| Field | Shape | Description |
|-------|-------|-------------|
| **Output 0** | `(1, 5, 8400)` | Raw YOLO detections |

Format: 5 channels × 8400 anchor points. Each column is:
- `[0, i]` = center_x (pixel coords)
- `[1, i]` = center_y (pixel coords)
- `[2, i]` = width (pixel coords)
- `[3, i]` = height (pixel coords)
- `[4, i]` = objectness confidence (0–1)

> **You must apply NMS (Non-Maximum Suppression)** to the raw output. The model does not include it. See the post-processing code below.

### Pose Model Output

| Field | Shape | Description |
|-------|-------|-------------|
| **Output 0** | `(1, 300, 18)` | Filtered detections with keypoints |

Format: 300 candidate detections × 18 values each. Each row is:
- `[0]` = x1 (left edge, pixel coords in model input space)
- `[1]` = y1 (top edge)
- `[2]` = x2 (right edge)
- `[3]` = y2 (bottom edge)
- `[4]` = objectness confidence
- `[5]` = class probability (≈0 for the photo class in single-class models)
- `[6, 7, 8]` = keypoint 0 (LL): x, y, confidence
- `[9, 10, 11]` = keypoint 1 (UL): x, y, confidence
- `[12, 13, 14]` = keypoint 2 (UR): x, y, confidence
- `[15, 16, 17]` = keypoint 3 (LR): x, y, confidence

> **The pose model output is already NMS-filtered.** It returns at most 300 detections sorted by confidence. Filter by `row[4] > your_confidence_threshold`.

### Keypoint Order

| Index | Name | Position |
|-------|------|----------|
| kp0 | LL | Lower-Left |
| kp1 | UL | Upper-Left |
| kp2 | UR | Upper-Right |
| kp3 | LR | Lower-Right |

---

## Project Setup

### Gradle Dependencies

```kotlin
// build.gradle.kts
dependencies {
    // ONNX Runtime
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // For JVM (non-Android) desktop apps, use:
    // implementation("com.microsoft.onnxruntime:onnxruntime:1.18.0")

    // Android Bitmap utilities (included in Android SDK)
    // No additional dependency needed
}
```

### Model Files

Copy both ONNX models into your assets:

```
app/src/main/assets/
├── detection_model.onnx    (10.2 MB)
└── pose_model.onnx         (10.0 MB)
```

---

## Implementation

### 1. Preprocessing

Both models share the same preprocessing. The key steps are:
- Resize to 640×640 while maintaining aspect ratio (letterbox)
- Convert to float32 RGB CHW format, normalized to [0, 1]

```kotlin
package com.example.photopose

import android.graphics.Bitmap
import android.graphics.Matrix

object ImagePreprocessor {
    const val INPUT_SIZE = 640

    /**
     * Preprocess a Bitmap for ONNX inference.
     * Returns a FloatBuffer in CHW RGB format, normalized to [0, 1].
     * Also returns the scale factors for mapping output coords back to original image.
     *
     * @return PreprocessedInput with float data and inverse-scale info
     */
    fun preprocess(bitmap: Bitmap): PreprocessedInput {
        val origWidth = bitmap.width
        val origHeight = bitmap.height

        // Calculate letterbox dimensions
        val scale = minOf(INPUT_SIZE.toFloat() / origWidth, INPUT_SIZE.toFloat() / origHeight)
        val scaledWidth = (origWidth * scale).toInt()
        val scaledHeight = (origHeight * scale).toInt()

        // Resize with letterboxing (pad to INPUT_SIZE × INPUT_SIZE)
        val resized = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        // Create padded image (gray padding = 114/255 ≈ 0.447)
        val padded = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        padded.eraseColor(android.graphics.Color.rgb(114, 114, 114))

        val canvas = android.graphics.Canvas(padded)
        val padX = (INPUT_SIZE - scaledWidth) / 2
        val padY = (INPUT_SIZE - scaledHeight) / 2
        canvas.drawBitmap(resized, padX.toFloat(), padY.toFloat(), null)

        // Convert to CHW float array normalized to [0, 1]
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        padded.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val channelSize = INPUT_SIZE * INPUT_SIZE
        val floatData = FloatArray(3 * channelSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            // Extract RGB (Android Bitmap is ARGB)
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // CHW format
            floatData[i] = r                          // R channel
            floatData[channelSize + i] = g            // G channel
            floatData[2 * channelSize + i] = b        // B channel
        }

        return PreprocessedInput(
            data = floatData,
            scaleX = 1.0f / scale,
            scaleY = 1.0f / scale,
            padX = padX,
            padY = padY,
            origWidth = origWidth,
            origHeight = origHeight
        )
    }

    data class PreprocessedInput(
        val data: FloatArray,
        val scaleX: Float,    // Multiply model output x-coords by this to get original image coords
        val scaleY: Float,    // Multiply model output y-coords by this to get original image coords
        val padX: Int,         // Horizontal padding added during letterboxing
        val padY: Int,         // Vertical padding added during letterboxing
        val origWidth: Int,
        val origHeight: Int
    )
}
```

### 2. Detection Model — Inference + NMS

The detection model outputs raw anchor predictions that **must be filtered with NMS**:

```kotlin
package com.example.photopose

import com.microsoft.onnxruntime.OnnxTensor
import com.microsoft.onnxruntime.OrtEnvironment
import com.microsoft.onnxruntime.OrtSession

class DetectionModel(modelPath: String) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val options = OrtSession.SessionOptions()
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.addConfigEntry("session.dynamic_option.enable", "1")

        session = env.createSession(modelPath, options)
        inputName = session.inputNames.iterator().next()
    }

    data class Detection(
        val xCenter: Float, val yCenter: Float,
        val width: Float, val height: Float,
        val confidence: Float
    ) {
        val x1: Float get() = xCenter - width / 2
        val y1: Float get() = xCenter - height / 2
        val x2: Float get() = xCenter + width / 2
        val y2: Float get() = yCenter + height / 2
    }

    /**
     * Run detection on preprocessed image data.
     * Output shape: (1, 5, 8400) where channel 4 is confidence.
     */
    fun detect(preprocessed: ImagePreprocessor.PreprocessedInput, confThreshold: Float = 0.5f): List<Detection> {
        val inputTensor = OnnxTensor.createTensor(
            env,
            preprocessed.data,
            longArrayOf(1, 3, ImagePreprocessor.INPUT_SIZE.toLong(), ImagePreprocessor.INPUT_SIZE.toLong())
        )

        val results = session.run(mapOf(inputName to inputTensor))
        val output = (results[0] as OnnxTensor).floatBuffer

        // Parse (1, 5, 8400) output
        val rawDetections = mutableListOf<Detection>()
        output.rewind()

        val numAnchors = 8400
        for (i in 0 until numAnchors) {
            val cx = output.get(i)                          // channel 0, anchor i
            val cy = output.get(numAnchors + i)             // channel 1, anchor i
            val w = output.get(2 * numAnchors + i)          // channel 2, anchor i
            val h = output.get(3 * numAnchors + i)          // channel 3, anchor i
            val conf = output.get(4 * numAnchors + i)        // channel 4, anchor i

            if (conf >= confThreshold) {
                rawDetections.add(Detection(cx, cy, w, h, conf))
            }
        }

        inputTensor.close()
        results.close()

        return applyNMS(rawDetections, iouThreshold = 0.45f)
    }

    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Detection>()

        for (det in sorted) {
            val overlaps = keep.any { existing ->
                calculateIoU(det, existing) > iouThreshold
            }
            if (!overlaps) keep.add(det)
        }
        return keep
    }

    private fun calculateIoU(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)

        if (x2 <= x1 || y2 <= y1) return 0f

        val intersection = (x2 - x1) * (y2 - y1)
        val union = a.width * a.height + b.width * b.height - intersection
        return intersection / union
    }

    override fun close() {
        session.close()
    }
}
```

### 3. Pose Model — Inference (No NMS Needed)

The pose model output is already filtered; just threshold by confidence:

```kotlin
package com.example.photopose

import com.microsoft.onnxruntime.OnnxTensor
import com.microsoft.onnxruntime.OrtEnvironment
import com.microsoft.onnxruntime.OrtSession

class PoseModel(modelPath: String) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val options = OrtSession.SessionOptions()
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

        session = env.createSession(modelPath, options)
        inputName = session.inputNames.iterator().next()
    }

    data class Keypoint(
        val x: Float, val y: Float, val confidence: Float
    )

    data class PoseDetection(
        val x1: Float, val y1: Float,
        val x2: Float, val y2: Float,
        val confidence: Float,
        val lowerLeft: Keypoint,    // kp0
        val upperLeft: Keypoint,    // kp1
        val upperRight: Keypoint,   // kp2
        val lowerRight: Keypoint    // kp3
    ) {
        val corners: List<Keypoint>
            get() = listOf(lowerLeft, upperLeft, upperRight, lowerRight)
    }

    /**
     * Run pose detection on preprocessed image data.
     * Output shape: (1, 300, 18) — already NMS-filtered.
     *
     * Each row layout:
     *   [0:4] = x1, y1, x2, y2 (xyxy bounding box)
     *   [4]   = objectness confidence
     *   [5]   = class probability
     *   [6:9] = kp0 (LL): x, y, confidence
     *   [9:12] = kp1 (UL): x, y, confidence
     *   [12:15] = kp2 (UR): x, y, confidence
     *   [15:18] = kp3 (LR): x, y, confidence
     */
    fun detect(preprocessed: ImagePreprocessor.PreprocessedInput, confThreshold: Float = 0.5f): List<PoseDetection> {
        val inputTensor = OnnxTensor.createTensor(
            env,
            preprocessed.data,
            longArrayOf(1, 3, ImagePreprocessor.INPUT_SIZE.toLong(), ImagePreprocessor.INPUT_SIZE.toLong())
        )

        val results = session.run(mapOf(inputName to inputTensor))
        val output = (results[0] as OnnxTensor).floatBuffer

        val detections = mutableListOf<PoseDetection>()
        output.rewind()

        // Parse (1, 300, 18) output
        val numDetections = 300
        val numValues = 18

        for (i in 0 until numDetections) {
            val offset = i * numValues
            val conf = output.get(offset + 4)

            if (conf < confThreshold) continue

            detections.add(PoseDetection(
                x1 = output.get(offset + 0),
                y1 = output.get(offset + 1),
                x2 = output.get(offset + 2),
                y2 = output.get(offset + 3),
                confidence = conf,
                lowerLeft = Keypoint(    // kp0
                    x = output.get(offset + 6),
                    y = output.get(offset + 7),
                    confidence = output.get(offset + 8)
                ),
                upperLeft = Keypoint(     // kp1
                    x = output.get(offset + 9),
                    y = output.get(offset + 10),
                    confidence = output.get(offset + 11)
                ),
                upperRight = Keypoint(    // kp2
                    x = output.get(offset + 12),
                    y = output.get(offset + 13),
                    confidence = output.get(offset + 14)
                ),
                lowerRight = Keypoint(   // kp3
                    x = output.get(offset + 15),
                    y = output.get(offset + 16),
                    confidence = output.get(offset + 17)
                )
            ))
        }

        inputTensor.close()
        results.close()

        return detections
    }

    override fun close() {
        session.close()
    }
}
```

### 4. Two-Model Pipeline

```kotlin
package com.example.photopose

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class DetectedPhoto(
    val confidence: Float,
    val boundingBox: DetectionModel.Detection,
    val corners: List<PoseModel.Keypoint>   // [LL, UL, UR, LR]
) {
    /** Calculate the skew angle in degrees (0 = perfectly rectangular). */
    fun skewAngle(): Float {
        val ll = corners[0]
        val lr = corners[3]
        val ul = corners[1]
        val ur = corners[2]
        val topWidth = distance(ll, lr)
        val bottomWidth = distance(ul, ur)
        val maxRatio = maxOf(topWidth, bottomWidth) / minOf(topWidth, bottomWidth)
        return if (maxRatio > 1.01f) {
            kotlin.math.acos(1f / maxRatio) * (180f / Math.PI.toFloat())
        } else 0f
    }

    private fun distance(a: PoseModel.Keypoint, b: PoseModel.Keypoint): Float {
        val dx = b.x - a.x
        val dy = b.y - a.y
        return sqrt(dx * dx + dy * dy)
    }
}

class PhotoScanPipeline(
    detectionModelPath: String,
    poseModelPath: String,
    private val confThreshold: Float = 0.5f
) : AutoCloseable {

    private val detectionModel = DetectionModel(detectionModelPath)
    private val poseModel = PoseModel(poseModelPath)

    /**
     * Detect photos in an image using the two-model pipeline.
     *
     * 1. Detection model finds bounding boxes
     * 2. Each region is cropped and fed to the pose model
     * 3. Corner keypoints are mapped back to original image coordinates
     */
    fun detectPhotos(bitmap: Bitmap): List<DetectedPhoto> {
        // Step 1: Run detection on full image
        val fullInput = ImagePreprocessor.preprocess(bitmap)
        val detections = detectionModel.detect(fullInput, confThreshold)

        if (detections.isEmpty()) return emptyList()

        val results = mutableListOf<DetectedPhoto>()

        for (det in detections) {
            // Step 2: Crop the detected region and run pose model on it
            val region = cropDetection(bitmap, det, fullInput)
            val regionInput = ImagePreprocessor.preprocess(region)

            val poseResults = poseModel.detect(regionInput, confThreshold)

            if (poseResults.isEmpty()) {
                // No pose found in this region — use bounding box only
                results.add(DetectedPhoto(
                    confidence = det.confidence,
                    boundingBox = det,
                    corners = emptyList()
                ))
                continue
            }

            // Take the highest-confidence pose detection
            val pose = poseResults.maxByOrNull { it.confidence }!!

            // Step 3: Map keypoints from 640×640 model space back to original image coords
            val mappedCorners = mapCornersToOriginal(
                pose, fullInput.scaleX, fullInput.scaleY, fullInput.padX, fullInput.padY
            )

            results.add(DetectedPhoto(
                confidence = minOf(det.confidence, pose.confidence),
                boundingBox = det,
                corners = mappedCorners
            ))
        }

        return results
    }

    /**
     * Detect photos using only the pose model (no detection step).
     * Simpler but may be slower for images with many objects.
     */
    fun detectPhotosPoseOnly(bitmap: Bitmap): List<DetectedPhoto> {
        val input = ImagePreprocessor.preprocess(bitmap)
        val poseResults = poseModel.detect(input, confThreshold)

        return poseResults.map { pose ->
            val mappedCorners = mapCornersToOriginal(
                pose, input.scaleX, input.scaleY, input.padX, input.padY
            )
            DetectedPhoto(
                confidence = pose.confidence,
                boundingBox = DetectionModel.Detection(
                    (pose.x1 + pose.x2) / 2f,  // xCenter from xyxy
                    (pose.y1 + pose.y2) / 2f,   // yCenter from xyxy
                    pose.x2 - pose.x1,           // width from xyxy
                    pose.y2 - pose.y1,            // height from xyxy
                    pose.confidence
                ),
                corners = mappedCorners
            )
        }
    }

    private fun cropDetection(
        bitmap: Bitmap, det: DetectionModel.Detection, input: ImagePreprocessor.PreprocessedInput
    ): Bitmap {
        // Convert model coords (640×640 letterboxed) back to original image coords
        val x1 = ((det.x1 - input.padX) * input.scaleX).toInt().coerceIn(0, bitmap.width)
        val y1 = ((det.y1 - input.padY) * input.scaleY).toInt().coerceIn(0, bitmap.height)
        val x2 = ((det.x2 - input.padX) * input.scaleX).toInt().coerceIn(0, bitmap.width)
        val y2 = ((det.y2 - input.padY) * input.scaleY).toInt().coerceIn(0, bitmap.height)

        // Add margin for better keypoint detection
        val margin = 20
        val cx = (x1 + x2) / 2
        val cy = (y1 + y2) / 2
        val w = minOf(x2 - x1 + margin * 2, bitmap.width)
        val h = minOf(y2 - y1 + margin * 2, bitmap.height)

        val cropX = maxOf(0, cx - w / 2)
        val cropY = maxOf(0, cy - h / 2)

        return Bitmap.createBitmap(bitmap, cropX, cropY,
            minOf(w, bitmap.width - cropX),
            minOf(h, bitmap.height - cropY))
    }

    private fun mapCornersToOriginal(
        pose: PoseModel.PoseDetection,
        scaleX: Float, scaleY: Float,
        padX: Int, padY: Int
    ): List<PoseModel.Keypoint> {
        // Model outputs are in 640×640 letterboxed space.
        // Map back: remove padding, then undo scaling.
        fun mapCoord(x: Float, y: Float): Pair<Float, Float> {
            return ((x - padX) * scaleX) to ((y - padY) * scaleY)
        }

        return pose.corners.map { kp ->
            val (origX, origY) = mapCoord(kp.x, kp.y)
            PoseModel.Keypoint(origX, origY, kp.confidence)
        }
    }

    override fun close() {
        detectionModel.close()
        poseModel.close()
    }
}
```

### 5. Perspective Extraction

Once you have the 4 corners (LL, UL, UR, LR), apply a perspective transform to extract a clean rectangular photo:

```kotlin
package com.example.photopose

import android.graphics.Bitmap
import android.graphics.Matrix
import kotlin.math.max
import kotlin.math.sqrt

object PerspectiveExtractor {

    /**
     * Extract a perspective-corrected photo from the original image.
     *
     * @param bitmap Original image
     * @param corners List of 4 keypoints: [LL, UL, UR, LR] in original image coordinates
     * @return A perspective-corrected Bitmap
     */
    fun extractPhoto(bitmap: Bitmap, corners: List<PoseModel.Keypoint>): Bitmap {
        require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }

        val ll = corners[0] // Lower-Left
        val ul = corners[1] // Upper-Left
        val ur = corners[2] // Upper-Right
        val lr = corners[3] // Lower-Right

        // Calculate output dimensions from edge lengths
        val topWidth = distance(ll, lr)
        val bottomWidth = distance(ul, ur)
        val leftHeight = distance(ll, ul)
        val rightHeight = distance(lr, ur)

        val outWidth = maxOf(topWidth, bottomWidth).toInt()
        val outHeight = maxOf(leftHeight, rightHeight).toInt()

        // Android Matrix can handle perspective transforms via Matrix.setPolyToPoly
        // Source: 4 corners in original image coords
        // Destination: rectangle corners
        val src = floatArrayOf(
            ll.x, ll.y,   // Lower-Left  → bottom-left of output
            ul.x, ul.y,   // Upper-Left  → top-left of output
            ur.x, ur.y,   // Upper-Right → top-right of output
            lr.x, lr.y    // Lower-Right → bottom-right of output
        )

        val dst = floatArrayOf(
            0f, outHeight.toFloat(),   // bottom-left
            0f, 0f,                     // top-left
            outWidth.toFloat(), 0f,     // top-right
            outWidth.toFloat(), outHeight.toFloat()  // bottom-right
        )

        val matrix = Matrix()
        matrix.setPolyToPoly(src, 0, dst, 0, 4)

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            .let { Bitmap.createBitmap(it, 0, 0, outWidth, outHeight) }
    }

    private fun distance(a: PoseModel.Keypoint, b: PoseModel.Keypoint): Float {
        val dx = b.x - a.x
        val dy = b.y - a.y
        return sqrt(dx * dx + dy * dy)
    }
}
```

---

## Complete Usage Example

```kotlin
// Initialize once (e.g., in Application.onCreate or Activity.onCreate)
val pipeline = PhotoScanPipeline(
    detectionModelPath = getModelPath("detection_model.onnx"),
    poseModelPath = getModelPath("pose_model.onnx"),
    confThreshold = 0.5f
)

// Process an image
val bitmap = BitmapFactory.decodeFile("/path/to/scanned_photo.jpg")

val detections = pipeline.detectPhotos(bitmap)

for ((i, photo) in detections.withIndex()) {
    println("Photo $i:")
    println("  Confidence: ${photo.confidence}")
    println("  Corners:")
    println("    LL: (${photo.corners[0].x}, ${photo.corners[0].y})")
    println("    UL: (${photo.corners[1].x}, ${photo.corners[1].y})")
    println("    UR: (${photo.corners[2].x}, ${photo.corners[2].y})")
    println("    LR: (${photo.corners[3].x}, ${photo.corners[3].y})")

    if (photo.corners.size == 4) {
        val extracted = PerspectiveExtractor.extractPhoto(bitmap, photo.corners)
        // Save extracted photo
        saveBitmap(extracted, "photo_$i.jpg")
    }
}

// Cleanup when done
pipeline.close()

// Helper: Copy model from assets to cache dir (required by ONNX Runtime)
private fun getModelPath(filename: String): String {
    val file = File(cacheDir, filename)
    if (!file.exists()) {
        assets.open("models/$filename").use { input ->
            file.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }
    return file.absolutePath
}
```

### Pose-Only (Simpler, No Detection Step)

```kotlin
val poseModel = PoseModel(getModelPath("pose_model.onnx"))
val input = ImagePreprocessor.preprocess(bitmap)
val poses = poseModel.detect(input, confThreshold = 0.5f)

// Map coordinates back to original image
poses.forEach { pose ->
    val origX = (pose.lowerLeft.x - input.padX) * input.scaleX
    val origY = (pose.lowerLeft.y - input.padY) * input.scaleY
    // ... etc for all keypoints
}

poseModel.close()
```

---

## Performance

### Single Inference

| Platform | Model | Input Size | Inference Time |
|----------|-------|-----------|----------------|
| Desktop CPU (i7) | Detection | 640×640 | ~30–50 ms |
| Desktop CPU (i7) | Pose | 640×640 | ~50–80 ms |
| Android (Pixel 6) | Detection | 640×640 | ~100–200 ms |
| Android (Pixel 6) | Pose | 640×640 | ~150–300 ms |
| Android + NNAPI | Detection | 640×640 | ~50–100 ms |
| Android + NNAPI | Pose | 640×640 | ~80–150 ms |

### Full Pipeline (4 photos, Desktop CPU)

| Pipeline | Sequential | 4 Threads | Speedup |
|----------|-----------|-----------|---------|
| Detect + pose (baseline) | ~700ms | ~400ms | 1.8× |
| + Corner refine (pose) | ~2,500ms | ~900ms | 2.8× |
| + Corner refine (detection) | ~1,900ms | ~700ms | 2.7× |

Corner refinement is **embarrassingly parallel** — all 4 corners × N photos are independent and can run simultaneously. With 4 threads, corner refinement adds only ~500ms to the baseline pipeline.

Tips for faster inference:
- Use `confThreshold = 0.5` (default) — higher values skip more detections
- For the two-model pipeline, limit detection to top-N results before running pose
- Corner refinement can be parallelized with a thread pool (4 threads recommended)
- Resize input to 320×320 for ~4× speed at some accuracy cost

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| All-zero outputs | Wrong preprocessing (BGR vs RGB, not normalized) | Ensure RGB + [0,1] normalization |
| Coordinates off by 2× | Missing letterbox padding correction | Subtract `padX/padY` then divide by `scale` |
| Keypoints in wrong order | Model expects LL/UL/UR/LR order | Verify `flip_idx: [3, 2, 1, 0]` was used in training |
| Too many detections | Low confidence threshold | Raise `confThreshold` (try 0.7) |
| Detections at wrong scale | Preprocessing mismatch | Ensure the same 640×640 letterboxing used during training |
| OOM on mobile | Full-resolution image passed to model | Resize to 640×640 before inference |
| `OrtException: No such file` | Model not extracted from assets | Copy ONNX files to cache dir first |

---

## Further Reading

- [ONNX Runtime Android Documentation](https://onnxruntime.ai/docs/reference/mobile/)
- [ONNX Runtime Kotlin API](https://onnxruntime.ai/docs/api/kotlin/)
- [Ultralytics YOLO Export Guide](https://docs.ultralytics.com/modes/export/)
- [Android Matrix.setPolyToPoly](https://developer.android.com/reference/android/graphics/Matrix#setPolyToPoly(float[],%20int,%20float[],%20int,%20int))