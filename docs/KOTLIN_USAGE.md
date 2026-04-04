# Kotlin Integration Guide

## Overview

This guide covers integrating the trained ONNX model into a Kotlin/Android application using:
- **ONNX Runtime** for model inference
- **BoofCV** for image processing and perspective transforms
- **JavaCV** for image loading

---

## Project Setup

### Gradle Configuration

```kotlin
// build.gradle.kts
plugins {
    kotlin("jvm") version "1.9.22"
    application
    
    // For Android
    id("com.android.application") version "8.2.0"
}

repositories {
    mavenCentral()
    google()
}

dependencies {
    // ONNX Runtime - Core
    implementation("org.onnxruntime:onnxruntime:1.17.0")
    
    // Platform-specific ONNX Runtime (pick one based on target)
    // For Android (armeabi-v7a, arm64-v8a, x86, x86_64)
    implementation("org.onnxruntime:onnxruntime-android:1.17.0")
    
    // For Linux/JVM
    // implementation("org.onnxruntime:onnxruntime-linux-x64-gpu:1.17.0")
    // implementation("org.onnxruntime:onnxruntime-linux-x64:1.17.0")
    
    // BoofCV for image processing
    implementation("boofcv:boofcv-core:0.40")
    implementation("boofcv:boofcv-android:0.40")
    
    // JavaCV for additional image loading
    implementation("org.bytedeco:javacv-platform:1.5.9")
    
    // Kotlin Coroutines (optional, for async processing)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}

// Android-specific
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }
    
    packagingOptions {
        exclude("META-INF/DEPENDENCIES")
        exclude("META-INF/LICENSE")
        exclude("META-INF/LICENSE.md")
        exclude("META-INF/LICENSE-notice.md")
    }
}
```

### Directory Structure

```
kotlin_integration/
├── src/main/kotlin/
│   ├── com/example/photopose/
│   │   ├── PhotoCornerDetector.kt    # Main detector class
│   │   ├── ModelLoader.kt           # ONNX model loading
│   │   ├── ImageProcessor.kt         # Pre/post processing
│   │   ├── PerspectiveTransform.kt   # Homography transforms
│   │   └── data/
│   │       └── DetectedPhoto.kt     # Data classes
│   └── Main.kt                      # Entry point
├── src/main/assets/
│   └── models/
│       └── photo-corner-detector.onnx  # Your trained model
├── src/test/kotlin/
│   └── PhotoCornerDetectorTest.kt
└── build.gradle.kts
```

---

## Core Implementation

### 1. Data Classes

```kotlin
package com.example.photopose

import georegression.struct.point.Point2D_F32
import java.awt.geom.Point2D

/**
 * Represents a detected photo with its corner keypoints.
 */
data class DetectedPhoto(
    val confidence: Float,
    val corners: List<Point2D_F32>,
    val boundingBox: BoundingBox
) {
    /**
     * Get corners as a list of Point2D for Java compatibility.
     */
    fun cornersAsPoint2D(): List<Point2D> = corners.map { Point2D.Float(it.x, it.y) }
    
    /**
     * Calculate the perspective skew angle.
     * Returns angle in degrees from 0 (perfect rectangle) to 90 (maximum skew).
     */
    fun calculateSkewAngle(): Float {
        val topWidth = distance(corners[0], corners[1])
        val bottomWidth = distance(corners[2], corners[3])
        val leftHeight = distance(corners[0], corners[3])
        val rightHeight = distance(corners[1], corners[2])
        
        val widthRatio = maxOf(topWidth, bottomWidth) / minOf(topWidth, bottomWidth)
        val heightRatio = maxOf(leftHeight, rightHeight) / minOf(leftHeight, rightHeight)
        
        // Angle based on aspect ratio distortion
        val maxRatio = maxOf(widthRatio, heightRatio)
        return if (maxRatio > 1.1f) {
            kotlin.math.acos(1f / maxRatio) * (180f / Math.PI.toFloat())
        } else {
            0f
        }
    }
    
    /**
     * Check if perspective correction is needed.
     */
    fun needsPerspectiveCorrection(threshold: Float = 5f): Boolean {
        return calculateSkewAngle() > threshold
    }
    
    private fun distance(p1: Point2D_F32, p2: Point2D_F32): Float {
        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}

/**
 * Bounding box in image coordinates.
 */
data class BoundingBox(
    val x: Float, val y: Float,
    val width: Float, val height: Float
) {
    fun contains(point: Point2D_F32): Boolean {
        return point.x >= x && point.x <= x + width &&
               point.y >= y && point.y <= y + height
    }
}

/**
 * Configuration for the detector.
 */
data class DetectorConfig(
    val confidenceThreshold: Float = 0.5f,
    val nmsThreshold: Float = 0.45f,
    val perspectiveCorrectionThreshold: Float = 5f,
    val outputImageQuality: Int = 95
)
```

### 2. ONNX Model Loader

```kotlin
package com.example.photopose

import org.onnxruntime.OnnxRuntime
import org.onnxruntime.OnnxTensor
import java.nio.FloatBuffer

/**
 * Handles ONNX model loading and inference.
 */
class ModelLoader(private val modelPath: String) : AutoCloseable {
    
    private val env = OnnxRuntime.getEnv()
    private val session: org.onnxruntime.Session
    
    val inputName: String
    val outputName: String
    
    init {
        val options = org.onnxruntime.SessionOptions().apply {
            // Enable optimization
            graphOptimizationLevel = org.onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            // Enable parallel execution
            intraOpNumThreads = 4
            interOpNumThreads = 2
        }
        
        session = env.createSession(modelPath, options)
        
        // Get input/output names
        inputName = session.inputNames.first()
        outputName = session.outputNames.first()
    }
    
    /**
     * Run inference on preprocessed input.
     * 
     * @param inputData Float array in CHW format (1, 3, 640, 640)
     * @return Model output as FloatBuffer
     */
    fun runInference(inputData: FloatArray): FloatBuffer {
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        
        val outputs = session.run(mapOf(inputName to inputTensor))
        val output = (outputs[0] as OnnxTensor).floatBuffer
        
        inputTensor.close()
        
        return output
    }
    
    override fun close() {
        session.close()
        env.close()
    }
    
    companion object {
        const val INPUT_SIZE = 640
    }
}
```

### 3. Image Preprocessing

```kotlin
package com.example.photopose

import boofcv.abst.filter.BlurFilter
import boofcv.alg.filter.blur.GBlurImageOps
import boofcv.factory.filter.blur.FactoryBlurFilter
import boofcv.struct.image.GrayF32
import boofcv.struct.image.InterleavedF32
import boofcv.struct.image.InterleavedU8
import java.awt.image.BufferedImage
import java.awt.image.DataBufferFloat

/**
 * Handles image preprocessing for ONNX input.
 */
class ImageProcessor(private val targetSize: Int = 640) {
    
    /**
     * Preprocess image for ONNX inference.
     * Converts to CHW float format normalized to [0, 1].
     */
    fun preprocess(image: BufferedImage): FloatArray {
        // Resize to target size
        val resized = BufferedImage(targetSize, targetSize, BufferedImage.TYPE_3BYTE_BGR)
        val graphics = resized.createGraphics()
        graphics.setRenderingHint(
            java.awt.RenderingHints.KEY_INTERPOLATION,
            java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR
        )
        graphics.drawImage(
            image.getScaledInstance(targetSize, targetSize, java.awt.Image.SCALE_SMOOTH),
            0, 0, null
        )
        graphics.dispose()
        
        // Convert to float CHW format
        val pixels = FloatArray(3 * targetSize * targetSize)
        val raster = resized.getRaster()
        val pixelBuffer = FloatArray(3)
        
        for (y in 0 until targetSize) {
            for (x in 0 until targetSize) {
                val pixelIndex = y * targetSize + x
                raster.getPixel(x, y, pixelBuffer)
                
                // BGR to RGB (or just use the format appropriately)
                pixels[pixelIndex] = pixelBuffer[0] / 255.0f              // B
                pixels[pixelIndex + targetSize * targetSize] = pixelBuffer[1] / 255.0f  // G
                pixels[pixelIndex + 2 * targetSize * targetSize] = pixelBuffer[2] / 255.0f  // R
            }
        }
        
        return pixels
    }
    
    /**
     * Convert BoofCV image to float array.
     */
    fun <T> preprocessBoofCV(image: T): FloatArray where T : boofcv.struct.image.ImageBase<T> {
        require(image.numChannels == 3) { "Expected 3 channel image" }
        
        val pixels = FloatArray(3 * targetSize * targetSize)
        
        // Scale and copy
        val scaled = image.createNew(targetSize, targetSize) as T
        GBlurImageOps.scaleImage(image, scaled, InterpolationType.BILINEAR)
        
        // Convert to float CHW
        val img = scaled as InterleavedF32
        for (y in 0 until targetSize) {
            for (x in 0 until targetSize) {
                val pixelIndex = y * targetSize + x
                val r = img.getBand(0).get(x, y) / 255f
                val g = img.getBand(1).get(x, y) / 255f
                val b = img.getBand(2).get(x, y) / 255f
                
                pixels[pixelIndex] = r
                pixels[pixelIndex + targetSize * targetSize] = g
                pixels[pixelIndex + 2 * targetSize * targetSize] = b
            }
        }
        
        return pixels
    }
}
```

### 4. Output Post-processing

```kotlin
package com.example.photopose

import georegression.struct.point.Point2D_F32
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Post-processes model output to extract keypoints.
 */
class OutputProcessor(
    private val confidenceThreshold: Float = 0.5f,
    private val nmsThreshold: Float = 0.45f
) {
    
    /**
     * Extract detections from YOLO pose output.
     * 
     * Output format: (batch, 4 + num_classes + keypoint_dims, num_predictions)
     * For pose with 4 keypoints: (1, 4 + 1 + 12, 8400) = (1, 17, 8400)
     * Keypoint dims = 4 keypoints * 3 values (x, y, visibility)
     */
    fun postprocess(
        output: FloatBuffer,
        origWidth: Int,
        origHeight: Int,
        outputShape: LongArray
    ): List<DetectedPhoto> {
        val detections = mutableListOf<DetectedPhoto>()
        
        // Parse output shape
        // Expected: [1, 4 + num_classes + (num_keypoints * 3), num_predictions]
        val numPredictions = outputShape[2].toInt()
        
        // Parse predictions
        val stride = 17  // 4 (box) + 1 (class) + 12 (4 keypoints * 3)
        
        for (i in 0 until numPredictions) {
            val offset = i * stride
            
            // Extract box (center x, center y, width, height in normalized coords)
            val cx = output.get(offset)
            val cy = output.get(offset + 1)
            val w = output.get(offset + 2)
            val h = output.get(offset + 3)
            
            // Extract class confidence (assuming single class = photo)
            val confidence = output.get(offset + 4)
            
            if (confidence < confidenceThreshold) continue
            
            // Extract keypoints
            val keypointOffset = offset + 5
            val corners = mutableListOf<Point2D_F32>()
            
            for (k in 0 in 0 until 4) {
                val kx = output.get(keypointOffset + k * 3)
                val ky = output.get(keypointOffset + k * 3 + 1)
                val visibility = output.get(keypointOffset + k * 3 + 2)
                
                if (visibility > 0) {
                    // Scale from [0, 1] to original image coordinates
                    corners.add(Point2D_F32(kx * origWidth, ky * origHeight))
                }
            }
            
            if (corners.size == 4) {
                val bbox = BoundingBox(
                    x = (cx - w / 2) * origWidth,
                    y = (cy - h / 2) * origHeight,
                    width = w * origWidth,
                    height = h * origHeight
                )
                
                detections.add(DetectedPhoto(confidence, corners, bbox))
            }
        }
        
        // Apply NMS
        return applyNMS(detections)
    }
    
    /**
     * Apply Non-Maximum Suppression to remove overlapping detections.
     */
    private fun applyNMS(detections: List<DetectedPhoto>): List<DetectedPhoto> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort by confidence (descending)
        val sorted = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<DetectedPhoto>()
        
        for (det in sorted) {
            val overlaps = keep.any { existing ->
                calculateIoU(det.boundingBox, existing.boundingBox) > nmsThreshold
            }
            
            if (!overlaps) {
                keep.add(det)
            }
        }
        
        return keep
    }
    
    /**
     * Calculate Intersection over Union for two bounding boxes.
     */
    private fun calculateIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = max(a.x, b.x)
        val y1 = max(a.y, b.y)
        val x2 = min(a.x + a.width, b.x + b.width)
        val y2 = min(a.y + a.height, b.y + b.height)
        
        if (x2 < x1 || y2 < y1) return 0f
        
        val intersection = (x2 - x1) * (y2 - y1)
        val union = a.width * a.height + b.width * b.height - intersection
        
        return intersection / union
    }
}
```

### 5. Perspective Transform

```kotlin
package com.example.photopose

import boofcv.alg.distort.DistortImageOps
import boofcv.alg.interpolate.InterpolateType
import boofcv.factory.interpolate.FactoryInterpolation
import boofcv.struct.border.BorderType
import boofcv.struct.image.GrayF32
import boofcv.struct.image.InterleavedF32
import georegression.alg.lines.RefineDistanceB点到线段
import georegression.struct.homography.ModelHomography2D_F32
import georegression.struct.point.Point2D_F32
import georegression.convert.ConvertFunctions
import georegression.fitting.HomographyFit
import java.awt.image.BufferedImage

/**
 * Handles perspective transformation using homography estimation.
 */
class PerspectiveTransformer {
    
    /**
     * Apply perspective transform to extract a rectangular photo from detected corners.
     */
    fun extractPhoto(
        image: BufferedImage,
        corners: List<Point2D_F32>
    ): BufferedImage {
        require(corners.size == 4) { "Expected exactly 4 corners" }
        
        // Calculate output dimensions
        val (outputWidth, outputHeight) = calculateOutputSize(corners)
        
        // Estimate homography from source to destination rectangle
        val homography = estimateHomography(corners, outputWidth, outputHeight)
        
        // Apply transform
        return applyHomography(image, homography, outputWidth, outputHeight)
    }
    
    /**
     * Calculate optimal output size based on detected corners.
     */
    private fun calculateOutputSize(corners: List<Point2D_F32>): Pair<Int, Int> {
        // Average of parallel edges
        val topWidth = distance(corners[0], corners[1])
        val bottomWidth = distance(corners[2], corners[3])
        val leftHeight = distance(corners[0], corners[3])
        val rightHeight = distance(corners[1], corners[2])
        
        val avgWidth = (topWidth + bottomWidth) / 2
        val avgHeight = (leftHeight + rightHeight) / 2
        
        return Pair(avgWidth.toInt(), avgHeight.toInt())
    }
    
    /**
     * Estimate homography matrix using Direct Linear Transform.
     */
    private fun estimateHomography(
        srcCorners: List<Point2D_F32>,
        outWidth: Int,
        outHeight: Int
    ): ModelHomography2D_F32 {
        // Destination corners (rectangle)
        val dstCorners = listOf(
            Point2D_F32(0f, 0f),
            Point2D_F32(outWidth.toFloat(), 0f),
            Point2D_F32(outWidth.toFloat(), outHeight.toFloat()),
            Point2D_F32(0f, outHeight.toFloat())
        )
        
        // Use georegression library for homography estimation
        val estimator = HomographyFit()
        
        // Convert to Point2D_F64 for the estimator
        val src64 = srcCorners.map { Point2D_F64(it.x.toDouble(), it.y.toDouble()) }
        val dst64 = dstCorners.map { Point2D_F64(it.x.toDouble(), it.y.toDouble()) }
        
        // Fit homography
        val fitModel = ModelHomography2D_F32()
        estimator.fit(
            src64.map { arrayOf(it.x, it.y, 1.0) }.toFloatArray(),
            dst64.map { arrayOf(it.x, it.y, 1.0) }.toFloatArray(),
            fitModel
        )
        
        return fitModel
    }
    
    /**
     * Apply homography transformation to an image.
     */
    private fun applyHomography(
        image: BufferedImage,
        homography: ModelHomography2D_F32,
        outWidth: Int,
        outHeight: Int
    ): BufferedImage {
        // Convert to BoofCV format
        val src = boofcv.io.image.ConvertBufferedImage.convertFromSingle(
            image, InterleavedF32::class.java
        )
        
        // Create destination image
        val dst = InterleavedF32(outWidth, outHeight, 3)
        
        // Create interpolator
        val interpolator = FactoryInterpolation.createBiLinearF32(
            InterleavedF32::class.java, BorderType.ZERO
        )
        
        // Apply distortion
        DistortImageOps.distort(src, dst, homography, interpolator)
        
        // Convert back to BufferedImage
        return boofcv.io.image.ConvertBufferedImage.convertTo_F32(dst, null)
    }
    
    /**
     * Calculate Euclidean distance between two points.
     */
    private fun distance(p1: Point2D_F32, p2: Point2D_F32): Float {
        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}
```

### 6. Main Detector Class

```kotlin
package com.example.photopose

import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import javax.imageio.ImageIO

/**
 * Main class for photo corner detection and extraction.
 */
class PhotoCornerDetector(
    private val modelPath: String,
    private val config: DetectorConfig = DetectorConfig()
) : AutoCloseable {
    
    private val modelLoader: ModelLoader
    private val imageProcessor: ImageProcessor
    private val outputProcessor: OutputProcessor
    private val perspectiveTransformer: PerspectiveTransformer
    
    init {
        modelLoader = ModelLoader(modelPath)
        imageProcessor = ImageProcessor()
        outputProcessor = OutputProcessor(
            confidenceThreshold = config.confidenceThreshold,
            nmsThreshold = config.nmsThreshold
        )
        perspectiveTransformer = PerspectiveTransformer()
    }
    
    /**
     * Detect photos in an image.
     * 
     * @param imagePath Path to the input image
     * @return List of detected photos with corner keypoints
     */
    fun detect(imagePath: String): List<DetectedPhoto> {
        val image = ImageIO.read(File(imagePath))
        return detect(image)
    }
    
    /**
     * Detect photos in a BufferedImage.
     * 
     * @param image Input image
     * @return List of detected photos with corner keypoints
     */
    fun detect(image: BufferedImage): List<DetectedPhoto> {
        // Preprocess
        val inputData = imageProcessor.preprocess(image)
        
        // Run inference
        val output = modelLoader.runInference(inputData)
        
        // Post-process
        return outputProcessor.postprocess(
            output,
            image.width,
            image.height,
            longArrayOf(1, 17, 8400)  // Output shape
        )
    }
    
    /**
     * Detect and extract photos from an image.
     * 
     * @param imagePath Path to input image
     * @param outputDir Directory to save extracted photos
     * @param applyPerspectiveCorrection Apply perspective transform to extracted photos
     * @return List of extracted photo information
     */
    fun extractPhotos(
        imagePath: String,
        outputDir: String,
        applyPerspectiveCorrection: Boolean = true
    ): List<ExtractedPhoto> {
        val image = ImageIO.read(File(imagePath))
        return extractPhotos(image, outputDir, applyPerspectiveCorrection)
    }
    
    /**
     * Detect and extract photos from a BufferedImage.
     */
    fun extractPhotos(
        image: BufferedImage,
        outputDir: String,
        applyPerspectiveCorrection: Boolean = true
    ): List<ExtractedPhoto> {
        val detections = detect(image)
        val dir = File(outputDir)
        dir.mkdirs()
        
        val extracted = mutableListOf<ExtractedPhoto>()
        
        detections.forEachIndexed { index, photo ->
            val needsCorrection = photo.needsPerspectiveCorrection(
                config.perspectiveCorrectionThreshold
            )
            
            val extractedImage = if (applyPerspectiveCorrection && needsCorrection) {
                perspectiveTransformer.extractPhoto(image, photo.corners)
            } else {
                // Just crop to bounding box
                cropToBoundingBox(image, photo.boundingBox)
            }
            
            // Save
            val outputFile = File(dir, "photo_${index}.jpg")
            ImageIO.write(extractedImage, "jpg", outputFile)
            
            extracted.add(ExtractedPhoto(
                index = index,
                confidence = photo.confidence,
                skewAngle = photo.calculateSkewAngle(),
                corrected = needsCorrection && applyPerspectiveCorrection,
                outputPath = outputFile.absolutePath,
                corners = photo.corners
            ))
        }
        
        return extracted
    }
    
    /**
     * Simple bounding box crop without perspective correction.
     */
    private fun cropToBoundingBox(image: BufferedImage, bbox: BoundingBox): BufferedImage {
        val x = bbox.x.toInt().coerceIn(0, image.width - 1)
        val y = bbox.y.toInt().coerceIn(0, image.height - 1)
        val w = bbox.width.toInt().coerceIn(1, image.width - x)
        val h = bbox.height.toInt().coerceIn(1, image.height - y)
        
        return image.getSubimage(x, y, w, h)
    }
    
    override fun close() {
        modelLoader.close()
    }
}

/**
 * Information about an extracted photo.
 */
data class ExtractedPhoto(
    val index: Int,
    val confidence: Float,
    val skewAngle: Float,
    val corrected: Boolean,
    val outputPath: String,
    val corners: List<Point2D_F32>
)
```

---

## Usage Examples

### Basic Detection

```kotlin
fun main() {
    // Initialize detector
    PhotoCornerDetector(
        modelPath = "models/photo-corner-detector.onnx",
        config = DetectorConfig(
            confidenceThreshold = 0.5f,
            perspectiveCorrectionThreshold = 5f
        )
    ).use { detector ->
        
        // Detect photos in an image
        val detections = detector.detect("input/scanned_photos.jpg")
        
        println("Found ${detections.size} photos")
        
        for (photo in detections) {
            println("Photo:")
            println("  Confidence: ${photo.confidence}")
            println("  Skew angle: ${photo.calculateSkewAngle()}°")
            println("  Needs correction: ${photo.needsPerspectiveCorrection()}")
            
            // Print corner coordinates
            photo.corners.forEachIndexed { i, corner ->
                println("  Corner $i: (${corner.x}, ${corner.y})")
            }
        }
    }
}
```

### Full Extraction Pipeline

```kotlin
fun main() {
    PhotoCornerDetector(
        modelPath = "models/photo-corner-detector.onnx"
    ).use { detector ->
        
        // Detect and extract all photos
        val extracted = detector.extractPhotos(
            imagePath = "input/scanned_photos.jpg",
            outputDir = "output/extracted",
            applyPerspectiveCorrection = true
        )
        
        println("Extracted ${extracted.size} photos:")
        
        for (photo in extracted) {
            println("Photo ${photo.index}:")
            println("  Confidence: ${photo.confidence}")
            println("  Skew angle: ${photo.skewAngle}°")
            println("  Was corrected: ${photo.corrected}")
            println("  Saved to: ${photo.outputPath}")
        }
    }
}
```

### Android Integration

```kotlin
// AndroidActivity.kt
class PhotoScanActivity : AppCompatActivity() {
    
    private lateinit var detector: PhotoCornerDetector
    private lateinit var binding: ActivityPhotoScanBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPhotoScanBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Initialize detector from assets
        val modelFile = File(cacheDir, "photo-corner-detector.onnx")
        assets.open("models/photo-corner-detector.onnx").use { input ->
            modelFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        
        detector = PhotoCornerDetector(modelFile.absolutePath)
        
        // Set up image picker
        pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
    
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let { processImage(it) }
    }
    
    private fun processImage(uri: Uri) {
        val inputStream = contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        // Convert Bitmap to BufferedImage
        val image = bitmap.toBufferedImage()
        
        // Detect and extract
        lifecycleScope.launch(Dispatchers.Default) {
            val extracted = detector.extractPhotos(
                image = image,
                outputDir = getExternalFilesDir("extracted")!!.absolutePath,
                applyPerspectiveCorrection = true
            )
            
            withContext(Dispatchers.Main) {
                // Display results
                showResults(extracted)
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        detector.close()
    }
    
    private fun Bitmap.toBufferedImage(): BufferedImage {
        val width = width
        val height = height
        val pixels = IntArray(width * height)
        getPixels(pixels, 0, width, 0, 0, width, height)
        
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        image.setRGB(0, 0, width, height, pixels, 0, width)
        return image
    }
}
```

---

## Advanced Topics

### Custom Output Shape

If your model has a different output shape, adjust the postprocessing:

```kotlin
// Check your model output shape
val outputInfo = session.outputNames.map { name ->
    val info = session.getOutput(name)
    println("$name: ${info.type} shape=${info.shape}")
}

// Adjust postprocess call
val detections = outputProcessor.postprocess(
    output,
    origWidth,
    origHeight,
    // Your model's actual output shape
    longArrayOf(1, 17, 8400)
)
```

### Using TensorRT (NVIDIA GPU)

For NVIDIA GPU acceleration:

```kotlin
// Add TensorRT ONNX Runtime dependency
dependencies {
    implementation("org.onnxruntime:onnxruntime-tensorrt:1.17.0")
}

// Session options for TensorRT
val options = SessionOptions().apply {
    graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
    // TensorRT will be used automatically if available
}
```

### Using Core ML (Apple Silicon)

```kotlin
dependencies {
    implementation("org.onnxruntime:onnxruntime-coreml:1.17.0")
}

// Session options for Core ML
val options = SessionOptions().apply {
    graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
}
```

---

## Troubleshooting

### Issue: ONNX Runtime Can't Find Model

```kotlin
// Check model file exists
val file = File(modelPath)
if (!file.exists()) {
    throw FileNotFoundException("Model not found: $modelPath")
}

// Check file is readable
if (!file.canRead()) {
    throw SecurityException("Cannot read model file: $modelPath")
}
```

### Issue: Image Dimension Errors

```kotlin
// Ensure input is valid
require(image.width > 0 && image.height > 0) { "Invalid image dimensions" }
require(image.width <= 4096 && image.height <= 4096) { "Image too large" }
```

### Issue: Out of Memory on Large Images

```kotlin
// Downscale large images before processing
fun preprocessForLargeImage(image: BufferedImage, maxDim: Int = 2048): BufferedImage {
    val scale = minOf(maxDim.toFloat() / image.width, maxDim.toFloat() / image.height)
    if (scale >= 1f) return image
    
    val newWidth = (image.width * scale).toInt()
    val newHeight = (image.height * scale).toInt()
    
    val scaled = BufferedImage(newWidth, newHeight, image.type)
    val g = scaled.createGraphics()
    g.drawImage(image, 0, 0, newWidth, newHeight, null)
    g.dispose()
    
    return scaled
}
```

### Issue: Slow Inference

1. Use GPU execution provider:
   ```kotlin
   val options = SessionOptions().apply {
       // Try CUDA, then Core ML, then CPU
   }
   ```

2. Reduce input size:
   ```kotlin
   val imageProcessor = ImageProcessor(targetSize = 320)  // Instead of 640
   ```

3. Use a smaller model (yolo26n instead of yolo26m)

---

## Performance Benchmarks

Expected inference times on different devices:

| Device | Input Size | Time per Image |
|--------|------------|----------------|
| Desktop (CPU) | 640x640 | ~50-100ms |
| Desktop (GPU) | 640x640 | ~10-20ms |
| Android (Pixel 6) | 640x640 | ~100-200ms |
| Android (Pixel 6 + NNAPI) | 640x640 | ~50-100ms |
| iPhone 14 | 640x640 | ~30-60ms |

---

## Further Reading

- [ONNX Runtime Kotlin API](https://onnxruntime.ai/docs/api/kotlin/)
- [BoofCV Homography Documentation](https://boofcv.org/)
- [Ultralytics YOLO Export](https://docs.ultralytics.com/modes/export/)
- [georegression Library](https://github.com/lessthanoptimal/georegression)
