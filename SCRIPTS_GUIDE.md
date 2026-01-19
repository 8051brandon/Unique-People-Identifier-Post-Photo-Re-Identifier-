# Scripts Guide

This document explains each script in the repository, what it does, and when to use it.

## 📹 Part 1: Data Collection

### `camera_capture.py` - Live Camera Tracking & Image Capture

**Purpose:** Capture visitor images during an event using live webcam(s) with real-time person tracking.

**When to use:**
- ✅ **During the event** to collect visitor images
- ✅ When you have access to camera(s) at the venue
- ✅ Want automated person detection and tracking
- ✅ Need to capture multiple angles of same person

**What it does:**
1. Opens one or more webcams
2. Uses YOLOv8 to detect people in real-time
3. Tracks each person across frames
4. Automatically captures and saves up to 5 reference images per visitor
5. Compares new detections against existing visitors
6. Displays live count and tracking visualization

**Features:**
- ✅ **Multi-camera support** - Use 2+ cameras simultaneously
- ✅ **Smart tracking** - Follows people across frames
- ✅ **Multi-reference capture** - Saves 5 images per person from different angles
- ✅ **Real-time matching** - Identifies returning visitors
- ✅ **Adjustable threshold** - Tune sensitivity on-the-fly
- ✅ **Debug mode** - See matching scores and track IDs

**Usage:**
```bash
python camera_capture.py
```

**Interactive controls:**
```
Q: Quit and save
R: Reset visitor database
D: Toggle debug mode (shows matching details)
+/-: Adjust similarity threshold
```

**Configuration:**
Edit in the script:
```python
# Line 18
self.similarity_threshold = 0.70  # Matching threshold (0-1)
self.max_images_per_visitor = 5   # Images to save per person
```

**Output:**
- `visitor_database/` - Folder with captured images
  - `visitor_1_img0.jpg` through `visitor_1_img4.jpg`
  - `visitor_2_img0.jpg` through `visitor_2_img4.jpg`
  - etc.
- `visitor_log.json` - Visitor metadata and timestamps

**Requirements:**
```bash
pip install ultralytics opencv-python numpy
```

**Camera Selection:**
- Single camera: Auto-selected
- Multiple cameras: Choose which ones to use interactively
- Supports Windows, Mac, Linux camera systems

**How It Works:**
1. **Detection:** YOLOv8 finds people in each frame
2. **Feature Extraction:** Calculates color histograms, spatial features, edge features
3. **Matching:** Compares against existing visitor database
4. **Decision:** If similarity > threshold → existing visitor, else → new visitor
5. **Capture:** Saves image if fewer than 5 references exist

**Performance:**
- **FPS:** 15-30 depending on camera resolution and CPU
- **Latency:** Real-time (~30-60ms per frame)
- **Memory:** ~500 MB

**Your Use Case:**
```
Event: 2-day open house
Cameras: Set up at entrance/booths
Duration: 8+ hours per day
Result: 32,330 captured images → 1,700 unique visitors
```

**Troubleshooting:**
- **No camera detected:** Check permissions, try different camera index
- **Slow FPS:** Reduce resolution or use faster computer
- **Too many/few visitors:** Adjust similarity threshold with +/- keys
- **YOLO model download:** First run downloads ~6MB model automatically

**Time:** Runs continuously during event

---

## 🎯 Part 2: Unique Visitor Analysis

### 1. `diagnostic_reid.py` - Find Optimal Threshold

**Purpose:** Analyze your dataset to determine the best threshold value.

**When to use:** 
- ✅ First time running on a new dataset
- ✅ When unsure what threshold to use
- ✅ When results seem wrong

**What it does:**
1. Samples 500 random images from your dataset
2. Extracts features from each image
3. Calculates pairwise distances
4. Analyzes distance distribution
5. Recommends optimal threshold values
6. Generates visualization graph

**Usage:**
```bash
python diagnostic_reid.py
```

**Configuration:**
Edit the script to set your image folder:
```python
IMAGE_FOLDER = "path/to/your/images"
SAMPLE_SIZE = 500  # Number of images to analyze
```

**Output:**
- `distance_analysis.png` - Distribution graph
- Console output with recommended thresholds

**Example Output:**
```
SUGGESTED THRESHOLDS:
VERY STRICT (10th percentile):  0.18
  → Use if you have MANY duplicates of same person
  
STRICT (25th percentile):       0.22
  → Recommended starting point (BEST for most cases)
  
MODERATE (50th percentile):     0.35
  → Use if people look very different
```

**Time:** ~5-10 minutes for 500 images

---

### 2. `memory_efficient_reid.py` - Main Re-ID System ⭐

**Purpose:** Count unique visitors from large datasets (1000+ images).

**When to use:**
- ✅ Production use for large datasets
- ✅ When you have 1000+ images
- ✅ After finding optimal threshold with diagnostic script
- ✅ **This is the main script you used for 32,330 images**

**What it does:**
1. Extracts features from all images
2. Uses incremental clustering (memory-efficient)
3. Groups similar images as same person
4. Generates comprehensive report
5. Creates sample visualization

**Features:**
- ✅ Memory-efficient (works on standard laptops)
- ✅ Handles 30K+ images without crashing
- ✅ Progress tracking
- ✅ Batch processing

**Usage:**
```bash
python memory_efficient_reid.py
```

**Configuration:**
```python
IMAGE_FOLDER = "path/to/your/images"
SIMILARITY_THRESHOLD = 0.1354  # Adjust based on diagnostic results
```

**Output:**
- `memory_efficient_report.json` - Full results with all clusters
- `sample_viz.jpg` - Visual grid showing sample clusters

**Time:** 
- 32K images: ~45-60 minutes
- 10K images: ~15-20 minutes
- 1K images: ~3-5 minutes

**Your Results:**
```
Total images:           32,330
Unique people:          1,700
Threshold:              0.1354
Avg images per person:  19.0
```

---

### 3. `reliability_checker.py` - Validate Results

**Purpose:** Check if your Re-ID results are accurate.

**When to use:**
- ✅ After running main Re-ID system
- ✅ To verify clustering quality
- ✅ Before finalizing results
- ✅ When results seem suspicious

**What it does:**
1. Loads results from `memory_efficient_report.json`
2. Analyzes cluster size distribution
3. Detects suspicious patterns (over/under-grouping)
4. Creates visual samples for manual verification
5. Recommends threshold adjustments

**Usage:**
```bash
python reliability_checker.py
```

**Prerequisites:** Must run `memory_efficient_reid.py` first

**Output:**
- `cluster_distribution.png` - Distribution graphs
- `reliability_check/` folder - Visual verification samples
- Console recommendations

**What to check:**
1. **Distribution graph** - Should look reasonable (not all same size)
2. **Visual samples** - Do images in each grid show same person?
3. **Statistics** - Compare to expected attendance

**Red Flags:**
- ⚠️ One cluster with >10% of all images
- ⚠️ >70% singletons (people with only 1 image)
- ⚠️ Visual samples show different people in same cluster

**Time:** ~2-5 minutes

---

## 🔧 Alternative Scripts

### 4. `lightweight_visitor_counter.py` - Simple Detection

**Purpose:** Quick people detection without Re-ID.

**When to use:**
- ✅ Just want to count detections (not unique people)
- ✅ Testing/prototyping
- ✅ Don't want to download large models
- ✅ Small datasets (<1000 images)

**What it does:**
1. Detects people using OpenCV HOG detector
2. Extracts simple features (color histograms)
3. Basic clustering with DBSCAN
4. Generates report and visualization

**Limitations:**
- ❌ Less accurate than main system
- ❌ Not optimized for large datasets
- ❌ Simple features (no deep learning)

**Usage:**
```bash
python lightweight_visitor_counter.py
```

**Configuration:**
```python
IMAGE_FOLDER = "camera_images"
SIMILARITY_THRESHOLD = 0.6  # Higher than main system (0.4-0.8)
```

**Output:**
- `visitor_report.json`
- `annotated_results/` folder

**Time:** Fast (~5-10 images/second)

---

### 5. `unique_visitor_counter.py` - Advanced Detection

**Purpose:** Better person detection using YOLOv3 + Re-ID.

**When to use:**
- ✅ Need better detection accuracy
- ✅ Complex scenes with occlusions
- ✅ People partially visible
- ✅ Willing to download large models

**What it does:**
1. Downloads YOLOv3 model (~250MB, first time only)
2. Detects people more accurately
3. Extracts features
4. Clusters to identify unique visitors

**Advantages:**
- ✅ Better detection than HOG
- ✅ Handles partial occlusions
- ✅ More accurate bounding boxes

**Disadvantages:**
- ❌ Large download (250MB)
- ❌ Slower than lightweight version
- ❌ Requires more memory

**Usage:**
```bash
python unique_visitor_counter.py
```

**First run:** Downloads models automatically

**Output:**
- `visitor_report.json`
- `results/` folder with annotated images

**Time:** ~5-8 images/second (slower than lightweight)

---

## 📊 Script Comparison

| Script | Purpose | Dataset Size | Memory | Speed | Accuracy |
|--------|---------|--------------|--------|-------|----------|
| **diagnostic_reid.py** | Find threshold | Any | Low | Fast | N/A |
| **memory_efficient_reid.py** ⭐ | Main Re-ID | 1K-100K+ | Low | Medium | High |
| **reliability_checker.py** | Validate | Any | Low | Fast | N/A |
| lightweight_visitor_counter.py | Simple detection | <1K | Low | Fast | Medium |
| unique_visitor_counter.py | Advanced detection | Any | Medium | Slow | High |

---

## 🎯 Complete Workflow

### Full Pipeline (Event → Analysis)

#### Stage 1: During Event

```bash
# Set up camera(s) at venue entrance/key areas
python camera_capture.py

# Let it run throughout the event
# Press Q when event ends to save data
```

**Output:** `visitor_database/` with thousands of images

---

#### Stage 2: Post-Event Analysis

```bash
# 1. Find optimal threshold
python diagnostic_reid.py

# 2. Edit memory_efficient_reid.py with recommended threshold
# 3. Run main system
python memory_efficient_reid.py

# 4. Validate results
python reliability_checker.py

# 5. If needed, adjust threshold and repeat step 3-4
```

---

### Alternative: Analysis Only (If You Already Have Images)

```bash
# Skip camera_capture.py
# Start directly with analysis
python diagnostic_reid.py
python memory_efficient_reid.py
python reliability_checker.py
```

---

## 📊 Script Comparison

| Script | Purpose | Dataset Size | Memory | Speed | Accuracy |
|--------|---------|--------------|--------|-------|----------|
| **camera_capture.py** 📹 | Capture images | N/A | Medium | Real-time | N/A |
| **diagnostic_reid.py** | Find threshold | Any | Low | Fast | N/A |
| **memory_efficient_reid.py** ⭐ | Main Re-ID | 1K-100K+ | Low | Medium | High |
| **reliability_checker.py** | Validate | Any | Low | Fast | N/A |
| lightweight_visitor_counter.py | Simple detection | <1K | Low | Fast | Medium |
| unique_visitor_counter.py | Advanced detection | Any | Medium | Slow | High |

---

## 🎯 Recommended Workflow

### For Large Datasets (1000+ images) - Like Your Project

```bash
# 1. Find optimal threshold
python diagnostic_reid.py

# 2. Edit memory_efficient_reid.py with recommended threshold
# 3. Run main system
python memory_efficient_reid.py

# 4. Validate results
python reliability_checker.py

# 5. If needed, adjust threshold and repeat step 3-4
```

### For Small Datasets (<1000 images)

```bash
# Quick approach
python lightweight_visitor_counter.py

# Or for better accuracy
python unique_visitor_counter.py
```

### For Testing/Prototyping

```bash
# Use lightweight version on subset
python lightweight_visitor_counter.py
```

---

## 🔧 Troubleshooting

### Problem: Memory Error (Out of RAM)

**Solution:** Use `memory_efficient_reid.py` (not the other scripts)

### Problem: Too Many/Few Unique People

**Solution:** 
1. Run `diagnostic_reid.py` again
2. Try recommended threshold
3. Use `reliability_checker.py` to validate

### Problem: Slow Processing

**Solutions:**
- Use `lightweight_visitor_counter.py` instead
- Process subset first to test threshold
- Reduce `SAMPLE_SIZE` in diagnostic script

### Problem: Different People Grouped Together

**Solution:** Lower threshold (be more strict)
```python
SIMILARITY_THRESHOLD = 0.10  # Instead of 0.15
```

### Problem: Same Person Split into Multiple Clusters

**Solution:** Raise threshold (be more lenient)
```python
SIMILARITY_THRESHOLD = 0.20  # Instead of 0.15
```

---

## 📝 Configuration Summary

### For 32K Images (Your Project)
```python
# diagnostic_reid.py
IMAGE_FOLDER = "path/to/images"
SAMPLE_SIZE = 500

# memory_efficient_reid.py  
IMAGE_FOLDER = "path/to/images"
SIMILARITY_THRESHOLD = 0.1354  # Your optimal value

# reliability_checker.py
# (reads from memory_efficient_report.json automatically)
```

### General Guidelines

| Dataset Size | Threshold Range |
|--------------|----------------|
| <100 images | 0.4 - 0.6 |
| 100-1K images | 0.25 - 0.4 |
| 1K-10K images | 0.15 - 0.25 |
| 10K+ images | 0.08 - 0.18 |

**Your project:** 32,330 images → 0.1354 ✅

---

## 🎓 For Your Teacher

**Main system used:** `memory_efficient_reid.py`

**Why?**
- Handles 32,330 images efficiently
- Memory-safe on standard laptops
- Validated with `reliability_checker.py`
- Threshold optimized with `diagnostic_reid.py`

**Other scripts:** Alternatives for different use cases or smaller datasets
