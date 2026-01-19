# Unique Visitor Counter using Person Re-Identification

A computer vision project to count unique visitors from camera footage by detecting and re-identifying people across multiple images using deep feature extraction and clustering algorithms.

## 📋 Project Overview

This system processes camera images from an event venue to accurately count unique visitors by:
1. Extracting distinctive features from each person detection
2. Comparing features using cosine distance metrics
3. Clustering similar features to identify unique individuals
4. Generating comprehensive reports and visualizations

### Use Case
Developed for analyzing visitor traffic at a secondary school open house event with:
- Day 1: School bus arrivals
- Day 2: Walk-in visitors
- Multiple camera captures of same individuals

## 🎯 Results

- **Total Images Processed:** 32,330
- **Unique Visitors Identified:** ~1,700
- **Duplicate Removal Rate:** 94.7%
- **Average Appearances per Visitor:** ~19 images
- **Optimal Threshold:** 0.1354

## 🚀 Quick Start

### Full Pipeline (From Scratch)

#### Stage 1: Capture Images (During Event)

```bash
# Install dependencies
pip install -r requirements.txt

# Run live camera capture with tracking
python camera_capture.py
```

**What it does:**
- Opens webcam(s) with YOLOv8 person detection
- Tracks people across frames
- Automatically saves images to `visitor_database/`
- Shows live count and preview

**Controls:**
- Q: Quit and save
- R: Reset count
- D: Toggle debug mode
- +/-: Adjust similarity threshold

**Output:** `visitor_database/` folder with captured images

---

#### Stage 2: Analyze Unique Visitors (After Event)

```bash
# Find optimal threshold
python diagnostic_reid.py

# Count unique visitors
python memory_efficient_reid.py

# Verify results
python reliability_checker.py
```

### Quick Analysis (If You Already Have Images)

```bash
# Clone the repository
git clone https://github.com/yourusername/visitor-reid-counter.git
cd visitor-reid-counter

# Install dependencies
pip install -r requirements.txt

# Configure your image folder in memory_efficient_reid.py
# Then run
python memory_efficient_reid.py
```

### Verify Results

```bash
python reliability_checker.py
```

## 📖 Complete Workflow

### First Time Setup

#### Step 1: Find Your Optimal Threshold

```bash
python diagnostic_reid.py
```

**What it does:**
- Samples 500 images from your dataset
- Calculates pairwise distances
- Shows distribution graph
- Recommends threshold values

**Output:**
```
SUGGESTED THRESHOLDS:
VERY STRICT (10th percentile):  0.18
STRICT (25th percentile):       0.22  ← Start here
MODERATE (50th percentile):     0.35
```

#### Step 2: Configure and Run Main System

Edit `memory_efficient_reid.py`:
```python
IMAGE_FOLDER = "path/to/your/images"
SIMILARITY_THRESHOLD = 0.22  # Use recommended value
```

Run:
```bash
python memory_efficient_reid.py
```

**Processing time:** ~45-60 minutes for 32K images

**Generates:**
- `memory_efficient_report.json` - Full results
- `sample_viz.jpg` - Sample verification images

#### Step 3: Validate Results

```bash
python reliability_checker.py
```

**What it checks:**
- Cluster size distribution
- Suspicious over/under-grouping
- Creates visual samples for manual verification

**If results look wrong:**
- Adjust threshold in `memory_efficient_reid.py`
- Re-run the main system
- Validate again

### Alternative Scripts

#### 📊 For Simple People Counting (Detection Only)

**Use:** `lightweight_visitor_counter.py`

```bash
python lightweight_visitor_counter.py
```

- No large downloads needed
- Uses OpenCV HOG detector
- Good for: Quick counting with duplicates
- **Note:** Counts detections, not unique people

#### 🎯 For Better Detection Accuracy

**Use:** `unique_visitor_counter.py`

```bash
python unique_visitor_counter.py
```

- Uses YOLOv3 (downloads ~250MB first time)
- Better person detection
- Good for: Complex scenes with occlusions
- Then use Re-ID scripts to count unique visitors

### Verify Results

## 🔧 Technical Approach

### Feature Extraction
- Multi-scale HSV and LAB color histograms
- Texture analysis using gradient-based features
- Regional analysis (upper/middle/lower body segments)
- L2 normalized feature vectors

### Clustering
- Memory-efficient incremental clustering
- Cosine distance similarity metric
- Threshold-based grouping (optimal: 0.1354)

### Key Features
- ✅ **Memory-efficient**: Handles 30K+ images on standard laptops
- ✅ **Scalable**: Incremental processing without full distance matrix
- ✅ **Robust**: Works with motion blur and partial occlusions
- ✅ **Validated**: Built-in reliability checking tools

## 📁 Project Structure

```
visitor-reid-counter/
├── Part 1: Data Collection
│   └── camera_capture.py              # 📹 Live camera capture with tracking (Run FIRST)
│
├── Part 2: Unique Visitor Counting (Core Scripts)
│   ├── diagnostic_reid.py             # Find optimal threshold for your data
│   ├── memory_efficient_reid.py       # ⭐ Main Re-ID system for large datasets
│   └── reliability_checker.py         # Validate and verify results
│
├── Alternative Re-ID Scripts
│   ├── lightweight_visitor_counter.py # Simple version (no large downloads)
│   ├── unique_visitor_counter.py      # YOLOv3-based detection + Re-ID
│   ├── advanced_visitor_counter.py    # YOLOv8 + ResNet50 (best accuracy)
│   ├── production_reid.py             # Alternative production script
│   ├── reid_only_counter.py           # For pre-cropped person images
│   └── improved_reid_counter.py       # Enhanced features version
│
├── Documentation
│   ├── README.md                      # This file - project overview
│   ├── SCRIPTS_GUIDE.md               # Detailed guide for each script
│   ├── QUICK_REFERENCE.md             # Quick cheat sheet
│   ├── METHODOLOGY.md                 # Technical explanation
│   ├── GITHUB_GUIDE.md                # How to upload to GitHub
│   └── LICENSE                        # MIT License
│
└── Configuration
    ├── requirements.txt               # Python dependencies
    └── .gitignore                    # Git ignore rules
```

### Complete Workflow

**Stage 1: Data Collection** (During event)
- `camera_capture.py` → Captures images from webcam(s) with YOLO tracking

**Stage 2: Unique Visitor Analysis** (After event)
- `diagnostic_reid.py` → Find optimal threshold
- `memory_efficient_reid.py` → Count unique visitors
- `reliability_checker.py` → Validate results

## 🎛️ Configuration

Edit `memory_efficient_reid.py`:

```python
IMAGE_FOLDER = "path/to/your/images"
SIMILARITY_THRESHOLD = 0.1354  # Adjust based on your dataset
```

### Threshold Selection Guide

| Dataset Size | Recommended Range | Description |
|--------------|------------------|-------------|
| < 100 images | 0.4 - 0.6 | Small datasets need higher thresholds |
| 100-1000 images | 0.25 - 0.4 | Medium datasets |
| 1000-10000 images | 0.15 - 0.25 | Large datasets |
| 10000+ images | 0.08 - 0.18 | Very large datasets (like this project) |

## 📊 Performance

- **Processing Speed**: ~10-15 images/second
- **Memory Usage**: ~500 MB peak
- **Total Time (32K images)**: 45-60 minutes
- **Accuracy**: 85-90% (validated)

## 🐛 Known Limitations

1. Similar clothing may cause grouping errors
2. Heavy motion blur reduces accuracy
3. Extreme partial views are challenging
4. Large pose variations affect matching

## 📚 Dependencies

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

**Your Name**
- GitHub: [8051brandon]([https://github.com/8051brandon])
- Email: 8051brandon@gmail.com

## 🙏 Acknowledgments

- School administration for providing the dataset
- OpenCV and scikit-learn communities

---

**Privacy Note**: This system is designed for ethical visitor counting in public spaces with appropriate consent.
