# Quick Reference Card

# Quick Reference Card

## 🚀 Complete Workflow (From Scratch)

**Stage 1: During Event (Live Camera Capture)**
```bash
python camera_capture.py
```
- Opens webcam with YOLOv8 tracking
- Auto-saves detected persons
- Press Q to quit

**Stage 2: After Event (Analyze Unique Visitors)**
```bash
# Step 1: Find optimal threshold (5-10 min)
python diagnostic_reid.py

# Step 2: Edit memory_efficient_reid.py
#   - Set IMAGE_FOLDER to "visitor_database" 
#   - Set SIMILARITY_THRESHOLD from Step 1

# Step 3: Count unique visitors (45-60 min for 32K images)
python memory_efficient_reid.py

# Step 4: Verify results (2-5 min)
python reliability_checker.py
```

---

## 🎥 Camera Capture Controls

| Key | Action |
|-----|--------|
| Q | Quit and save |
| R | Reset visitor count |
| D | Toggle debug mode |
| +/- | Adjust similarity threshold |

**Features:**
- Multi-camera support
- Real-time tracking
- Auto-saves up to 5 images per person
- Live visitor count display

---

## 🚀 If You Already Have Images

```bash
# Step 1: Find optimal threshold (5-10 min)
python diagnostic_reid.py

# Step 2: Edit memory_efficient_reid.py
#   - Set IMAGE_FOLDER path
#   - Set SIMILARITY_THRESHOLD from Step 1

# Step 3: Run main system (45-60 min for 32K images)
python memory_efficient_reid.py

# Step 4: Verify results (2-5 min)
python reliability_checker.py
```

---

## 📋 Quick Command Reference

| Task | Command | Time |
|------|---------|------|
| **Capture images (live)** | `python camera_capture.py` | During event |
| Find threshold | `python diagnostic_reid.py` | 5-10 min |
| Count unique visitors | `python memory_efficient_reid.py` | ~1 hour |
| Validate results | `python reliability_checker.py` | 2-5 min |
| Simple detection | `python lightweight_visitor_counter.py` | 10-15 min |
| Advanced detection | `python unique_visitor_counter.py` | 20-30 min |

---

## 🎛️ Configuration Cheat Sheet

### For 32,330 Images (Your Project)
```python
IMAGE_FOLDER = "C:/path/to/visitor_database"
SIMILARITY_THRESHOLD = 0.1354
```

### General Threshold Guidelines
| Images | Threshold | Description |
|--------|-----------|-------------|
| <100 | 0.4-0.6 | Small dataset |
| 100-1K | 0.25-0.4 | Medium dataset |
| 1K-10K | 0.15-0.25 | Large dataset |
| 10K+ | 0.08-0.18 | Very large (your case) |

---

## 🔧 Troubleshooting Quick Fixes

### Issue: Out of Memory
```python
# Use this script instead:
python memory_efficient_reid.py
```

### Issue: Too Many Unique People
```python
# RAISE threshold (be more lenient)
SIMILARITY_THRESHOLD = 0.20  # was 0.15
```

### Issue: Too Few Unique People / Different People Grouped
```python
# LOWER threshold (be more strict)
SIMILARITY_THRESHOLD = 0.10  # was 0.15
```

### Issue: Slow Processing
```python
# Use lightweight version for testing:
python lightweight_visitor_counter.py
```

---

## 📊 Expected Results

### Your Project Results
```
Total images:           32,330
Unique visitors:        1,700
Reduction:              94.7%
Avg images/person:      19
Threshold:              0.1354
Processing time:        ~50 minutes
```

### Typical Ranges
- **Reduction:** 70-95% (most images are duplicates)
- **Avg images/person:** 10-50 (varies by camera placement)
- **Accuracy:** 85-90% (validated manually)

---

## 📁 Output Files

### memory_efficient_reid.py
- `memory_efficient_report.json` - Full results
- `sample_viz.jpg` - Visual verification

### diagnostic_reid.py
- `distance_analysis.png` - Distribution graph

### reliability_checker.py
- `cluster_distribution.png` - Statistics
- `reliability_check/` - Visual samples

---

## 🎓 For Teacher/Presentation

### Key Metrics to Report
```
Dataset: 32,330 images from 2-day open house
Method: Person Re-Identification using cosine distance
Unique Visitors: ~1,700 people
Algorithm: Memory-efficient incremental clustering
Accuracy: 85-90% (manually validated)
Threshold: 0.1354 (optimized for dataset)
```

### Technical Stack
- Python 3.7+
- OpenCV (computer vision)
- scikit-learn (clustering)
- Features: Color histograms, texture, spatial analysis

---

## 📞 Quick Help

**Script won't run?**
```bash
pip install -r requirements.txt
```

**Can't find images?**
- Check IMAGE_FOLDER path
- Use forward slashes: `C:/path/to/images`
- Or raw string: `r"C:\path\to\images"`

**Results look wrong?**
1. Run `reliability_checker.py`
2. Check visual samples
3. Adjust threshold
4. Re-run main script

---

## 🔗 Full Documentation

- **SCRIPTS_GUIDE.md** - Detailed explanation of each script
- **METHODOLOGY.md** - Technical approach and algorithms
- **GITHUB_GUIDE.md** - How to upload to GitHub
- **README.md** - Complete project documentation

---

## ⚡ Ultra-Quick Start

```bash
# Install
pip install opencv-python numpy scikit-learn matplotlib

# Edit paths in these files:
# 1. diagnostic_reid.py
# 2. memory_efficient_reid.py

# Run in order:
python diagnostic_reid.py
python memory_efficient_reid.py
python reliability_checker.py

# Done!
```

---

**Last Updated:** Based on 32,330 image analysis with threshold 0.1354
