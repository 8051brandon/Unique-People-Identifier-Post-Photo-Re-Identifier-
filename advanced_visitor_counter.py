"""
ADVANCED Unique Visitor Counter with YOLOv8 + Deep Re-ID
Uses state-of-the-art models for best accuracy:
- YOLOv8 for person detection (better than YOLOv3)
- Deep learning features for person re-identification
- More accurate matching with better features
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import torch
import torchvision.transforms as transforms
from torchvision import models

class AdvancedVisitorCounter:
    def __init__(self, similarity_threshold=0.7, use_gpu=True):
        """
        Initialize advanced visitor counter with deep learning models
        
        Args:
            similarity_threshold: Distance threshold (0.5-0.85 recommended for deep features)
            use_gpu: Use GPU if available (much faster)
        """
        self.similarity_threshold = similarity_threshold
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.detector = None
        self.feature_extractor = None
        
        self.embeddings = []
        self.image_names = []
        self.bbox_data = []
        self.confidence_scores = []
    
    def setup_models(self):
        """
        Setup YOLOv8 for detection and ResNet for feature extraction
        """
        print("\n" + "="*60)
        print("Setting up models (this may take a few minutes first time)...")
        print("="*60)
        
        # 1. YOLOv8 for person detection
        try:
            from ultralytics import YOLO
            print("Loading YOLOv8 detector...")
            self.detector = YOLO('yolov8n.pt')  # nano version (fast & accurate)
            print("✓ YOLOv8 loaded successfully")
        except ImportError:
            print("ERROR: ultralytics not installed. Installing now...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'ultralytics', '--break-system-packages'])
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.pt')
            print("✓ YOLOv8 loaded successfully")
        
        # 2. ResNet50 for feature extraction (pre-trained on ImageNet)
        print("Loading ResNet50 feature extractor...")
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer to get features
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        print("✓ ResNet50 loaded successfully")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard Re-ID size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("="*60)
        print("✓ All models ready!\n")
    
    def detect_people(self, image_path):
        """
        Detect people using YOLOv8 (much better than HOG or old YOLO)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return [], None, []
        
        # Run YOLOv8 detection
        results = self.detector(image, classes=[0], verbose=False)  # class 0 = person
        
        boxes = []
        confidences = []
        
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Convert to [x, y, w, h] format
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    
                    if conf > 0.4:  # Confidence threshold
                        boxes.append([x, y, w, h])
                        confidences.append(conf)
        
        return boxes, image, confidences
    
    def extract_deep_features(self, image, bbox):
        """
        Extract deep learning features using ResNet50
        Much better than hand-crafted features!
        """
        x, y, w, h = bbox
        
        # Ensure bbox is within image bounds
        height, width = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 10 or h <= 10:
            return None
        
        # Crop person
        person_crop = image[y:y+h, x:x+w]
        
        if person_crop.size == 0:
            return None
        
        # Convert BGR to RGB
        person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.squeeze().cpu().numpy()
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def extract_hybrid_features(self, image, bbox):
        """
        Combine deep features with color/shape features for best results
        """
        # Get deep features
        deep_features = self.extract_deep_features(image, bbox)
        if deep_features is None:
            return None
        
        # Get color features (lightweight but useful)
        x, y, w, h = bbox
        height, width = image.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        
        if w <= 10 or h <= 10:
            return deep_features
        
        person_crop = image[y:y+h, x:x+w]
        person_crop = cv2.resize(person_crop, (64, 128))
        
        # Color histogram in HSV
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [30], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        
        color_features = np.concatenate([hist_h, hist_s])
        color_features = color_features / (np.linalg.norm(color_features) + 1e-6)
        
        # Combine: 90% deep features, 10% color features
        combined = np.concatenate([
            deep_features * 0.9,
            color_features * 0.1
        ])
        
        # Normalize again
        combined = combined / (np.linalg.norm(combined) + 1e-6)
        
        return combined
    
    def process_images(self, image_folder, verbose=True):
        """
        Process all images with advanced detection and feature extraction
        """
        image_folder = Path(image_folder)
        
        # Support multiple formats
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(image_folder.glob(ext)))
        
        if len(image_files) == 0:
            print(f"ERROR: No images found in {image_folder}")
            return 0
        
        print(f"\n{'='*60}")
        print(f"Found {len(image_files)} images to process")
        print(f"{'='*60}\n")
        
        total_detections = 0
        
        for img_idx, img_file in enumerate(image_files, 1):
            if verbose:
                print(f"[{img_idx}/{len(image_files)}] {img_file.name}...", end=' ')
            
            boxes, image, confidences = self.detect_people(img_file)
            
            if image is None:
                print("FAILED")
                continue
            
            detected_in_image = 0
            for bbox, conf in zip(boxes, confidences):
                # Use hybrid features for best accuracy
                features = self.extract_hybrid_features(image, bbox)
                
                if features is not None:
                    self.embeddings.append(features)
                    self.image_names.append(img_file.name)
                    self.bbox_data.append(bbox)
                    self.confidence_scores.append(conf)
                    total_detections += 1
                    detected_in_image += 1
            
            if verbose:
                print(f"✓ Found {detected_in_image} person(s)")
        
        print(f"\n{'='*60}")
        print(f"Total detections: {total_detections}")
        print(f"{'='*60}\n")
        
        return total_detections
    
    def cluster_embeddings(self):
        """
        Cluster using cosine distance (best for deep features)
        """
        if len(self.embeddings) == 0:
            return []
        
        embeddings_array = np.array(self.embeddings)
        distance_matrix = cosine_distances(embeddings_array)
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.similarity_threshold,
            min_samples=1,
            metric='precomputed'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        return labels
    
    def analyze_clusters(self, labels):
        """
        Detailed cluster analysis
        """
        if len(labels) == 0:
            return {}, {}
        
        unique_labels = set(labels)
        clusters = defaultdict(list)
        
        for idx, label in enumerate(labels):
            clusters[label].append({
                'image': self.image_names[idx],
                'bbox': self.bbox_data[idx],
                'confidence': self.confidence_scores[idx]
            })
        
        cluster_sizes = [len(clusters[label]) for label in unique_labels]
        
        stats = {
            'total_detections': len(labels),
            'unique_visitors': len(unique_labels),
            'duplicates_removed': len(labels) - len(unique_labels),
            'reduction_percentage': (len(labels) - len(unique_labels)) / len(labels) * 100 if len(labels) > 0 else 0,
            'avg_appearances_per_visitor': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_appearances': max(cluster_sizes) if cluster_sizes else 0,
            'min_appearances': min(cluster_sizes) if cluster_sizes else 0,
        }
        
        return stats, clusters
    
    def print_results(self, stats, clusters):
        """
        Display formatted results
        """
        print(f"\n{'='*60}")
        print(f"RESULTS - ADVANCED VISITOR COUNTER")
        print(f"{'='*60}")
        print(f"Total detections:              {stats['total_detections']}")
        print(f"Unique visitors identified:    {stats['unique_visitors']}")
        print(f"Duplicates removed:            {stats['duplicates_removed']}")
        print(f"Reduction:                     {stats['reduction_percentage']:.1f}%")
        print(f"{'='*60}")
        print(f"Average appearances/visitor:   {stats['avg_appearances_per_visitor']:.1f}")
        print(f"Maximum appearances:           {stats['max_appearances']}")
        print(f"{'='*60}\n")
        
        # Visitor breakdown
        print("Visitor breakdown:")
        for visitor_id in sorted(clusters.keys()):
            appearances = clusters[visitor_id]
            images = list(set([a['image'] for a in appearances]))
            avg_conf = np.mean([a['confidence'] for a in appearances])
            
            print(f"  Visitor #{visitor_id}: {len(appearances)} appearance(s), "
                  f"avg conf: {avg_conf:.2f}")
            print(f"    Images: {', '.join(images[:5])}"
                  f"{'...' if len(images) > 5 else ''}")
    
    def generate_report(self, output_file="advanced_visitor_report.json"):
        """
        Generate comprehensive JSON report
        """
        labels = self.cluster_embeddings()
        stats, clusters = self.analyze_clusters(labels)
        
        serializable_clusters = {}
        for visitor_id, detections in clusters.items():
            serializable_clusters[f'visitor_{visitor_id}'] = {
                'total_appearances': len(detections),
                'avg_confidence': float(np.mean([d['confidence'] for d in detections])),
                'unique_images': list(set([d['image'] for d in detections])),
                'all_appearances': detections
            }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': 'YOLOv8 + ResNet50',
            'statistics': stats,
            'similarity_threshold': self.similarity_threshold,
            'device': self.device,
            'visitors': serializable_clusters
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved: {output_file}")
        return report, stats, clusters
    
    def visualize_results(self, image_folder, output_folder="advanced_results", max_images=15):
        """
        Create annotated visualizations
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        labels = self.cluster_embeddings()
        unique_labels = set(labels)
        
        # Generate distinct colors
        np.random.seed(42)
        colors = {}
        for label in unique_labels:
            colors[label] = tuple(int(c) for c in np.random.randint(50, 255, 3).tolist())
        
        # Group by image
        image_detections = defaultdict(list)
        for idx, img_name in enumerate(self.image_names):
            image_detections[img_name].append({
                'bbox': self.bbox_data[idx],
                'label': labels[idx],
                'confidence': self.confidence_scores[idx]
            })
        
        print(f"\nGenerating visualizations...")
        processed = 0
        
        for img_name, detections in list(image_detections.items())[:max_images]:
            img_path = Path(image_folder) / img_name
            image = cv2.imread(str(img_path))
            
            if image is None:
                continue
            
            for det in detections:
                x, y, w, h = det['bbox']
                color = colors[det['label']]
                
                # Draw thick border
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
                
                # Draw label with background
                label_text = f"V{det['label']} ({det['confidence']:.2f})"
                (text_w, text_h), _ = cv2.getTextSize(label_text, 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x, y-text_h-10), (x+text_w+5, y), color, -1)
                cv2.putText(image, label_text, (x+3, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            output_path = output_folder / f"advanced_{img_name}"
            cv2.imwrite(str(output_path), image)
            processed += 1
        
        print(f"✓ {processed} visualizations saved to: {output_folder}/")
    
    def run_complete_analysis(self, image_folder, threshold=None):
        """
        Run full pipeline
        """
        if threshold is not None:
            self.similarity_threshold = threshold
        
        # Process
        total_det = self.process_images(image_folder)
        
        if total_det == 0:
            print("No people detected!")
            return None
        
        # Generate report
        report, stats, clusters = self.generate_report()
        self.print_results(stats, clusters)
        
        # Visualize
        self.visualize_results(image_folder)
        
        return stats


def main():
    """
    Main execution
    """
    print("="*60)
    print("ADVANCED VISITOR COUNTER")
    print("YOLOv8 + ResNet50 Deep Learning")
    print("="*60)
    
    # ===== CONFIGURATION =====
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive/Desktop/human_tracking_test/visitor_database 2nd session/visitor_database"
    SIMILARITY_THRESHOLD = 0.7  # Higher threshold for deep features (0.6-0.85)
    USE_GPU = True  # Set False if you don't have NVIDIA GPU
    # =========================
    
    if not Path(IMAGE_FOLDER).exists():
        print(f"ERROR: Folder not found: {IMAGE_FOLDER}")
        return
    
    # Initialize
    counter = AdvancedVisitorCounter(
        similarity_threshold=SIMILARITY_THRESHOLD,
        use_gpu=USE_GPU
    )
    
    try:
        # Setup models
        counter.setup_models()
        
        # Run analysis
        stats = counter.run_complete_analysis(IMAGE_FOLDER)
        
        if stats:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE!")
            print("="*60)
            print("Generated files:")
            print("  • advanced_visitor_report.json")
            print("  • advanced_results/ (visualizations)")
            print("\nThreshold adjustment guide:")
            print("  • 0.6-0.7  = Strict (fewer false matches)")
            print("  • 0.7-0.75 = Balanced (RECOMMENDED)")
            print("  • 0.75-0.85 = Loose (may group similar people)")
            print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
