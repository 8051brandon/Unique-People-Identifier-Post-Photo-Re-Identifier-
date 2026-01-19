"""
Unique Visitor Counter using Person Re-Identification
This script processes multiple images to count unique visitors by:
1. Detecting people in images
2. Extracting feature embeddings
3. Clustering similar embeddings (same person)
4. Counting unique visitors
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

class UniqueVisitorCounter:
    def __init__(self, similarity_threshold=0.5):
        """
        Initialize the visitor counter
        
        Args:
            similarity_threshold: Lower = more strict matching (0.3-0.6 recommended)
        """
        self.similarity_threshold = similarity_threshold
        self.detector = None
        self.feature_extractor = None
        self.embeddings = []
        self.image_names = []
        self.bbox_data = []
        
    def setup_models(self):
        """Setup YOLO for detection and feature extraction"""
        print("Setting up detection model...")
        
        # Using YOLOv3 for person detection
        self.detector = cv2.dnn.readNet(
            "yolov3.weights",  # You'll need to download these
            "yolov3.cfg"
        )
        
        # Load COCO class names
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print("Models ready!")
    
    def download_models(self):
        """Helper to download required model files"""
        import urllib.request
        
        files = {
            "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
            "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        }
        
        print("Downloading model files (this may take a few minutes)...")
        for filename, url in files.items():
            if not Path(filename).exists():
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
        print("Download complete!")
    
    def detect_people(self, image_path):
        """
        Detect people in an image
        
        Returns:
            List of bounding boxes [x, y, w, h] for detected people
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read {image_path}")
            return []
        
        height, width = image.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.detector.setInput(blob)
        
        # Get output layer names
        layer_names = self.detector.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.detector.getUnconnectedOutLayers()]
        
        # Run detection
        outputs = self.detector.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only keep 'person' class (class_id = 0 in COCO)
                if class_id == 0 and confidence > 0.5:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
        
        return final_boxes, image
    
    def extract_features(self, image, bbox):
        """
        Extract feature vector from a person crop
        Uses color histogram and HOG features as a simple Re-ID approach
        """
        x, y, w, h = bbox
        
        # Ensure bbox is within image bounds
        height, width = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 0 or h <= 0:
            return None
        
        person_crop = image[y:y+h, x:x+w]
        
        if person_crop.size == 0:
            return None
        
        # Resize to standard size
        person_crop = cv2.resize(person_crop, (64, 128))
        
        # Extract color histogram features (in HSV space)
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Extract HOG features
        hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(person_crop).flatten()
        
        # Normalize HOG features
        hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-6)
        
        # Combine features
        features = np.concatenate([hist_h, hist_s, hist_v, hog_features])
        
        # L2 normalize the final feature vector
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def process_images(self, image_folder):
        """
        Process all images in a folder and extract person embeddings
        """
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
        
        print(f"Found {len(image_files)} images to process")
        
        total_detections = 0
        
        for img_file in image_files:
            print(f"Processing {img_file.name}...")
            
            boxes, image = self.detect_people(img_file)
            
            for bbox in boxes:
                features = self.extract_features(image, bbox)
                
                if features is not None:
                    self.embeddings.append(features)
                    self.image_names.append(img_file.name)
                    self.bbox_data.append(bbox)
                    total_detections += 1
        
        print(f"\nTotal detections across all images: {total_detections}")
        return total_detections
    
    def cluster_embeddings(self):
        """
        Cluster embeddings to identify unique individuals
        Uses DBSCAN with cosine distance
        """
        if len(self.embeddings) == 0:
            print("No embeddings to cluster!")
            return []
        
        embeddings_array = np.array(self.embeddings)
        
        # Calculate cosine distance matrix
        distance_matrix = cosine_distances(embeddings_array)
        
        # Use DBSCAN for clustering
        # eps is the maximum distance for samples to be considered in same cluster
        clustering = DBSCAN(
            eps=self.similarity_threshold,
            min_samples=1,
            metric='precomputed'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        return labels
    
    def count_unique_visitors(self):
        """
        Count unique visitors based on clustering
        """
        labels = self.cluster_embeddings()
        
        if len(labels) == 0:
            return 0, {}
        
        unique_visitors = len(set(labels))
        
        # Create a mapping of cluster to detections
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append({
                'image': self.image_names[idx],
                'bbox': self.bbox_data[idx]
            })
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"Total detections: {len(labels)}")
        print(f"Unique visitors: {unique_visitors}")
        print(f"Overcounting reduction: {len(labels) - unique_visitors} duplicates removed")
        print(f"{'='*60}\n")
        
        return unique_visitors, clusters
    
    def generate_report(self, output_file="visitor_report.json"):
        """
        Generate a detailed report
        """
        unique_count, clusters = self.count_unique_visitors()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(self.embeddings),
            'unique_visitors': unique_count,
            'similarity_threshold': self.similarity_threshold,
            'clusters': {}
        }
        
        for cluster_id, detections in clusters.items():
            report['clusters'][f'visitor_{cluster_id}'] = {
                'count': len(detections),
                'appearances': detections
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {output_file}")
        return report
    
    def visualize_results(self, image_folder, output_folder="results"):
        """
        Create visualizations showing unique visitors
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        labels = self.cluster_embeddings()
        unique_labels = set(labels)
        
        print(f"Creating visualizations for {len(unique_labels)} unique visitors...")
        
        # Create a color map for different visitors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Process each image and draw bounding boxes
        processed_images = set()
        
        for idx, (img_name, bbox, label) in enumerate(zip(self.image_names, self.bbox_data, labels)):
            if img_name in processed_images:
                continue
            
            img_path = Path(image_folder) / img_name
            image = cv2.imread(str(img_path))
            
            if image is None:
                continue
            
            # Draw all boxes for this image
            for i, (name, box, lbl) in enumerate(zip(self.image_names, self.bbox_data, labels)):
                if name == img_name:
                    x, y, w, h = box
                    color = tuple(int(c * 255) for c in color_map[lbl][:3])
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(image, f"Visitor {lbl}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save annotated image
            output_path = output_folder / f"annotated_{img_name}"
            cv2.imwrite(str(output_path), image)
            processed_images.add(img_name)
        
        print(f"Visualizations saved to {output_folder}/")


def main():
    """
    Main function to run the visitor counter
    """
    print("="*60)
    print("UNIQUE VISITOR COUNTER")
    print("="*60)
    
    # Configuration
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive/Desktop/human_tracking_test/visitor_database 2nd session/visitor_database"
    SIMILARITY_THRESHOLD = 0.5  # Lower = stricter matching (0.3-0.6 recommended)
    
    # Initialize counter
    counter = UniqueVisitorCounter(similarity_threshold=SIMILARITY_THRESHOLD)
    
    # Download models if needed (first time only)
    try:
        counter.setup_models()
    except:
        print("Model files not found. Downloading...")
        counter.download_models()
        counter.setup_models()
    
    # Process images
    counter.process_images(IMAGE_FOLDER)
    
    # Count unique visitors
    unique_count, clusters = counter.count_unique_visitors()
    
    # Generate report
    counter.generate_report("visitor_report.json")
    
    # Create visualizations
    counter.visualize_results(IMAGE_FOLDER)
    
    print("\nProcessing complete!")
    print(f"Check 'visitor_report.json' for detailed results")
    print(f"Check 'results/' folder for annotated images")


if __name__ == "__main__":
    main()
