"""
DIAGNOSTIC Re-ID Script
Analyzes your dataset to find the optimal threshold
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

class DiagnosticReID:
    def __init__(self):
        self.embeddings = []
        self.image_names = []
        
    def extract_features(self, image_path):
        """
        Fast feature extraction optimized for large datasets
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize
        image = cv2.resize(image, (64, 128))
        
        features_list = []
        
        # 1. Color histograms (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Split into 3 regions
        h = image.shape[0]
        regions = [
            hsv[:h//3, :],      # Upper
            hsv[h//3:2*h//3, :], # Middle
            hsv[2*h//3:, :]      # Lower
        ]
        
        for region in regions:
            hist_h = cv2.calcHist([region], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([region], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([region], [2], None, [32], [0, 256])
            
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            features_list.extend([hist_h, hist_s, hist_v])
        
        # 2. Texture (simplified LBP)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple texture descriptor
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        texture_hist = np.histogram(gradient, bins=20)[0]
        texture_hist = texture_hist / (np.sum(texture_hist) + 1e-6)
        features_list.append(texture_hist)
        
        # 3. Color moments
        for channel in cv2.split(image):
            mean = np.mean(channel) / 255.0
            std = np.std(channel) / 255.0
            features_list.append(np.array([mean, std]))
        
        # Combine
        features = np.concatenate(features_list)
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def process_images(self, image_folder, max_images=None, sample_rate=1):
        """
        Process images with optional sampling for speed
        
        Args:
            max_images: Max number to process (None = all)
            sample_rate: Process every Nth image (1 = all, 10 = every 10th)
        """
        image_folder = Path(image_folder)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(image_folder.glob(ext)))
        
        image_files = sorted(image_files)
        
        # Sample if needed
        if sample_rate > 1:
            image_files = image_files[::sample_rate]
            print(f"Sampling: Processing every {sample_rate}th image")
        
        if max_images:
            image_files = image_files[:max_images]
            print(f"Limiting to first {max_images} images")
        
        total = len(image_files)
        print(f"\n{'='*60}")
        print(f"Processing {total} images...")
        print(f"{'='*60}\n")
        
        processed = 0
        failed = 0
        
        for idx, img_file in enumerate(image_files, 1):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            
            features = self.extract_features(img_file)
            
            if features is not None:
                self.embeddings.append(features)
                self.image_names.append(img_file.name)
                processed += 1
            else:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Processed: {processed}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")
        
        return processed
    
    def analyze_distance_distribution(self, sample_size=3000):
        """
        Analyze the distribution of distances between images
        This helps find the right threshold!
        """
        if len(self.embeddings) == 0:
            print("No embeddings to analyze!")
            return
        
        print("\n" + "="*60)
        print("ANALYZING DISTANCE DISTRIBUTION")
        print("="*60)
        
        embeddings_array = np.array(self.embeddings)
        
        # Sample for speed if dataset is large
        if len(embeddings_array) > sample_size:
            indices = np.random.choice(len(embeddings_array), sample_size, replace=False)
            sample_embeddings = embeddings_array[indices]
            print(f"Using sample of {sample_size} images for analysis")
        else:
            sample_embeddings = embeddings_array
        
        # Calculate pairwise distances
        print("Calculating pairwise distances...")
        distances = cosine_distances(sample_embeddings)
        
        # Get upper triangle (avoid duplicates and self-distances)
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        distance_values = distances[upper_tri_indices]
        
        # Statistics
        print(f"\nDistance Statistics:")
        print(f"  Min distance:     {np.min(distance_values):.4f}")
        print(f"  Max distance:     {np.max(distance_values):.4f}")
        print(f"  Mean distance:    {np.mean(distance_values):.4f}")
        print(f"  Median distance:  {np.median(distance_values):.4f}")
        print(f"  Std deviation:    {np.std(distance_values):.4f}")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(distance_values, p)
            print(f"  {p}th percentile: {val:.4f}")
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(distance_values, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Distance')
        plt.ylabel('Frequency')
        plt.title('Distance Distribution')
        plt.axvline(np.mean(distance_values), color='r', linestyle='--', label=f'Mean: {np.mean(distance_values):.3f}')
        plt.axvline(np.median(distance_values), color='g', linestyle='--', label=f'Median: {np.median(distance_values):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(distance_values, bins=50, cumulative=True, density=True, 
                edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Distance')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.grid(True, alpha=0.3)
        
        # Mark suggested thresholds
        for percentile, color in [(10, 'blue'), (25, 'green'), (50, 'orange')]:
            val = np.percentile(distance_values, percentile)
            plt.axvline(val, color=color, linestyle='--', 
                       label=f'{percentile}th: {val:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('distance_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved distance distribution plot: distance_analysis.png")
        
        # Suggest thresholds
        print(f"\n{'='*60}")
        print("SUGGESTED THRESHOLDS:")
        print(f"{'='*60}")
        
        percentile_10 = np.percentile(distance_values, 10)
        percentile_25 = np.percentile(distance_values, 25)
        percentile_50 = np.percentile(distance_values, 50)
        
        print(f"VERY STRICT (10th percentile):  {percentile_10:.4f}")
        print(f"  → Use if you have MANY duplicates of same person")
        print(f"  → Groups only very similar images")
        
        print(f"\nSTRICT (25th percentile):       {percentile_25:.4f}")
        print(f"  → Recommended starting point")
        print(f"  → Good balance for most cases")
        
        print(f"\nMODERATE (50th percentile):     {percentile_50:.4f}")
        print(f"  → Use if people look very different")
        print(f"  → More lenient grouping")
        
        print(f"{'='*60}\n")
        
        return {
            'min': np.min(distance_values),
            'max': np.max(distance_values),
            'mean': np.mean(distance_values),
            'median': np.median(distance_values),
            'p10': percentile_10,
            'p25': percentile_25,
            'p50': percentile_50
        }
    
    def test_threshold(self, threshold):
        """
        Test a specific threshold and show results
        """
        if len(self.embeddings) == 0:
            return None
        
        embeddings_array = np.array(self.embeddings)
        distance_matrix = cosine_distances(embeddings_array)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        unique_people = len(set(labels))
        
        # Get cluster sizes
        clusters = defaultdict(int)
        for label in labels:
            clusters[label] += 1
        
        cluster_sizes = list(clusters.values())
        
        print(f"\nThreshold {threshold:.4f}:")
        print(f"  Unique people: {unique_people}")
        print(f"  Avg images per person: {np.mean(cluster_sizes):.1f}")
        print(f"  Largest cluster: {max(cluster_sizes)} images")
        print(f"  Smallest cluster: {min(cluster_sizes)} images")
        
        return unique_people
    
    def run_diagnostic(self, image_folder, sample_images=3000):
        """
        Full diagnostic pipeline
        """
        # Process images
        processed = self.process_images(image_folder, max_images=sample_images)
        
        if processed == 0:
            print("No images processed!")
            return
        
        # Analyze distances
        stats = self.analyze_distance_distribution()
        
        # Test suggested thresholds
        print("\n" + "="*60)
        print("TESTING SUGGESTED THRESHOLDS:")
        print("="*60)
        
        thresholds_to_test = [
            stats['p10'],
            stats['p25'],
            stats['p50']
        ]
        
        for threshold in thresholds_to_test:
            self.test_threshold(threshold)
        
        print("\n" + "="*60)
        print("DIAGNOSTIC COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Look at 'distance_analysis.png' to see the distribution")
        print("2. Choose a threshold based on results above")
        print("3. Use that threshold in the main Re-ID script")
        print("="*60)


def main():
    """
    Run diagnostics
    """
    print("="*60)
    print("RE-ID DIAGNOSTIC TOOL")
    print("Find the optimal threshold for your dataset")
    print("="*60)
    
    # ===== CONFIGURATION =====
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive/Desktop/human_tracking_test/Full Visitor Database over Open House (D1D2)"
    
    # For 4000+ images, we'll sample for speed
    SAMPLE_SIZE = 3000  # Analyze 3000 random images to find optimal threshold
    # =========================
    
    if not Path(IMAGE_FOLDER).exists():
        print(f"ERROR: Folder not found: {IMAGE_FOLDER}")
        return
    
    diagnostic = DiagnosticReID()
    diagnostic.run_diagnostic(IMAGE_FOLDER, sample_images=SAMPLE_SIZE)


if __name__ == "__main__":
    main()
