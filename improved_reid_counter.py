"""
IMPROVED Person Re-ID with Advanced Features
Better handling of difficult cases with similar clothing
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class ImprovedReIDCounter:
    def __init__(self, similarity_threshold=0.5, use_hierarchical=True):
        """
        Improved Re-ID with better features
        
        Args:
            similarity_threshold: 0.2-0.6 recommended
            use_hierarchical: Use hierarchical clustering (better for similar people)
        """
        self.similarity_threshold = similarity_threshold
        self.use_hierarchical = use_hierarchical
        self.embeddings = []
        self.image_names = []
        self.image_shapes = []  # Store original shapes for body shape analysis
    
    def extract_advanced_features(self, image_path):
        """
        Extract comprehensive features with focus on subtle differences
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Store original shape ratios
        h, w = image.shape[:2]
        self.image_shapes.append((h, w))
        
        # Resize to standard size
        image_resized = cv2.resize(image, (64, 128))
        
        features_list = []
        
        # === 1. MULTI-SCALE COLOR FEATURES ===
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
        
        # Divide into multiple regions for finer detail
        h_regions = 4  # Divide vertically into 4 parts
        region_height = image_resized.shape[0] // h_regions
        
        for i in range(h_regions):
            y_start = i * region_height
            y_end = (i + 1) * region_height if i < h_regions - 1 else image_resized.shape[0]
            
            region_hsv = hsv[y_start:y_end, :]
            region_lab = lab[y_start:y_end, :]
            
            # HSV histograms
            hist_h = cv2.calcHist([region_hsv], [0], None, [36], [0, 180])
            hist_s = cv2.calcHist([region_hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([region_hsv], [2], None, [32], [0, 256])
            
            # LAB histograms (better for subtle color differences)
            hist_l = cv2.calcHist([region_lab], [0], None, [32], [0, 256])
            hist_a = cv2.calcHist([region_lab], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([region_lab], [2], None, [32], [0, 256])
            
            # Normalize and add
            for hist in [hist_h, hist_s, hist_v, hist_l, hist_a, hist_b]:
                hist_norm = cv2.normalize(hist, hist).flatten()
                features_list.append(hist_norm)
        
        # === 2. TEXTURE FEATURES (Very important for similar colors) ===
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern (LBP) - captures fabric texture
        lbp = self.compute_lbp(gray)
        lbp_hist = np.histogram(lbp, bins=26, range=(0, 26))[0]
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-6)
        features_list.append(lbp_hist)
        
        # Gabor filters (different orientations) - captures patterns
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_hist = np.histogram(filtered, bins=20)[0]
            gabor_hist = gabor_hist / (np.sum(gabor_hist) + 1e-6)
            features_list.append(gabor_hist)
        
        # === 3. EDGE FEATURES (Body shape and clothing boundaries) ===
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge density per region
        for i in range(h_regions):
            y_start = i * region_height
            y_end = (i + 1) * region_height if i < h_regions - 1 else edges.shape[0]
            region_edges = edges[y_start:y_end, :]
            edge_density = np.sum(region_edges > 0) / region_edges.size
            features_list.append(np.array([edge_density]))
        
        # === 4. COLOR MOMENTS (Statistical color features) ===
        for channel in cv2.split(image_resized):
            mean = np.mean(channel) / 255.0
            std = np.std(channel) / 255.0
            skewness = np.mean(((channel / 255.0 - mean) / (std + 1e-6)) ** 3)
            features_list.append(np.array([mean, std, skewness]))
        
        # === 5. GRADIENT FEATURES ===
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_hist = np.histogram(gradient_mag, bins=20)[0]
        gradient_hist = gradient_hist / (np.sum(gradient_hist) + 1e-6)
        features_list.append(gradient_hist)
        
        # === 6. SPATIAL PYRAMID (Multi-scale spatial info) ===
        # 2x2 grid of color histograms
        for i in range(2):
            for j in range(2):
                h_start = i * 64
                w_start = j * 32
                region = hsv[h_start:h_start+64, w_start:w_start+32]
                
                hist_h = cv2.calcHist([region], [0], None, [18], [0, 180])
                hist_s = cv2.calcHist([region], [1], None, [16], [0, 256])
                
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                hist_s = cv2.normalize(hist_s, hist_s).flatten()
                
                features_list.extend([hist_h, hist_s])
        
        # Combine all features
        features = np.concatenate(features_list)
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def compute_lbp(self, gray_image):
        """
        Compute Local Binary Pattern for texture analysis
        """
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                code = 0
                
                # Compare with 8 neighbors
                code |= (gray_image[i-1, j-1] > center) << 7
                code |= (gray_image[i-1, j] > center) << 6
                code |= (gray_image[i-1, j+1] > center) << 5
                code |= (gray_image[i, j+1] > center) << 4
                code |= (gray_image[i+1, j+1] > center) << 3
                code |= (gray_image[i+1, j] > center) << 2
                code |= (gray_image[i+1, j-1] > center) << 1
                code |= (gray_image[i, j-1] > center) << 0
                
                lbp[i, j] = code
        
        return lbp
    
    def process_images(self, image_folder, verbose=True):
        """
        Process all images
        """
        image_folder = Path(image_folder)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(image_folder.glob(ext)))
        
        if len(image_files) == 0:
            print(f"ERROR: No images found in {image_folder}")
            return 0
        
        image_files = sorted(image_files)
        
        print(f"\n{'='*60}")
        print(f"Processing {len(image_files)} images with ADVANCED features")
        print(f"{'='*60}\n")
        
        processed = 0
        
        for img_idx, img_file in enumerate(image_files, 1):
            if verbose:
                print(f"[{img_idx}/{len(image_files)}] {img_file.name}...", end=' ')
            
            features = self.extract_advanced_features(img_file)
            
            if features is not None:
                self.embeddings.append(features)
                self.image_names.append(img_file.name)
                processed += 1
                if verbose:
                    print("✓")
            else:
                if verbose:
                    print("FAILED")
        
        print(f"\n{'='*60}")
        print(f"Successfully processed: {processed}/{len(image_files)}")
        print(f"{'='*60}\n")
        
        return processed
    
    def cluster_embeddings(self):
        """
        Cluster using better algorithm for difficult cases
        """
        if len(self.embeddings) == 0:
            return []
        
        embeddings_array = np.array(self.embeddings)
        
        if self.use_hierarchical:
            # Hierarchical clustering - better for similar people
            distance_matrix = cosine_distances(embeddings_array)
            
            # Use average linkage (more robust than single/complete)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.similarity_threshold,
                metric='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(distance_matrix)
        else:
            # DBSCAN (original method)
            from sklearn.cluster import DBSCAN
            distance_matrix = cosine_distances(embeddings_array)
            
            clustering = DBSCAN(
                eps=self.similarity_threshold,
                min_samples=1,
                metric='precomputed'
            )
            
            labels = clustering.fit_predict(distance_matrix)
        
        return labels
    
    def analyze_results(self):
        """
        Analyze clustering results
        """
        labels = self.cluster_embeddings()
        
        if len(labels) == 0:
            print("No data to analyze!")
            return None, None
        
        unique_labels = set(labels)
        clusters = defaultdict(list)
        
        for idx, label in enumerate(labels):
            clusters[label].append({
                'image': self.image_names[idx],
                'index': idx
            })
        
        cluster_sizes = [len(clusters[label]) for label in unique_labels]
        
        stats = {
            'total_images': len(labels),
            'unique_people': len(unique_labels),
            'duplicates': len(labels) - len(unique_labels),
            'reduction_percentage': (len(labels) - len(unique_labels)) / len(labels) * 100 if len(labels) > 0 else 0,
            'avg_images_per_person': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_images_per_person': max(cluster_sizes) if cluster_sizes else 0,
        }
        
        return stats, clusters
    
    def print_results(self, stats, clusters):
        """
        Display formatted results
        """
        print(f"\n{'='*60}")
        print(f"IMPROVED RE-IDENTIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Total images processed:        {stats['total_images']}")
        print(f"Unique people identified:      {stats['unique_people']}")
        print(f"Duplicate images removed:      {stats['duplicates']}")
        print(f"Reduction:                     {stats['reduction_percentage']:.1f}%")
        print(f"{'='*60}")
        print(f"Avg images per person:         {stats['avg_images_per_person']:.1f}")
        print(f"Max images for one person:     {stats['max_images_per_person']}")
        print(f"{'='*60}\n")
        
        print("Detailed breakdown:\n")
        for person_id in sorted(clusters.keys()):
            images = clusters[person_id]
            print(f"Person #{person_id}: {len(images)} image(s)")
            for img_data in images:
                print(f"  - {img_data['image']}")
            print()
    
    def generate_report(self, output_file="improved_reid_report.json"):
        """
        Generate JSON report
        """
        stats, clusters = self.analyze_results()
        
        if stats is None:
            return None
        
        serializable_clusters = {}
        for person_id, images in clusters.items():
            serializable_clusters[f'person_{person_id}'] = {
                'image_count': len(images),
                'images': [img['image'] for img in images]
            }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Hierarchical Clustering' if self.use_hierarchical else 'DBSCAN',
            'statistics': stats,
            'similarity_threshold': self.similarity_threshold,
            'people': serializable_clusters
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved: {output_file}\n")
        
        self.print_results(stats, clusters)
        
        return report
    
    def create_visualization(self, image_folder, output_file="improved_reid_viz.jpg"):
        """
        Create visual grid
        """
        stats, clusters = self.analyze_results()
        
        if stats is None:
            return
        
        image_folder = Path(image_folder)
        
        # Load images
        images = []
        for img_name in self.image_names:
            img_path = image_folder / img_name
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (150, 200))
                images.append(img)
            else:
                images.append(np.zeros((200, 150, 3), dtype=np.uint8))
        
        # Color map
        np.random.seed(42)
        colors = {}
        labels = self.cluster_embeddings()
        for label in set(labels):
            colors[label] = tuple(int(c) for c in np.random.randint(50, 255, 3).tolist())
        
        # Add borders and labels
        bordered_images = []
        for img, label, name in zip(images, labels, self.image_names):
            color = colors[label]
            bordered = cv2.copyMakeBorder(img, 10, 10, 10, 10, 
                                         cv2.BORDER_CONSTANT, value=color)
            
            # Add person label
            cv2.putText(bordered, f"P{label}", (15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add filename (small)
            cv2.putText(bordered, name[:15], (5, bordered.shape[0]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            
            bordered_images.append(bordered)
        
        # Create grid
        imgs_per_row = 5
        rows = []
        for i in range(0, len(bordered_images), imgs_per_row):
            row_imgs = bordered_images[i:i+imgs_per_row]
            while len(row_imgs) < imgs_per_row:
                row_imgs.append(np.ones_like(bordered_images[0]) * 255)
            rows.append(np.hstack(row_imgs))
        
        if rows:
            grid = np.vstack(rows)
            cv2.imwrite(output_file, grid)
            print(f"✓ Visualization saved: {output_file}")
    
    def run_analysis(self, image_folder, threshold=None):
        """
        Complete pipeline
        """
        if threshold is not None:
            self.similarity_threshold = threshold
        
        processed = self.process_images(image_folder)
        
        if processed == 0:
            return None
        
        self.generate_report()
        self.create_visualization(image_folder)
        
        return self.analyze_results()[0]


def test_multiple_thresholds(image_folder, thresholds=[0.3, 0.4, 0.5, 0.6]):
    """
    Test multiple thresholds to find the best one
    """
    print("\n" + "="*60)
    print("TESTING MULTIPLE THRESHOLDS")
    print("="*60)
    
    results = []
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        counter = ImprovedReIDCounter(similarity_threshold=threshold)
        counter.process_images(image_folder, verbose=False)
        stats, clusters = counter.analyze_results()
        
        if stats:
            print(f"Unique people found: {stats['unique_people']}")
            results.append((threshold, stats['unique_people']))
    
    print("\n" + "="*60)
    print("SUMMARY OF THRESHOLD TESTS:")
    print("="*60)
    for threshold, n_people in results:
        print(f"Threshold {threshold}: {n_people} unique people")
    print("="*60)
    
    return results


def main():
    """
    Main execution
    """
    print("="*60)
    print("IMPROVED PERSON RE-IDENTIFICATION")
    print("Advanced features + Better clustering")
    print("="*60)
    
    # ===== CONFIGURATION =====
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive\Desktop/human_tracking_test/visitor_database D2 4.40-7.30/visitor_database"
    
    # Test mode: Try multiple thresholds
    TEST_MODE = True  # Set False to use single threshold
    
    if TEST_MODE:
        # Try multiple thresholds to find best
        THRESHOLDS_TO_TEST = [0.3, 0.35, 0.4, 0.45, 0.5]
        test_multiple_thresholds(IMAGE_FOLDER, THRESHOLDS_TO_TEST)
    else:
        # Single threshold mode
        SIMILARITY_THRESHOLD = 0.4
        counter = ImprovedReIDCounter(similarity_threshold=SIMILARITY_THRESHOLD)
        counter.run_analysis(IMAGE_FOLDER)
    
    print("\n" + "="*60)
    print("TIP: Based on results, adjust SIMILARITY_THRESHOLD:")
    print("  • Too many people? Raise threshold (0.45-0.55)")
    print("  • Too few people? Lower threshold (0.3-0.4)")
    print("="*60)


if __name__ == "__main__":
    main()
    