"""
PRODUCTION Re-ID for Large Datasets
Optimized for 1000+ images with proper memory management
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import time

class ProductionReID:
    def __init__(self, similarity_threshold=0.2964):
        """
        Production Re-ID optimized for large datasets
        
        Args:
            similarity_threshold: MUCH LOWER for large datasets! (0.15-0.30)
                                 Lower = stricter = fewer false matches
        """
        self.similarity_threshold = similarity_threshold
        self.embeddings = []
        self.image_names = []
        
    def extract_features_fast(self, image_path):
        """
        Fast but effective feature extraction
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize for speed
        image = cv2.resize(image, (64, 128))
        
        features_list = []
        
        # HSV color histograms (3 vertical regions)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image.shape[0]
        
        for region_idx in range(3):
            start = (region_idx * h) // 3
            end = ((region_idx + 1) * h) // 3
            region = hsv[start:end, :]
            
            # Coarser histograms for speed
            hist_h = cv2.calcHist([region], [0], None, [24], [0, 180])
            hist_s = cv2.calcHist([region], [1], None, [24], [0, 256])
            hist_v = cv2.calcHist([region], [2], None, [24], [0, 256])
            
            for hist in [hist_h, hist_s, hist_v]:
                hist_norm = cv2.normalize(hist, hist).flatten()
                features_list.append(hist_norm)
        
        # LAB color (better for subtle differences)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [24], [0, 256])
            hist_norm = cv2.normalize(hist, hist).flatten()
            features_list.append(hist_norm)
        
        # Texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        texture_hist = np.histogram(gradient, bins=16)[0]
        texture_hist = texture_hist / (np.sum(texture_hist) + 1e-6)
        features_list.append(texture_hist)
        
        # Edge density per region
        edges = cv2.Canny(gray, 50, 150)
        for region_idx in range(3):
            start = (region_idx * h) // 3
            end = ((region_idx + 1) * h) // 3
            region_edges = edges[start:end, :]
            edge_density = np.sum(region_edges > 0) / region_edges.size
            features_list.append(np.array([edge_density]))
        
        # Color statistics
        for channel in cv2.split(image):
            mean = np.mean(channel) / 255.0
            std = np.std(channel) / 255.0
            features_list.append(np.array([mean, std]))
        
        # Combine and normalize
        features = np.concatenate(features_list)
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def process_images_batch(self, image_folder, batch_size=100, verbose=True):
        """
        Process images in batches with progress tracking
        """
        image_folder = Path(image_folder)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(image_folder.glob(ext)))
        
        image_files = sorted(image_files)
        total = len(image_files)
        
        if total == 0:
            print("ERROR: No images found!")
            return 0
        
        print(f"\n{'='*60}")
        print(f"Processing {total} images in batches of {batch_size}")
        print(f"{'='*60}\n")
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_files = image_files[batch_start:batch_end]
            
            if verbose:
                elapsed = time.time() - start_time
                if processed > 0:
                    eta = (elapsed / processed) * (total - processed)
                    print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                          f"ETA: {eta/60:.1f} min")
            
            for img_file in batch_files:
                features = self.extract_features_fast(img_file)
                
                if features is not None:
                    self.embeddings.append(features)
                    self.image_names.append(img_file.name)
                    processed += 1
                else:
                    failed += 1
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Processed: {processed}")
        print(f"Failed: {failed}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Speed: {processed/elapsed:.1f} images/sec")
        print(f"{'='*60}\n")
        
        return processed
    
    def cluster_hierarchical(self):
        """
        Hierarchical clustering (better for large datasets)
        """
        if len(self.embeddings) == 0:
            return []
        
        print("Calculating distance matrix...")
        embeddings_array = np.array(self.embeddings)
        distance_matrix = cosine_distances(embeddings_array)
        
        print(f"Clustering with threshold {self.similarity_threshold}...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.similarity_threshold,
            metric='precomputed',
            linkage='average'  # Average linkage works well
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        return labels
    
    def analyze_results(self, labels):
        """
        Analyze clustering results
        """
        if len(labels) == 0:
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
            'reduction_percentage': (len(labels) - len(unique_labels)) / len(labels) * 100,
            'avg_images_per_person': np.mean(cluster_sizes),
            'max_images_per_person': max(cluster_sizes),
            'min_images_per_person': min(cluster_sizes),
        }
        
        return stats, clusters
    
    def print_results(self, stats, clusters, show_details=True):
        """
        Display results
        """
        print(f"\n{'='*60}")
        print(f"RE-IDENTIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Total images:                  {stats['total_images']:,}")
        print(f"Unique people identified:      {stats['unique_people']:,}")
        print(f"Duplicate images removed:      {stats['duplicates']:,}")
        print(f"Reduction:                     {stats['reduction_percentage']:.1f}%")
        print(f"{'='*60}")
        print(f"Avg images per person:         {stats['avg_images_per_person']:.1f}")
        print(f"Max images for one person:     {stats['max_images_per_person']:,}")
        print(f"Min images for one person:     {stats['min_images_per_person']:,}")
        print(f"{'='*60}\n")
        
        if show_details and stats['unique_people'] <= 50:
            print("Breakdown by person (showing first 50):\n")
            for person_id in sorted(list(clusters.keys())[:50]):
                images = clusters[person_id]
                print(f"Person #{person_id}: {len(images)} image(s)")
                # Show first 5 images
                for img in images[:5]:
                    print(f"  - {img['image']}")
                if len(images) > 5:
                    print(f"  ... and {len(images)-5} more")
                print()
        elif stats['unique_people'] > 50:
            print(f"Note: Too many people to show details ({stats['unique_people']} people)")
            print("Check the JSON report for full breakdown")
    
    def generate_report(self, labels, output_file="production_reid_report.json"):
        """
        Generate comprehensive report
        """
        stats, clusters = self.analyze_results(labels)
        
        if stats is None:
            return None
        
        # Prepare serializable data
        serializable_clusters = {}
        for person_id, images in clusters.items():
            serializable_clusters[f'person_{person_id}'] = {
                'image_count': len(images),
                'images': [img['image'] for img in images]
            }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'similarity_threshold': self.similarity_threshold,
            'people': serializable_clusters
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved: {output_file}\n")
        
        self.print_results(stats, clusters, show_details=(stats['unique_people'] <= 20))
        
        return report
    
    def create_sample_visualization(self, image_folder, labels, 
                                   output_file="sample_visualization.jpg",
                                   max_samples_per_person=3,
                                   max_people=10):
        """
        Create visualization with samples (not all 4000 images!)
        """
        print("\nCreating sample visualization...")
        
        stats, clusters = self.analyze_results(labels)
        
        if stats is None:
            return
        
        image_folder = Path(image_folder)
        
        # Colors
        np.random.seed(42)
        colors = {}
        for person_id in clusters.keys():
            colors[person_id] = tuple(int(c) for c in np.random.randint(50, 255, 3).tolist())
        
        # Sample images per person
        sample_images = []
        sample_labels = []
        sample_names = []
        
        for person_id in sorted(list(clusters.keys())[:max_people]):
            images = clusters[person_id][:max_samples_per_person]
            
            for img_data in images:
                img_path = image_folder / img_data['image']
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    img = cv2.resize(img, (120, 160))
                    
                    # Add border and label
                    color = colors[person_id]
                    bordered = cv2.copyMakeBorder(img, 8, 8, 8, 8, 
                                                 cv2.BORDER_CONSTANT, value=color)
                    
                    # Add text
                    cv2.putText(bordered, f"P{person_id}", (10, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    sample_images.append(bordered)
                    sample_labels.append(person_id)
                    sample_names.append(img_data['image'][:15])
        
        if len(sample_images) == 0:
            print("No images to visualize!")
            return
        
        # Create grid
        imgs_per_row = 6
        rows = []
        
        for i in range(0, len(sample_images), imgs_per_row):
            row_imgs = sample_images[i:i+imgs_per_row]
            
            # Pad row
            while len(row_imgs) < imgs_per_row:
                row_imgs.append(np.ones_like(sample_images[0]) * 255)
            
            rows.append(np.hstack(row_imgs))
        
        if rows:
            grid = np.vstack(rows)
            cv2.imwrite(output_file, grid)
            print(f"✓ Sample visualization saved: {output_file}")
            print(f"  (Showing {len(sample_images)} sample images from first {min(max_people, len(clusters))} people)")
    
    def run_full_pipeline(self, image_folder):
        """
        Complete Re-ID pipeline
        """
        # Process
        processed = self.process_images_batch(image_folder)
        
        if processed == 0:
            return None
        
        # Cluster
        labels = self.cluster_hierarchical()
        
        # Report
        self.generate_report(labels)
        
        # Visualization
        self.create_sample_visualization(image_folder, labels)
        
        return self.analyze_results(labels)[0]


def main():
    """
    Main execution
    """
    print("="*60)
    print("PRODUCTION RE-ID FOR LARGE DATASETS")
    print("="*60)
    
    # ===== CONFIGURATION =====
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive/Desktop/human_tracking_test/Full Visitor Database over Open House (D1D2)"
        
    # CRITICAL: For 4000+ images, use MUCH LOWER threshold!
    # The more images you have, the lower the threshold should be
    SIMILARITY_THRESHOLD = 0.20  # Start here for 4000+ images
    
    # For initial testing, you can limit to first N images
    # Set to None to process all
    # MAX_IMAGES = 500  # Test with 500 first
    # =========================
    
    if not Path(IMAGE_FOLDER).exists():
        print(f"ERROR: Folder not found!")
        return
    
    # Run
    reid = ProductionReID(similarity_threshold=SIMILARITY_THRESHOLD)
    
    try:
        stats = reid.run_full_pipeline(IMAGE_FOLDER)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print("Files generated:")
        print("  • production_reid_report.json - Full results")
        print("  • sample_visualization.jpg - Sample images")
        print("\nAdjustment guide for 4000+ images:")
        print("  • Way too many people? LOWER threshold (try 0.15)")
        print("  • Still grouping everyone? LOWER threshold (try 0.10)")
        print("  • Different people merged? RAISE threshold (try 0.25-0.30)")
        print("\nNote: With 4000+ images, thresholds are MUCH lower than")
        print("with small datasets. 0.15-0.25 is typical for large datasets.")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
