"""
MEMORY-EFFICIENT Re-ID for Large Datasets
Uses incremental clustering to avoid memory issues
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import time

class MemoryEfficientReID:
    def __init__(self, similarity_threshold=0.20):
        """
        Memory-efficient Re-ID
        
        Args:
            similarity_threshold: 0.15-0.30 for large datasets
        """
        self.similarity_threshold = similarity_threshold
        self.embeddings = []
        self.image_names = []
        
    def extract_features_fast(self, image_path):
        """
        Fast feature extraction
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image = cv2.resize(image, (64, 128))
        features_list = []
        
        # HSV histograms (3 regions)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image.shape[0]
        
        for region_idx in range(3):
            start = (region_idx * h) // 3
            end = ((region_idx + 1) * h) // 3
            region = hsv[start:end, :]
            
            hist_h = cv2.calcHist([region], [0], None, [24], [0, 180])
            hist_s = cv2.calcHist([region], [1], None, [24], [0, 256])
            hist_v = cv2.calcHist([region], [2], None, [24], [0, 256])
            
            for hist in [hist_h, hist_s, hist_v]:
                features_list.append(cv2.normalize(hist, hist).flatten())
        
        # LAB color
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [24], [0, 256])
            features_list.append(cv2.normalize(hist, hist).flatten())
        
        # Texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        texture_hist = np.histogram(gradient, bins=16)[0]
        features_list.append(texture_hist / (np.sum(texture_hist) + 1e-6))
        
        # Color stats
        for channel in cv2.split(image):
            features_list.append(np.array([np.mean(channel)/255.0, np.std(channel)/255.0]))
        
        features = np.concatenate(features_list)
        return features / (np.linalg.norm(features) + 1e-6)
    
    def process_images(self, image_folder, batch_size=100):
        """
        Process images in batches
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
        print(f"Processing {total} images...")
        print(f"{'='*60}\n")
        
        processed = 0
        start_time = time.time()
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            
            elapsed = time.time() - start_time
            if processed > 0:
                eta = (elapsed / processed) * (total - processed)
                print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | ETA: {eta/60:.1f} min")
            
            for img_file in image_files[batch_start:batch_end]:
                features = self.extract_features_fast(img_file)
                
                if features is not None:
                    self.embeddings.append(features)
                    self.image_names.append(img_file.name)
                    processed += 1
        
        print(f"\n✓ Processed {processed} images in {(time.time()-start_time)/60:.1f} minutes\n")
        return processed
    
    def cosine_distance(self, vec1, vec2):
        """Calculate cosine distance between two vectors"""
        return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    
    def incremental_clustering(self):
        """
        Memory-efficient clustering - processes one image at a time
        No need to store full distance matrix!
        """
        if len(self.embeddings) == 0:
            return []
        
        print("Running incremental clustering...")
        print("(This is memory-efficient but may take a few minutes)")
        
        embeddings_array = np.array(self.embeddings)
        n_samples = len(embeddings_array)
        
        # Initialize: first image is first cluster
        cluster_representatives = [embeddings_array[0]]  # One representative per cluster
        labels = [0]  # First image belongs to cluster 0
        
        # Process each image
        for i in range(1, n_samples):
            if i % 100 == 0:
                print(f"  Clustering: {i}/{n_samples} ({i/n_samples*100:.1f}%) | "
                      f"Clusters so far: {len(cluster_representatives)}")
            
            current_embedding = embeddings_array[i]
            
            # Find closest cluster
            min_distance = float('inf')
            closest_cluster = -1
            
            for cluster_id, representative in enumerate(cluster_representatives):
                distance = self.cosine_distance(current_embedding, representative)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            # Assign to cluster or create new one
            if min_distance <= self.similarity_threshold:
                # Assign to existing cluster
                labels.append(closest_cluster)
                
                # Update cluster representative (running average)
                cluster_representatives[closest_cluster] = (
                    cluster_representatives[closest_cluster] * 0.9 + 
                    current_embedding * 0.1
                )
            else:
                # Create new cluster
                new_cluster_id = len(cluster_representatives)
                cluster_representatives.append(current_embedding)
                labels.append(new_cluster_id)
        
        print(f"✓ Clustering complete! Found {len(cluster_representatives)} unique people\n")
        
        return np.array(labels)
    
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
    
    def print_results(self, stats, clusters, show_top=20):
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
        
        if stats['unique_people'] <= show_top:
            print(f"Showing all {stats['unique_people']} people:\n")
            show_all = True
        else:
            print(f"Showing top {show_top} people (sorted by image count):\n")
            show_all = False
        
        # Sort by cluster size
        sorted_clusters = sorted(clusters.items(), 
                                key=lambda x: len(x[1]), 
                                reverse=True)
        
        for idx, (person_id, images) in enumerate(sorted_clusters[:show_top if not show_all else None]):
            print(f"Person #{person_id}: {len(images)} image(s)")
            for img in images[:3]:
                print(f"  - {img['image']}")
            if len(images) > 3:
                print(f"  ... and {len(images)-3} more")
            print()
    
    def generate_report(self, labels, output_file="memory_efficient_report.json"):
        """
        Generate JSON report
        """
        stats, clusters = self.analyze_results(labels)
        
        if stats is None:
            return None
        
        # Create report
        serializable_clusters = {}
        for person_id, images in clusters.items():
            serializable_clusters[f'person_{person_id}'] = {
                'image_count': len(images),
                'images': [img['image'] for img in images]
            }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Incremental Clustering (Memory-Efficient)',
            'statistics': stats,
            'similarity_threshold': self.similarity_threshold,
            'people': serializable_clusters
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved: {output_file}\n")
        
        self.print_results(stats, clusters)
        
        return report
    
    def create_sample_viz(self, image_folder, labels, output_file="sample_viz.jpg",
                         samples_per_person=3, max_people=12):
        """
        Create sample visualization
        """
        print("Creating sample visualization...")
        
        stats, clusters = self.analyze_results(labels)
        if stats is None:
            return
        
        image_folder = Path(image_folder)
        
        # Sort by cluster size (largest first)
        sorted_clusters = sorted(clusters.items(), 
                                key=lambda x: len(x[1]), 
                                reverse=True)
        
        # Colors
        np.random.seed(42)
        colors = {}
        for person_id, _ in sorted_clusters[:max_people]:
            colors[person_id] = tuple(int(c) for c in np.random.randint(50, 255, 3).tolist())
        
        # Collect sample images
        all_samples = []
        
        for person_id, images in sorted_clusters[:max_people]:
            for img_data in images[:samples_per_person]:
                img_path = image_folder / img_data['image']
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    img = cv2.resize(img, (120, 160))
                    color = colors[person_id]
                    
                    # Border
                    bordered = cv2.copyMakeBorder(img, 8, 8, 8, 8, 
                                                 cv2.BORDER_CONSTANT, value=color)
                    
                    # Label
                    cv2.putText(bordered, f"P{person_id}", (10, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Count
                    cv2.putText(bordered, f"({len(images)} imgs)", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    all_samples.append(bordered)
        
        if len(all_samples) == 0:
            print("No images to visualize!")
            return
        
        # Grid
        imgs_per_row = 6
        rows = []
        
        for i in range(0, len(all_samples), imgs_per_row):
            row_imgs = all_samples[i:i+imgs_per_row]
            while len(row_imgs) < imgs_per_row:
                row_imgs.append(np.ones_like(all_samples[0]) * 255)
            rows.append(np.hstack(row_imgs))
        
        if rows:
            grid = np.vstack(rows)
            cv2.imwrite(output_file, grid)
            print(f"✓ Visualization saved: {output_file}")
            print(f"  (Showing top {min(max_people, stats['unique_people'])} people)\n")
    
    def run_pipeline(self, image_folder):
        """
        Full pipeline
        """
        # Process
        processed = self.process_images(image_folder)
        if processed == 0:
            return None
        
        # Cluster
        labels = self.incremental_clustering()
        
        # Report
        self.generate_report(labels)
        
        # Viz
        self.create_sample_viz(image_folder, labels)
        
        return self.analyze_results(labels)[0]


def main():
    """
    Main
    """
    print("="*60)
    print("MEMORY-EFFICIENT RE-ID")
    print("For large datasets (no RAM issues!)")
    print("="*60)
    
    # ===== CONFIGURATION =====
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive/Desktop/human_tracking_test/Full Visitor Database over Open House (D1D2)"
    
    # For 4000+ images, use LOW threshold
    SIMILARITY_THRESHOLD = 0.13# Try: 0.15, 0.20, 0.25, 0.30
    # =========================
    
    if not Path(IMAGE_FOLDER).exists():
        print(f"ERROR: Folder not found!")
        return
    
    reid = MemoryEfficientReID(similarity_threshold=SIMILARITY_THRESHOLD)
    
    try:
        stats = reid.run_pipeline(IMAGE_FOLDER)
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print("Files:")
        print("  • memory_efficient_report.json")
        print("  • sample_viz.jpg")
        print("\nThreshold adjustment:")
        print(f"  Current: {SIMILARITY_THRESHOLD}")
        print("  • Too many people? RAISE to 0.25-0.30")
        print("  • Still grouping everyone? LOWER to 0.10-0.15")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
