"""
RELIABILITY CHECKER for Re-ID Results
Analyzes clustering quality and helps verify accuracy
"""

import cv2
import numpy as np
from pathlib import Path
import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

class ReliabilityChecker:
    def __init__(self, report_file="memory_efficient_report.json"):
        """
        Load and analyze Re-ID results
        """
        with open(report_file, 'r') as f:
            self.report = json.load(f)
        
        self.stats = self.report['statistics']
        self.people = self.report['people']
        self.threshold = self.report['similarity_threshold']
    
    def print_overview(self):
        """
        Print overview statistics
        """
        print("\n" + "="*60)
        print("RELIABILITY ANALYSIS OVERVIEW")
        print("="*60)
        print(f"Total images:           {self.stats['total_images']:,}")
        print(f"Unique people found:    {self.stats['unique_people']:,}")
        print(f"Duplicates removed:     {self.stats['duplicates']:,}")
        print(f"Reduction:              {self.stats['reduction_percentage']:.1f}%")
        print(f"Threshold used:         {self.threshold}")
        print("="*60 + "\n")
    
    def analyze_cluster_distribution(self):
        """
        Analyze how images are distributed across clusters
        """
        print("CLUSTER SIZE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        cluster_sizes = [person['image_count'] for person in self.people.values()]
        
        print(f"\nCluster Statistics:")
        print(f"  Total clusters: {len(cluster_sizes)}")
        print(f"  Mean images per person: {np.mean(cluster_sizes):.1f}")
        print(f"  Median images per person: {np.median(cluster_sizes):.0f}")
        print(f"  Std deviation: {np.std(cluster_sizes):.1f}")
        print(f"  Min images: {np.min(cluster_sizes)}")
        print(f"  Max images: {np.max(cluster_sizes)}")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(cluster_sizes, p)
            print(f"  {p}th percentile: {val:.0f} images")
        
        # Count by size ranges
        print(f"\nDistribution by cluster size:")
        ranges = [
            (1, 1, "Singletons (1 image)"),
            (2, 5, "Small (2-5 images)"),
            (6, 10, "Medium (6-10 images)"),
            (11, 20, "Large (11-20 images)"),
            (21, 50, "Very Large (21-50 images)"),
            (51, float('inf'), "Huge (50+ images)")
        ]
        
        for min_size, max_size, label in ranges:
            count = sum(1 for size in cluster_sizes if min_size <= size <= max_size)
            pct = count / len(cluster_sizes) * 100
            total_imgs = sum(size for size in cluster_sizes if min_size <= size <= max_size)
            print(f"  {label:25} {count:6} clusters ({pct:5.1f}%) → {total_imgs:,} images")
        
        # Create histogram
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(cluster_sizes, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Images per Person')
        plt.ylabel('Number of People')
        plt.title('Distribution of Cluster Sizes')
        plt.axvline(np.mean(cluster_sizes), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(cluster_sizes):.1f}')
        plt.axvline(np.median(cluster_sizes), color='g', linestyle='--', 
                   label=f'Median: {np.median(cluster_sizes):.0f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(cluster_sizes, bins=50, cumulative=True, density=True,
                edgecolor='black', alpha=0.7)
        plt.xlabel('Images per Person')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: cluster_distribution.png\n")
        
        return cluster_sizes
    
    def detect_suspicious_clusters(self):
        """
        Find clusters that might be wrong
        """
        print("SUSPICIOUS CLUSTER DETECTION")
        print("="*60)
        
        cluster_sizes = [(person_id, person['image_count']) 
                        for person_id, person in self.people.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        total_images = self.stats['total_images']
        
        # 1. Extremely large clusters (possible over-grouping)
        print("\n1. VERY LARGE CLUSTERS (possible over-grouping):")
        threshold_large = max(100, total_images * 0.05)  # 5% of total or 100
        
        large_clusters = [(pid, size) for pid, size in cluster_sizes if size > threshold_large]
        
        if large_clusters:
            print(f"   Found {len(large_clusters)} suspiciously large clusters:")
            for pid, size in large_clusters[:10]:
                pct = size / total_images * 100
                print(f"   - {pid}: {size:,} images ({pct:.1f}% of total)")
                # Show some image names
                images = self.people[pid]['images'][:5]
                print(f"     Sample images: {', '.join(images)}")
            
            if len(large_clusters) > 10:
                print(f"   ... and {len(large_clusters)-10} more")
            
            print(f"\n   ⚠️  WARNING: These clusters might be grouping different people!")
            print(f"   → Consider LOWERING threshold (try {self.threshold * 0.8:.3f})")
        else:
            print("   ✓ No suspiciously large clusters detected")
        
        # 2. Too many singletons (possible under-grouping)
        print("\n2. SINGLETON ANALYSIS (clusters with only 1 image):")
        singletons = [size for _, size in cluster_sizes if size == 1]
        singleton_count = len(singletons)
        singleton_pct = singleton_count / len(cluster_sizes) * 100
        
        print(f"   Singletons: {singleton_count:,} ({singleton_pct:.1f}% of clusters)")
        
        if singleton_pct > 70:
            print(f"   ⚠️  WARNING: Too many singletons!")
            print(f"   → Most people only appear once - might be under-grouping")
            print(f"   → Consider RAISING threshold (try {self.threshold * 1.2:.3f})")
        elif singleton_pct > 50:
            print(f"   ⚡ CAUTION: Many singletons")
            print(f"   → This might be correct if visitors truly appear once")
            print(f"   → Or might indicate slight under-grouping")
        else:
            print(f"   ✓ Singleton ratio looks reasonable")
        
        # 3. Very similar cluster sizes (suspicious pattern)
        print("\n3. PATTERN ANALYSIS:")
        if len(cluster_sizes) > 10:
            top_10_sizes = [size for _, size in cluster_sizes[:10]]
            size_std = np.std(top_10_sizes)
            size_mean = np.mean(top_10_sizes)
            
            if size_std / size_mean < 0.1:  # Very similar sizes
                print(f"   ⚠️  Top clusters have very similar sizes")
                print(f"   → Sizes: {top_10_sizes}")
                print(f"   → This might indicate systematic grouping issues")
            else:
                print(f"   ✓ Cluster sizes vary naturally")
        
        print()
    
    def sample_visual_check(self, image_folder, n_samples=5, output_folder="reliability_check"):
        """
        Create visual samples for manual verification
        """
        print("CREATING VISUAL VERIFICATION SAMPLES")
        print("="*60)
        
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        # Get cluster sizes
        cluster_list = [(pid, person['image_count'], person['images']) 
                       for pid, person in self.people.items()]
        cluster_list.sort(key=lambda x: x[1], reverse=True)
        
        # Sample different types of clusters
        samples_to_check = []
        
        # 1. Largest clusters (might be over-grouped)
        print("\nSampling largest clusters...")
        samples_to_check.extend([
            (pid, imgs, "LARGE") for pid, size, imgs in cluster_list[:3]
        ])
        
        # 2. Medium clusters
        print("Sampling medium-sized clusters...")
        medium = [c for c in cluster_list if 10 <= c[1] <= 30]
        if medium:
            samples_to_check.extend([
                (pid, imgs, "MEDIUM") for pid, size, imgs in random.sample(medium, min(2, len(medium)))
            ])
        
        # 3. Small clusters
        print("Sampling small clusters...")
        small = [c for c in cluster_list if 3 <= c[1] <= 9]
        if small:
            samples_to_check.extend([
                (pid, imgs, "SMALL") for pid, size, imgs in random.sample(small, min(2, len(small)))
            ])
        
        # Create visual grids
        for idx, (person_id, images, cluster_type) in enumerate(samples_to_check, 1):
            print(f"  Creating sample {idx}/{len(samples_to_check)}: {person_id} ({cluster_type}, {len(images)} images)")
            
            # Load up to 12 images
            sample_images = []
            for img_name in images[:12]:
                img_path = image_folder / img_name
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    img = cv2.resize(img, (150, 200))
                    
                    # Add filename
                    cv2.putText(img, img_name[:20], (5, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(img, img_name[:20], (5, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    
                    sample_images.append(img)
            
            if sample_images:
                # Create grid
                imgs_per_row = 4
                rows = []
                
                for i in range(0, len(sample_images), imgs_per_row):
                    row = sample_images[i:i+imgs_per_row]
                    while len(row) < imgs_per_row:
                        row.append(np.ones_like(sample_images[0]) * 255)
                    rows.append(np.hstack(row))
                
                grid = np.vstack(rows)
                
                # Add title
                title_height = 50
                title_bar = np.ones((title_height, grid.shape[1], 3), dtype=np.uint8) * 240
                cv2.putText(title_bar, 
                          f"{person_id} - {cluster_type} cluster - {len(images)} total images",
                          (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                final = np.vstack([title_bar, grid])
                
                output_file = output_folder / f"sample_{idx}_{cluster_type.lower()}_{person_id.replace('person_', '')}.jpg"
                cv2.imwrite(str(output_file), final)
        
        print(f"\n✓ Created {len(samples_to_check)} verification samples in '{output_folder}/'")
        print(f"\n📋 MANUAL CHECK:")
        print(f"   1. Open the '{output_folder}/' folder")
        print(f"   2. Look at each sample grid")
        print(f"   3. Ask: Do these images look like the SAME person?")
        print(f"   4. If NO → threshold is too high (groups different people)")
        print(f"   5. If YES → threshold is working well!")
        print()
    
    def recommend_threshold_adjustment(self):
        """
        Based on analysis, recommend threshold changes
        """
        print("THRESHOLD ADJUSTMENT RECOMMENDATIONS")
        print("="*60)
        
        cluster_sizes = [person['image_count'] for person in self.people.values()]
        
        # Metrics
        singleton_pct = sum(1 for s in cluster_sizes if s == 1) / len(cluster_sizes) * 100
        large_cluster_pct = sum(1 for s in cluster_sizes if s > 100) / len(cluster_sizes) * 100
        avg_size = np.mean(cluster_sizes)
        max_size = max(cluster_sizes)
        
        print(f"\nCurrent threshold: {self.threshold}")
        print(f"\nKey metrics:")
        print(f"  - Singleton %: {singleton_pct:.1f}%")
        print(f"  - Avg cluster size: {avg_size:.1f}")
        print(f"  - Max cluster size: {max_size}")
        print(f"  - Large clusters (>100): {large_cluster_pct:.1f}%")
        
        recommendations = []
        
        # Too many singletons
        if singleton_pct > 70:
            new_threshold = min(self.threshold * 1.3, 0.35)
            recommendations.append({
                'issue': 'Too many singletons',
                'suggestion': f'RAISE threshold to {new_threshold:.3f}',
                'reasoning': 'Most clusters have only 1 image - might be splitting same person'
            })
        
        # Very large clusters
        if max_size > self.stats['total_images'] * 0.1:
            new_threshold = max(self.threshold * 0.7, 0.08)
            recommendations.append({
                'issue': 'Very large clusters detected',
                'suggestion': f'LOWER threshold to {new_threshold:.3f}',
                'reasoning': f'Largest cluster has {max_size} images - might be grouping different people'
            })
        
        # High average size
        if avg_size > 50:
            new_threshold = max(self.threshold * 0.8, 0.10)
            recommendations.append({
                'issue': 'High average cluster size',
                'suggestion': f'LOWER threshold to {new_threshold:.3f}',
                'reasoning': 'Average person has many images - might be over-grouping'
            })
        
        # Balanced
        if not recommendations and 20 <= singleton_pct <= 60 and avg_size < 30:
            recommendations.append({
                'issue': 'Results look balanced',
                'suggestion': 'Current threshold seems good!',
                'reasoning': 'Cluster distribution appears reasonable'
            })
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['issue']}")
            print(f"   → {rec['suggestion']}")
            print(f"   Reason: {rec['reasoning']}")
        
        print("\n" + "="*60 + "\n")
    
    def run_full_analysis(self, image_folder):
        """
        Run complete reliability check
        """
        self.print_overview()
        cluster_sizes = self.analyze_cluster_distribution()
        self.detect_suspicious_clusters()
        self.sample_visual_check(image_folder)
        self.recommend_threshold_adjustment()
        
        print("="*60)
        print("RELIABILITY CHECK COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  • cluster_distribution.png - Distribution analysis")
        print("  • reliability_check/ - Visual samples for manual verification")
        print("\n Next steps:")
        print("  1. Look at cluster_distribution.png")
        print("  2. Manually check images in reliability_check/ folder")
        print("  3. Adjust threshold based on recommendations")
        print("  4. Re-run memory_efficient_reid.py with new threshold")
        print("="*60)


def main():
    """
    Run reliability checker
    """
    print("="*60)
    print("RE-ID RELIABILITY CHECKER")
    print("="*60)
    
    # Configuration
    IMAGE_FOLDER = "C:/Users/ltcou/OneDrive/Desktop/human_tracking_test/Full Visitor Database over Open House (D1D2)"
    REPORT_FILE = "memory_efficient_report.json"
    
    if not Path(REPORT_FILE).exists():
        print(f"ERROR: Report file not found: {REPORT_FILE}")
        print("Run memory_efficient_reid.py first to generate the report!")
        return
    
    if not Path(IMAGE_FOLDER).exists():
        print(f"ERROR: Image folder not found: {IMAGE_FOLDER}")
        return
    
    checker = ReliabilityChecker(REPORT_FILE)
    checker.run_full_analysis(IMAGE_FOLDER)


if __name__ == "__main__":
    main()
