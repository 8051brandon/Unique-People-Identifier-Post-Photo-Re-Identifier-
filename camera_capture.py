from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
import numpy as np
import platform
import time

class VisitorTracker:
    def __init__(self, database_folder="visitor_database", log_file="visitor_log.json"):
        self.database_folder = database_folder
        self.log_file = log_file
        self.visitors = []
        self.visitor_embeddings = {}  # Store multiple embeddings per visitor
        self.current_visitor = None
        self.visitor_count = 0
        self.similarity_threshold = 0.70  # Lower default for multi-image matching
        self.max_images_per_visitor = 5  # Store up to 5 reference images per visitor
        
        # Don't initialize YOLO here - each camera will have its own
        self.person_class_id = 0
        
        # Global track mapping (shared across cameras)
        self.track_to_visitor = {}
        self.track_last_processed = {}
        self.track_image_count = {}  # Track how many images we've captured per track
        self.active_tracks = set()
        
        # Per-camera track mapping to avoid ID conflicts
        self.camera_track_offset = 10000  # Offset to separate camera track IDs
        
        if not os.path.exists(self.database_folder):
            os.makedirs(self.database_folder)
            print(f"Created visitor database folder: {self.database_folder}")
        
        self.load_visitor_log()
    
    def get_global_track_id(self, camera_id, track_id):
        """Convert camera-specific track ID to global unique ID"""
        return track_id + (camera_id * self.camera_track_offset)
    
    def extract_body_features(self, body_crop):
        """Extract multiple features from body crop for robust matching"""
        try:
            # Resize to standard size
            resized = cv2.resize(body_crop, (128, 256))
            
            # 1. Color Histogram in HSV space (robust to lighting changes)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # 2. Spatial color distribution (divide into grid)
            grid_size = 4
            h_step = resized.shape[0] // grid_size
            w_step = resized.shape[1] // grid_size
            
            spatial_features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    grid_cell = resized[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                    mean_color = cv2.mean(grid_cell)[:3]  # BGR mean
                    spatial_features.extend(mean_color)
            
            spatial_features = np.array(spatial_features)
            
            # 3. Edge features (for body shape/structure)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Combine all features
            features = {
                'h_hist': h_hist,
                's_hist': s_hist,
                'v_hist': v_hist,
                'spatial': spatial_features,
                'edge_density': edge_density,
                'image': resized  # Store for visual comparison
            }
            
            return features
            
        except Exception as e:
            print(f"  [DEBUG] Feature extraction error: {e}")
            return None
    
    def compare_features(self, features1, features2):
        """Compare two feature sets and return similarity score (0-1)"""
        try:
            # 1. Compare color histograms using correlation
            h_similarity = cv2.compareHist(features1['h_hist'], features2['h_hist'], cv2.HISTCMP_CORREL)
            s_similarity = cv2.compareHist(features1['s_hist'], features2['s_hist'], cv2.HISTCMP_CORREL)
            v_similarity = cv2.compareHist(features1['v_hist'], features2['v_hist'], cv2.HISTCMP_CORREL)
            
            # Average histogram similarity (weight hue more heavily)
            hist_similarity = (h_similarity * 0.5 + s_similarity * 0.3 + v_similarity * 0.2)
            
            # 2. Compare spatial color distribution
            spatial_diff = np.linalg.norm(features1['spatial'] - features2['spatial'])
            max_spatial_diff = np.linalg.norm(np.ones_like(features1['spatial']) * 255)
            spatial_similarity = 1 - (spatial_diff / max_spatial_diff)
            
            # 3. Compare edge density
            edge_diff = abs(features1['edge_density'] - features2['edge_density'])
            edge_similarity = 1 - edge_diff
            
            # 4. Structural similarity (SSIM) on grayscale images
            gray1 = cv2.cvtColor(features1['image'], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(features2['image'], cv2.COLOR_BGR2GRAY)
            
            # Simple SSIM calculation
            mean1 = np.mean(gray1)
            mean2 = np.mean(gray2)
            std1 = np.std(gray1)
            std2 = np.std(gray2)
            covariance = np.mean((gray1 - mean1) * (gray2 - mean2))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) / \
                   ((mean1**2 + mean2**2 + c1) * (std1**2 + std2**2 + c2))
            
            # Combine all similarities with weights
            total_similarity = (
                hist_similarity * 0.4 +      # Color histogram (most important)
                spatial_similarity * 0.3 +    # Spatial color distribution
                ssim * 0.2 +                  # Structural similarity
                edge_similarity * 0.1         # Edge features
            )
            
            return max(0, min(1, total_similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"    Comparison error: {e}")
            return 0.0
    
    def save_visitor_image(self, frame, visitor_id, image_index):
        """Save a reference image for a visitor"""
        try:
            image_path = os.path.join(self.database_folder, f"visitor_{visitor_id}_img{image_index}.jpg")
            cv2.imwrite(image_path, frame)
            return image_path
        except Exception as e:
            print(f"  [DEBUG] Save failed: {e}")
            return None
    
    def load_visitor_log(self):
        """Load previous visitor records and their embeddings"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.visitor_count = data.get('total_visitors', 0)
                self.visitors = data.get('visitors', [])
                
                # Load embeddings for each visitor (multiple images per visitor)
                for visitor in self.visitors:
                    visitor_id = visitor['id']
                    image_paths = visitor.get('image_paths', [visitor.get('image_path')])  # Support old format
                    
                    self.visitor_embeddings[visitor_id] = []
                    
                    for image_path in image_paths:
                        if image_path and os.path.exists(image_path):
                            img = cv2.imread(image_path)
                            if img is not None:
                                features = self.extract_body_features(img)
                                if features is not None:
                                    self.visitor_embeddings[visitor_id].append(features)
                
                total_images = sum(len(emb) for emb in self.visitor_embeddings.values())
                print(f"Loaded {self.visitor_count} previous visitors from log")
                print(f"Loaded {total_images} reference images total")
        else:
            print("No previous visitor log found. Starting fresh.")
    
    def save_visitor_log(self):
        """Save visitor records to file"""
        data = {
            'total_visitors': self.visitor_count,
            'visitors': self.visitors
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, indent=4, fp=f)
    
    def is_known_visitor(self, body_crop):
        """Check if this person has visited before using body features"""
        if self.visitor_count == 0:
            print("  [DEBUG] No previous visitors")
            return False, None
        
        print(f"  [DEBUG] Checking against {self.visitor_count} visitors")
        
        # Extract features from current body crop
        current_features = self.extract_body_features(body_crop)
        if current_features is None:
            print("  [DEBUG] Could not extract features from current crop")
            return False, None
        
        best_match_id = None
        best_similarity = 0.0
        
        for visitor_id, visitor_features_list in self.visitor_embeddings.items():
            # Compare against ALL reference images for this visitor and take the best match
            max_similarity_for_visitor = 0.0
            
            for idx, visitor_features in enumerate(visitor_features_list):
                similarity = self.compare_features(current_features, visitor_features)
                if similarity > max_similarity_for_visitor:
                    max_similarity_for_visitor = similarity
            
            num_refs = len(visitor_features_list)
            print(f"    Visitor {visitor_id} ({num_refs} refs): best_similarity={max_similarity_for_visitor:.4f}")
            
            if max_similarity_for_visitor > best_similarity:
                best_similarity = max_similarity_for_visitor
                best_match_id = visitor_id
        
        # Use best match if above threshold
        if best_match_id is not None and best_similarity >= self.similarity_threshold:
            print(f"  [DEBUG] ✓ Matched to Visitor #{best_match_id}! (similarity: {best_similarity:.4f})")
            return True, best_match_id
        
        print(f"  [DEBUG] No match - new visitor (best: {best_similarity:.4f}, threshold: {self.similarity_threshold})")
        return False, None
    
    def add_new_visitor(self, frame):
        """Add a new unique visitor with initial reference image"""
        self.visitor_count += 1
        visitor_id = self.visitor_count
        
        # Save first reference image
        image_path = self.save_visitor_image(frame, visitor_id, 1)
        
        # Extract and store features
        features = self.extract_body_features(frame)
        if features is not None:
            self.visitor_embeddings[visitor_id] = [features]
        else:
            self.visitor_embeddings[visitor_id] = []
        
        visitor_info = {
            'id': visitor_id,
            'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_paths': [image_path],
            'visit_count': 1,
            'reference_image_count': 1
        }
        self.visitors.append(visitor_info)
        self.save_visitor_log()
        
        print(f"✓ New visitor #{visitor_id} registered! (1 reference image)")
        return visitor_id
    
    def add_reference_image(self, visitor_id, frame):
        """Add an additional reference image for an existing visitor"""
        # Find visitor info
        visitor_info = None
        for v in self.visitors:
            if v['id'] == visitor_id:
                visitor_info = v
                break
        
        if visitor_info is None:
            return False
        
        # Check if we already have max images
        current_count = visitor_info.get('reference_image_count', 1)
        if current_count >= self.max_images_per_visitor:
            return False
        
        # Save new reference image
        new_index = current_count + 1
        image_path = self.save_visitor_image(frame, visitor_id, new_index)
        
        if image_path:
            # Add to image paths
            visitor_info['image_paths'].append(image_path)
            visitor_info['reference_image_count'] = new_index
            
            # Extract and add features
            features = self.extract_body_features(frame)
            if features is not None:
                if visitor_id not in self.visitor_embeddings:
                    self.visitor_embeddings[visitor_id] = []
                self.visitor_embeddings[visitor_id].append(features)
            
            self.save_visitor_log()
            print(f"  ✓ Added reference image #{new_index} for Visitor #{visitor_id}")
            return True
        
        return False
    
    def update_visitor_timestamp(self, visitor_id):
        """Update the last seen time for a returning visitor"""
        for visitor in self.visitors:
            if visitor['id'] == visitor_id:
                visitor['last_seen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                visitor['visit_count'] = visitor.get('visit_count', 1) + 1
                self.save_visitor_log()
                ref_count = visitor.get('reference_image_count', 1)
                print(f"✓ Updated Visitor #{visitor_id} - Visit #{visitor['visit_count']} ({ref_count} refs)")
                break
    
    def process_person_detection(self, frame, global_track_id, bbox, frame_count, camera_id):
        """Process detected person with body recognition"""
        print(f"\n[Cam{camera_id} Track {global_track_id}] Processing...")
        
        # Check if already mapped
        if global_track_id in self.track_to_visitor:
            visitor_id = self.track_to_visitor[global_track_id]
            
            # Try to add reference image periodically for better matching
            if global_track_id not in self.track_image_count:
                self.track_image_count[global_track_id] = 1
            
            # Add reference image every 60 frames (about every 2 seconds at 30fps)
            if frame_count - self.track_last_processed.get(global_track_id, 0) >= 60:
                x1, y1, x2, y2 = map(int, bbox)
                height = y2 - y1
                width = x2 - x1
                y1_new = int(y1 + height * 0.15)
                padding_x = int(width * 0.05)
                x1 = max(0, x1 - padding_x)
                x2 = min(frame.shape[1], x2 + padding_x)
                y1 = max(0, y1_new)
                y2 = min(frame.shape[0], y2)
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size > 0:
                    self.add_reference_image(visitor_id, person_crop)
                    self.track_last_processed[global_track_id] = frame_count
                    self.track_image_count[global_track_id] += 1
            
            print(f"[Cam{camera_id} Track {global_track_id}] Mapped to Visitor #{visitor_id}")
            return visitor_id, f"Visitor #{visitor_id}", (0, 255, 0)
        
        # Check if processed recently
        if global_track_id in self.track_last_processed:
            if frame_count - self.track_last_processed[global_track_id] < 30:
                print(f"[Cam{camera_id} Track {global_track_id}] Processed too recently, skipping")
                return None, "Processing...", (255, 255, 0)
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Use full body (excluding head area to focus on clothing/body)
        height = y2 - y1
        width = x2 - x1
        
        # Skip top 15% (head) and use torso/legs for more distinctive features
        y1_new = int(y1 + height * 0.15)
        
        # Add small padding
        padding_x = int(width * 0.05)
        x1 = max(0, x1 - padding_x)
        x2 = min(frame.shape[1], x2 + padding_x)
        y1 = max(0, y1_new)
        y2 = min(frame.shape[0], y2)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0 or person_crop.shape[0] < 80 or person_crop.shape[1] < 40:
            print(f"[Cam{camera_id} Track {global_track_id}] Invalid crop size: {person_crop.shape}")
            return None, "Invalid crop", (255, 255, 0)
        
        print(f"[Cam{camera_id} Track {global_track_id}] Body crop size: {person_crop.shape}")
        
        try:
            is_known, visitor_id = self.is_known_visitor(person_crop)
            
            if is_known:
                self.track_to_visitor[global_track_id] = visitor_id
                self.track_last_processed[global_track_id] = frame_count
                self.track_image_count[global_track_id] = 1
                self.update_visitor_timestamp(visitor_id)
                
                # Add this as a new reference image for future matching
                self.add_reference_image(visitor_id, person_crop)
                
                status = f"Visitor #{visitor_id}"
                color = (0, 255, 0)
                print(f"[Cam{camera_id} Track {global_track_id}] ✓ Recognized as Visitor #{visitor_id}")
            else:
                new_visitor_id = self.add_new_visitor(person_crop)
                self.track_to_visitor[global_track_id] = new_visitor_id
                self.track_last_processed[global_track_id] = frame_count
                self.track_image_count[global_track_id] = 1
                visitor_id = new_visitor_id
                status = f"NEW Visitor #{new_visitor_id}!"
                color = (255, 0, 255)
                print(f"[Cam{camera_id} Track {global_track_id}] ✓ Registered as NEW Visitor #{new_visitor_id}")
            
            return visitor_id, status, color
            
        except Exception as e:
            print(f"[Cam{camera_id} Track {global_track_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            self.track_last_processed[global_track_id] = frame_count
            return None, "Error processing", (255, 0, 0)


class CameraProcessor:
    """Process a camera and return frames"""
    def __init__(self, camera_index, tracker, camera_label):
        self.camera_index = camera_index
        self.camera_label = camera_label
        self.tracker = tracker
        self.cap = None
        self.frame_count = 0
        self.process_interval = 20
        self.detected_persons = {}
        self.is_active = False
        
        # Each camera gets its own YOLO model instance for independent tracking
        self.yolo_model = None
        
    def initialize(self):
        """Initialize camera capture with multiple backend attempts"""
        # Try different backends in order of preference for Windows
        if platform.system() == 'Windows':
            backends_to_try = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto")
            ]
        else:
            backends_to_try = [(cv2.CAP_ANY, "Auto")]
        
        for backend, backend_name in backends_to_try:
            print(f"  Trying {self.camera_label} with {backend_name}...", end=" ")
            
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                
                # Give it a moment to initialize
                time.sleep(0.5)
                
                if self.cap.isOpened():
                    # Try to read a test frame
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        print("✓ Success!")
                        
                        # Configure camera settings
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # For secondary cameras, set additional properties
                        if self.camera_index > 0:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Initialize YOLO model for this camera
                        print(f"  Loading YOLOv8 model for {self.camera_label}...")
                        self.yolo_model = YOLO('yolov8n.pt')
                        
                        self.is_active = True
                        return True
                    else:
                        print("✗ Can't read frames")
                        self.cap.release()
                else:
                    print("✗ Can't open")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                if self.cap is not None:
                    self.cap.release()
        
        print(f"  ❌ Failed to initialize {self.camera_label} with all backends")
        return False
    
    def process_frame(self, debug_mode):
        """Process one frame and return annotated frame"""
        if not self.is_active:
            # Return blank frame with error message
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"{self.camera_label} Unavailable", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return blank
        
        ret, frame = self.cap.read()
        if not ret:
            print(f"{self.camera_label}: Failed to grab frame")
            self.is_active = False
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"{self.camera_label} Error", (180, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return blank
        
        # Run YOLO tracking with this camera's own model instance
        results = self.yolo_model.track(
            frame,
            persist=True,
            classes=[self.tracker.person_class_id],
            conf=0.5,
            verbose=False
        )
        
        current_frame_tracks = set()
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for track_id, conf, bbox in zip(track_ids, confidences, xyxy):
                global_track_id = self.tracker.get_global_track_id(self.camera_index, track_id)
                current_frame_tracks.add(global_track_id)
                
                should_process = (self.frame_count % self.process_interval == 0) or (global_track_id not in self.detected_persons)
                
                if should_process:
                    visitor_id, status, color = self.tracker.process_person_detection(
                        frame, global_track_id, bbox, self.frame_count, self.camera_index
                    )
                    self.detected_persons[global_track_id] = {
                        'visitor_id': visitor_id,
                        'status': status,
                        'color': color,
                        'bbox': bbox,
                        'conf': conf
                    }
                
                # Draw bounding box and info
                if global_track_id in self.detected_persons:
                    info = self.detected_persons[global_track_id]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), info['color'], 2)
                    
                    label = f"T{track_id}: {info['status']}"
                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                (x1 + text_width + 10, y1), info['color'], -1)
                    
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    conf_text = f"{conf:.2f}"
                    cv2.putText(frame, conf_text, (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Clean up tracks that are no longer visible
        removed_tracks = set(self.detected_persons.keys()) - current_frame_tracks
        for global_track_id in removed_tracks:
            if global_track_id in self.detected_persons:
                print(f"\n[{self.camera_label} Track {global_track_id}] Left frame")
                del self.detected_persons[global_track_id]
        
        # Display statistics UI
        ui_height = 160 if debug_mode else 120
        cv2.rectangle(frame, (10, 10), (400, ui_height), (0, 0, 0), -1)
        
        cam_text = f"{self.camera_label} (Multi-Ref Body Tracking)"
        cv2.putText(frame, cam_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
        
        status_text = f"Active Persons: {len(current_frame_tracks)}"
        cv2.putText(frame, status_text, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        count_text = f"Total Unique Visitors: {self.tracker.visitor_count}"
        cv2.putText(frame, count_text, (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if debug_mode:
            total_refs = sum(len(emb) for emb in self.tracker.visitor_embeddings.values())
            refs_text = f"Total Reference Images: {total_refs} (max {self.tracker.max_images_per_visitor}/visitor)"
            cv2.putText(frame, refs_text, (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            
            thresh_text = f"Threshold: {self.tracker.similarity_threshold:.2f}"
            cv2.putText(frame, thresh_text, (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        self.frame_count += 1
        return frame
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_active = False


def detect_available_cameras():
    """Detect all available cameras"""
    print("Detecting available cameras...")
    available_cameras = []
    
    # On Windows, try DirectShow first, then Media Foundation
    if platform.system() == 'Windows':
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [cv2.CAP_ANY]
    
    for backend in backends:
        backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF" if backend == cv2.CAP_MSMF else "Auto"
        
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Check if not already added
                        if not any(c['index'] == i for c in available_cameras):
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            
                            camera_info = {
                                'index': i,
                                'label': f"Camera {i}",
                                'resolution': f"{int(width)}x{int(height)}",
                                'backend': backend,
                                'backend_name': backend_name
                            }
                            available_cameras.append(camera_info)
                            print(f"  ✓ Camera {i} detected ({backend_name}, {int(width)}x{int(height)})")
                cap.release()
            except:
                pass
        
        # If we found cameras with this backend, don't try others
        if available_cameras:
            break
    
    return available_cameras


def track_visitors():
    tracker = VisitorTracker()
    
    print("\n" + "="*60)
    print("Multi-Reference Body-Based Visitor Tracking")
    print("="*60)
    print(f"Current unique visitors: {tracker.visitor_count}")
    print(f"Similarity threshold: {tracker.similarity_threshold}")
    print(f"Max reference images per visitor: {tracker.max_images_per_visitor}")
    print("Method: Multi-angle color + structural matching")
    print("="*60 + "\n")
    
    # Detect available cameras
    available_cameras = detect_available_cameras()
    
    if len(available_cameras) == 0:
        print("❌ Error: No cameras detected!")
        return
    
    print(f"\n{'='*60}")
    print(f"✓ Found {len(available_cameras)} camera(s):")
    for i, cam in enumerate(available_cameras):
        print(f"  [{i}] {cam['label']} - {cam['resolution']}")
    print("="*60)
    
    # Let user choose cameras
    if len(available_cameras) == 1:
        selected_indices = [0]
        print(f"\nUsing only available camera: {available_cameras[0]['label']}")
    else:
        print("\nSelect cameras to use:")
        print("  Enter indices separated by commas (e.g., '0,1' or just '0')")
        print("  Or press Enter to use all cameras")
        
        while True:
            user_input = input("  Your choice: ").strip()
            
            if user_input == "":
                selected_indices = list(range(len(available_cameras)))
                break
            
            try:
                selected_indices = [int(x.strip()) for x in user_input.split(',')]
                if all(0 <= idx < len(available_cameras) for idx in selected_indices):
                    break
                else:
                    print(f"  Invalid selection. Choose from 0 to {len(available_cameras)-1}")
            except ValueError:
                print("  Invalid input. Use numbers separated by commas.")
    
    selected_cameras = [available_cameras[i] for i in selected_indices]
    
    print(f"\n✓ Selected camera(s):")
    for cam in selected_cameras:
        print(f"  - {cam['label']}")
    print("\nInitializing cameras...")
    
    # Initialize camera processors
    camera_processors = []
    
    for cam_info in selected_cameras:
        processor = CameraProcessor(cam_info['index'], tracker, cam_info['label'])
        if processor.initialize():
            camera_processors.append(processor)
    
    if len(camera_processors) == 0:
        print("\n❌ Error: No cameras could be initialized!")
        return
    
    print(f"\n✓ {len(camera_processors)} camera(s) ready!")
    print("\nControls:")
    print("  Q: Quit")
    print("  R: Reset visitor database")
    print("  D: Toggle debug mode")
    print("  +/-: Adjust similarity threshold")
    print("\nFeature: System captures up to 5 reference images per visitor")
    print("         from different angles/times for better matching!")
    print("="*60 + "\n")
    
    debug_mode = True
    
    while True:
        frames = []
        
        # Get frames from all cameras
        for processor in camera_processors:
            frame = processor.process_frame(debug_mode)
            frames.append(frame)
        
        # Resize all frames to same size for consistent display
        # Change these numbers to make windows bigger/smaller
        target_height = 600   # Height of each camera window
        target_width = 1100   # Width of each camera window
        
        resized_frames = []
        for frame in frames:
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                resized = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(resized)
            else:
                resized_frames.append(frame)
        
        frames = resized_frames
        
        # Combine frames
        if len(frames) == 1:
            combined_frame = frames[0]
        elif len(frames) == 2:
            combined_frame = np.hstack(frames)
        else:
            num_cams = len(frames)
            cols = 2
            rows = (num_cams + cols - 1) // cols
            
            while len(frames) < rows * cols:
                blank = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                frames.append(blank)
            
            grid_rows = []
            for r in range(rows):
                row_frames = frames[r*cols:(r+1)*cols]
                grid_rows.append(np.hstack(row_frames))
            
            combined_frame = np.vstack(grid_rows)
        
        # Add global controls at the bottom
        bottom_bar_height = 40
        bottom_bar = np.zeros((bottom_bar_height, combined_frame.shape[1], 3), dtype=np.uint8)
        
        total_refs = sum(len(emb) for emb in tracker.visitor_embeddings.values())
        control_text = f"Q:Quit | R:Reset | D:Debug | +/-:Threshold({tracker.similarity_threshold:.2f}) | Refs:{total_refs}"
        cv2.putText(bottom_bar, control_text, (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        display_frame = np.vstack([combined_frame, bottom_bar])
        
        cv2.imshow('Multi-Reference Body Tracking', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nShutting down...")
            break
        elif key == ord('r'):
            print("\n" + "="*60)
            response = input("Reset visitor count? (yes/no): ")
            if response.lower() == 'yes':
                tracker.visitor_count = 0
                tracker.visitors = []
                tracker.visitor_embeddings = {}
                tracker.track_to_visitor = {}
                tracker.track_last_processed = {}
                tracker.track_image_count = {}
                tracker.save_visitor_log()
                
                for processor in camera_processors:
                    processor.detected_persons = {}
                
                print("✓ Visitor count reset!")
            print("="*60 + "\n")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            tracker.similarity_threshold = min(1.0, tracker.similarity_threshold + 0.05)
            print(f"Threshold increased to {tracker.similarity_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            tracker.similarity_threshold = max(0.0, tracker.similarity_threshold - 0.05)
            print(f"Threshold decreased to {tracker.similarity_threshold:.2f}")
    
    # Release cameras
    for processor in camera_processors:
        processor.release()
    
    cv2.destroyAllWindows()
    
    # Cleanup temp files
    for temp_file in ['temp_frame.jpg', 'temp_align.jpg']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    for f in os.listdir('.'):
        if (f.startswith('temp_cam') or f.startswith('temp_track_')) and f.endswith('.jpg'):
            os.remove(f)
    
    total_refs = sum(len(emb) for emb in tracker.visitor_embeddings.values())
    print(f"\n{'='*60}")
    print(f"Session Summary:")
    print(f"Total Unique Visitors: {tracker.visitor_count}")
    print(f"Total Reference Images: {total_refs}")
    print(f"Cameras Used: {len(camera_processors)}")
    print(f"Final Threshold: {tracker.similarity_threshold:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    track_visitors()