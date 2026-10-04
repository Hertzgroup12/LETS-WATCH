"""
Vehicle Detector Module

Uses YOLOv8 (You Only Look Once v8) for real-time detection of:
- Vehicles (cars, motorcycles, trucks)
- Pedestrians
- License plates (in images)

YOLOv8 is state-of-the-art object detection model that balances
speed and accuracy, making it perfect for real-time traffic monitoring
"""

from ultralytics import YOLO
from config import Config
import cv2
import numpy as np

class VehicleDetector:
    """
    Detects vehicles and relevant objects using YOLOv8
    """
    
    def __init__(self):
        """
        Initialize YOLOv8 model
        Uses nano model (yolov8n) for faster inference on edge devices
        """
        try:
            # Load pretrained YOLOv8 nano model
            # Models available: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
            self.model = YOLO(Config.YOLO_MODEL_PATH)
            print(f"✓ YOLOv8 model loaded: {Config.YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"✗ Error loading YOLOv8 model: {e}")
            raise
    
    def detect_vehicles(self, image):
        """
        Detect vehicles in the given image
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)
        
        Returns:
            dict: Contains detection results with:
                - 'vehicles': list of detected vehicle boxes
                - 'persons': list of detected pedestrians
                - 'raw_results': raw YOLO output for advanced analysis
        """
        try:
            # Run YOLO detection on image
            # conf = confidence threshold (0.5 by default)
            results = self.model(image, conf=Config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            
            vehicles = []
            persons = []
            
            # Extract detections from results
            if results and len(results) > 0:
                detections = results[0]  # Get first (and usually only) result
                
                # Iterate through all detected objects
                for detection in detections.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence score
                    confidence = float(detection.conf[0])
                    
                    # Get class ID (which type of object)
                    class_id = int(detection.cls[0])
                    
                    # Get class name from model
                    class_name = detections.names[class_id]
                    
                    # Package detection data
                    detection_info = {
                        'bbox': (x1, y1, x2, y2),  # Bounding box coordinates
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    # Categorize based on class (COCO dataset classes)
                    # Car class IDs: 2 (car), 5 (bus), 7 (truck), 3 (motorcycle)
                    if class_id in [2, 5, 7, 3]:  # Vehicle classes
                        vehicles.append(detection_info)
                    elif class_id == 0:  # Person class
                        persons.append(detection_info)
            
            return {
                'vehicles': vehicles,
                'persons': persons,
                'raw_results': results,
                'detection_count': len(vehicles) + len(persons)
            }
            
        except Exception as e:
            print(f"✗ Error in vehicle detection: {e}")
            return {
                'vehicles': [],
                'persons': [],
                'raw_results': None,
                'detection_count': 0,
                'error': str(e)
            }
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image for visualization
        
        Args:
            image (numpy.ndarray): Input image
            detections (dict): Detection results from detect_vehicles()
        
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        try:
            img_copy = image.copy()
            
            # Draw vehicle detections in green
            for vehicle in detections['vehicles']:
                x1, y1, x2, y2 = vehicle['bbox']
                confidence = vehicle['confidence']
                class_name = vehicle['class_name']
                
                # Draw green rectangle for vehicles
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put label with class name and confidence
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    img_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # Draw person detections in blue
            for person in detections['persons']:
                x1, y1, x2, y2 = person['bbox']
                confidence = person['confidence']
                
                # Draw blue rectangle for persons
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Put label
                label = f"Person: {confidence:.2f}"
                cv2.putText(
                    img_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
            
            return img_copy
            
        except Exception as e:
            print(f"✗ Error drawing detections: {e}")
            return image
    
    def extract_vehicle_roi(self, image, detection):
        """
        Extract Region of Interest (ROI) for a detected vehicle
        Useful for further processing like number plate recognition
        
        Args:
            image (numpy.ndarray): Original image
            detection (dict): Single detection from detect_vehicles()
        
        Returns:
            numpy.ndarray: Cropped image region containing the vehicle
        """
        try:
            x1, y1, x2, y2 = detection['bbox']
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            roi = image[y1:y2, x1:x2]
            return roi
            
        except Exception as e:
            print(f"✗ Error extracting ROI: {e}")
            return None
