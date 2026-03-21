import cv2
import time
import numpy as np
import winsound

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    has_fiftyone = True
except ImportError:
    has_fiftyone = False

def initialize_model():
    print("Loading Pose Estimation Model...")
    if has_fiftyone:
        try:
            model = foz.load_zoo_model("isaac-0.2-pose-estimation")
            return model, False
        except Exception as e:
            print(f"Warning: Model not found in zoo ({e}). Using mock fallback mode for demo.")
            return None, True
    else:
        print("Warning: FiftyOne not imported. Using mock fallback mode for demo.")
        return None, True

def calculate_spine_angle(neck, pelvis):
    dx = pelvis[0] - neck[0]
    dy = pelvis[1] - neck[1]
    return np.degrees(np.arctan2(abs(dx), dy))

def main():
    model, is_mock = initialize_model()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    THRESHOLD_ANGLE = 20.0
    ALERT_DELAY = 3.0
    bad_posture_start_time = None
    
    print("PosturePartner is now actively monitoring your posture. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        angle = 15.0  # Default good posture
        
        if not is_mock:
            predictions = model.predict(frame)
            if predictions and hasattr(predictions, 'keypoints') and len(predictions.keypoints) > 0:
                points = predictions.keypoints[0].points 
                if len(points) >= 13:
                    l_shoulder, r_shoulder = points[5], points[6]
                    neck = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
                    
                    l_hip, r_hip = points[11], points[12]
                    pelvis = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
                    
                    angle = calculate_spine_angle(neck, pelvis)
        else:
            # Mock behavior: simulate a leaning posture cycle smoothly to test the logic
            # Lean forward past threshold after ~5 seconds, trigger beep, return
            elapsed = time.time()
            angle = 10.0 + max(0, np.sin(elapsed * 0.5) * 15)
            
        if angle > THRESHOLD_ANGLE:
            if bad_posture_start_time is None:
                bad_posture_start_time = time.time()
            elif time.time() - bad_posture_start_time >= ALERT_DELAY:
                winsound.Beep(800, 300)
                bad_posture_start_time = time.time() - (ALERT_DELAY - 1.0)
        else:
            bad_posture_start_time = None

        color = (0, 255, 0) if angle <= THRESHOLD_ANGLE else (0, 0, 255)
        cv2.putText(frame, f"Spine Lean Angle: {angle:.1f} deg", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        if is_mock:
            cv2.putText(frame, "MOCK MODE (No Isaac Model Found)", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        cv2.imshow('PosturePartner Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
