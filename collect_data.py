import cv2
import os
import sys

# Get user inputs from command line
try:
    label_name = sys.argv[1]  # Rock, Paper, or Scissors
    num_samples = int(sys.argv[2])  # Total images per class
except:
    print("❌ Missing arguments.")
    print(desc)
    exit(-1)

# Create dataset folder structure
IMG_SAVE_PATH = "dataset"
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

os.makedirs(IMG_SAVE_PATH, exist_ok=True)
os.makedirs(IMG_CLASS_PATH, exist_ok=True)

print(f"📂 Images will be saved in: {IMG_CLASS_PATH}")

# Start webcam
cap = cv2.VideoCapture(0)

start = False  # Controls when images are captured
count = 0  # Counter for saved images

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Draw a box (ROI) to capture hand gestures
    cv2.rectangle(frame, (100, 100), (600, 600), (255, 255, 255), 2)

    if start:
        # Extract Region of Interest (ROI)
        roi = frame[100:600, 100:600]  
        
        # Resize to match model input size
        roi = cv2.resize(roi, (100, 100))

        # Save image
        save_path = os.path.join(IMG_CLASS_PATH, f"{count + 1}.jpg")
        cv2.imwrite(save_path, roi)
        count += 1

    # Display current count
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Collecting: {count}/{num_samples}", 
                (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Collecting images", frame)

    # Keypress Controls
    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start  # Toggle start/pause

    if k == ord('q') or count >= num_samples:
        break  # Stop collecting when limit is reached

print(f"\n✅ {count} images saved in: {IMG_CLASS_PATH}")

cap.release()
cv2.destroyAllWindows()
