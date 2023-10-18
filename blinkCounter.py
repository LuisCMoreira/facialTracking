import cv2
import dlib
import numpy as np

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'.\shape_predictor_68_face_landmarks.dat')

# Function to calculate the EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    eye = [np.array([point.x, point.y]) for point in eye]  # Convert Dlib points to NumPy arrays
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for blink detection
EAR_THRESHOLD = 0.2  # Adjust this threshold as needed
CONSECUTIVE_FRAMES = 3  # Number of consecutive frames for a blink

# Initialize variables for blink detection
left_blink_counter = 0
right_blink_counter = 0
left_consecutive_frames = 0
right_consecutive_frames = 0

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        
        # Calculate EAR for the left eye
        left_eye_ear = eye_aspect_ratio([shape.part(i) for i in range(36, 42)])

        # Calculate EAR for the right eye
        right_eye_ear = eye_aspect_ratio([shape.part(i) for i in range(42, 48)])

        # Check if the EAR for the left eye is below the threshold
        if left_eye_ear < EAR_THRESHOLD:
            left_consecutive_frames += 1
        else:
            if left_consecutive_frames >= CONSECUTIVE_FRAMES:
                left_blink_counter += 1
                print("Left Eye Blink Detected! Total Left Eye Blinks:", left_blink_counter)
            left_consecutive_frames = 0

        # Check if the EAR for the right eye is below the threshold
        if right_eye_ear < EAR_THRESHOLD:
            right_consecutive_frames += 1
        else:
            if right_consecutive_frames >= CONSECUTIVE_FRAMES:
                right_blink_counter += 1
                print("Right Eye Blink Detected! Total Right Eye Blinks:", right_blink_counter)
            right_consecutive_frames = 0

    cv2.putText(frame, f"Left Eye Blink Count: {left_blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Eye Blink Count: {right_blink_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
