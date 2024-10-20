import cv2
import numpy as np
from keras.models import load_model  # Updated import

# Dictionary to map predicted labels to emotion names
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                 3: "Happy", 4: "Neutral", 5: "Sad",
                 6: "Surprised"}

# Load the entire model directly
emotion_model = load_model('emotion_model.h5')  # Adjust the path accordingly

print("Loaded model from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frame is captured

    frame = cv2.resize(frame, (1280, 720))  # Resize the frame for consistent display

    # Load Haar Cascade for face detection
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Preprocess each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)  # Draw rectangle around face
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # Resize and expand dimensions for model input
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))  # Get the index of the highest probability
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion prediction
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
