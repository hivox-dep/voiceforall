# voiceforall
import cv2
import numpy as np
import tensorflow as tf
tf.keras.layers.Convolution2D
from tensorflow.keras.layers import DepthwiseConv2D
import pyttsx3

# Define a custom DepthwiseConv2D layer (if needed, without 'groups')
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def _init_(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super()._init_(**kwargs)

# Load the trained model
try:
    # Use custom_objects to override the layer with the unsupported argument
    model = tf.keras.models.load_model(
        'C:/Users/ADMIN/Desktop/HandGesture.h5',
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Define the gesture classes
gesture_classes = [
    "Stop",
    "Ok",
    "Sorry",
    "Silent",
    "Victory",
    "Losers",
    "Hello",
    "Good Luck",
    "Good Bye"
]

# Define constants
input_shape = (224, 224, 3)

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (input_shape[0], input_shape[1]))  # Resize image to match model input
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to detect hand gestures using color segmentation (example approach)
def detect_hand_region(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for detecting skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply mask to original frame
    hand_segment = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convert mask to grayscale and find contours
    gray = cv2.cvtColor(hand_segment, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours of the hand
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assuming the largest contour is the hand
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop hand region from frame
        hand_roi = frame[y:y+h, x:x+w]
        return hand_roi
    else:
        return None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Detect hand region
    hand_roi = detect_hand_region(frame)
    
    if hand_roi is not None:
        # Preprocess the hand region image
        processed_image = preprocess_image(hand_roi)
        
        # Make predictions
        try:
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            gesture = gesture_classes[predicted_class]
        except Exception as e:
            print(f"Error making predictions: {e}")
            continue
        
        # Display the resulting frame with the predicted gesture
        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Speak the gesture
        engine.say(gesture)
        engine.runAndWait()
    
    # Show the frame
    cv2.imshow('Gesture Recognition', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
