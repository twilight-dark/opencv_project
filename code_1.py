import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' is usually the default camera

# Define the HSV color range for detecting a red laser pointer
lower_red = np.array([160, 100, 100])
upper_red = np.array([180, 255, 255])

def calculate_distance(area):
    """Estimate the distance based on the area of the detected laser spot."""
    # Placeholder for distance estimation based on area
    # This function should be calibrated with real distance measurements
    return max(2 - 0.1 * area, 0)  # Example equation; adjust as necessary

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to filter out only the red laser light
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Filter out small contours that are likely noise
            # Get bounding box coordinates for the detected laser spot
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimate the distance based on the area of the laser spot
            distance = calculate_distance(area)
            if distance <= 2:  # Check if the estimated distance is within the range
                cv2.putText(frame, f"Distance: {distance:.2f} m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the processed frame with detection results
    cv2.imshow("Laser Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
