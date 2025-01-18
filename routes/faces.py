from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

router = APIRouter(
    tags=["Faces"]
)

# Load the pre-trained model
model = load_model("drowsiness_faces_fleetx.h5")  
categories = ["Drowsy", "Non Drowsy"]

@router.get("/capture-and-predict-class/")
def capture_and_predict_class():
    """
    Captures an image from the webcam, preprocesses it, and predicts its class using the model.

    Returns:
        JSONResponse: The predicted class label for the captured image.
    """
    try:
        # Open the camera
        cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not access the camera.")

        print("Press 'Space' to capture an image and 'q' to quit.")
        captured_frame = None

        while True:
            # Read the video stream frame-by-frame
            ret, frame = cap.read()

            if not ret:
                raise HTTPException(status_code=500, detail="Failed to capture frame from camera.")

            # Display the live video feed
            cv2.imshow("Camera", frame)

            # Wait for a key press
            key = cv2.waitKey(1)

            if key == ord(" "):  # Space key to capture the frame
                captured_frame = frame
                break
            elif key == ord("q"):  # Quit if 'q' is pressed
                cap.release()
                cv2.destroyAllWindows()
                return JSONResponse(content={"message": "Camera feed closed without capturing."})

        # Release the camera and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is None:
            raise HTTPException(status_code=400, detail="No image captured.")

        # Preprocess the captured image
        resized_frame = cv2.resize(captured_frame, (224, 224))  # Resize to match ResNet50V2 input size
        img_array = img_to_array(resized_frame)  # Convert to NumPy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess for ResNet50V2

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)  # Get the index of the highest probability
        predicted_label = categories[predicted_index]  # Map index to class label

        return JSONResponse(content={"predicted_label": predicted_label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")