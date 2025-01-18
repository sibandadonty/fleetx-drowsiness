from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2

app = FastAPI()

# Load your pre-trained model and define the categories
model = load_model("drowsiness_open_close_eye.h5")  # Replace with your model's path
categories = ['Closed_Eyes', 'Open_Eyes']  # Replace with your actual categories


@app.post("/predict-eyes/")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and get the predicted label.
    
    Parameters:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: The predicted label for the uploaded image.
    """
    if not file.filename.endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a JPG or PNG image.")

    try:
        # Read the image file
        contents = await file.read()
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(contents)

        # Preprocess the image
        img = load_img(image_path, color_mode="grayscale", target_size=(64, 64))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_label = categories[np.argmax(prediction[0])]

        # Clean up the temporary file
        os.remove(image_path)

        return JSONResponse(content={"predicted_label": predicted_label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/capture-and-predict-eyes/")
def capture_and_predict():
    """
    Captures an image from the webcam, preprocesses it, and predicts its label.

    Returns:
        JSONResponse: The predicted label for the captured image.
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
        gray_image = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_image = cv2.resize(gray_image, (64, 64))  # Resize to match model input
        normalized_image = resized_image / 255.0  # Normalize pixel values
        img_array = np.expand_dims(normalized_image, axis=(0, -1))  # Add batch and channel dimensions

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_label = categories[np.argmax(prediction[0])]

        return JSONResponse(content={"predicted_label": predicted_label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
