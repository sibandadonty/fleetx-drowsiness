from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

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
