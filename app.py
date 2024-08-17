## Fast API Module with Uvicorn to create API of model
# Model is stored in model.pkl file

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pickle
from PIL import Image
import io

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

# Function to preprocess the image before prediction
def preprocess_image(image: Image.Image):
    # Resize the image to the required size (e.g., 224x224 if the model requires this)
    image = image.resize((224, 224))
    # Convert the image to a numpy array and normalize it (0-1)
    image_array = np.array(image) / 255.0
    # Add batch dimension (if required by the model)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/")
def home():
    return {"message": "Welcome to the prediction API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    output_probability = prediction[0]  # Assuming the model outputs probabilities

    # Create response
    return JSONResponse(content={"prediction_probability": float(output_probability)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
