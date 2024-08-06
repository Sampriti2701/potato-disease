from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1.keras")
CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"
def read_file_as_image(data) -> np.ndarray:
    Image = np.array(Image.open(BytesIO(data)))
    return image
    
@app.post("/predict")
async def predict(
   file: UploadFile = File(...)
 ):
     image = read_file_as_image(await file.read())
     img_batch = np.expand_dims(image,0)
     prediction =  MODEL.predict(img_batch)
     pass

if __name__ =="__main__":
 uvicorn.run(app, host='localhost', port=8000)