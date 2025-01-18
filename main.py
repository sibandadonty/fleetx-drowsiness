from fastapi import FastAPI
from routes import eyes, faces

app = FastAPI()

app.include_router(eyes.router)
app.include_router(faces.router)

