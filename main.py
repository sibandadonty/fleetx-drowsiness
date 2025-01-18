from fastapi import FastAPI
from routes import eyes

app = FastAPI()

app.include_router(eyes.router)

