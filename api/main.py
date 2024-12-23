from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="DMV Document Validator and Assistant",
    description="A combined API for document validation and DMV assistance.",
    version="1.0.0",
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the DMV Document Validator and Assistant API"}
