from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import requests
import os
from openai import OpenAI
import json

# Initialize FastAPI app
app = FastAPI(
    title="DMV Assistant",
    description="An AI assistant for navigating DMV processes and document requirements in the USA.",
    version="1.0.0"
)

# Define models for requests and responses
class QuestionRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    document_type: str

class DocumentResponse(BaseModel):
    document_name: str
    url: str

# Mocked database of documents
DOCUMENTS_DB = {
    "driver_license_application": {
        "document_name": "Driver's License Application Form",
        "url": "https://dmv.example.com/forms/driver_license_application.pdf"
    },
    "id_card_application": {
        "document_name": "State ID Application Form",
        "url": "https://dmv.example.com/forms/id_card_application.pdf"
    },
    "vehicle_registration": {
        "document_name": "Vehicle Registration Form",
        "url": "https://dmv.example.com/forms/vehicle_registration.pdf"
    },
}

# OpenAI Configuration
MODEL_NAME = "grok-beta"
XAI_API_KEY = os.getenv("XAI_API_KEY")  # Ensure your API key is set in your environment
client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# Define external tools
functions = [
    {
        "name": "get_document",
        "description": "Fetch a document related to DMV processes.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "Type of document (e.g., 'driver_license_application', 'id_card_application')",
                },
            },
            "required": ["document_type"],
        },
    }
]

# Function implementations
def get_document(document_type: str):
    document = DOCUMENTS_DB.get(document_type)
    if not document:
        raise ValueError("Document type not found")
    return document

@app.get("/")
def read_root():
    return {"message": "Welcome to the DMV Assistant API"}

@app.post("/ask", response_model=List[str])
def ask_question(request: QuestionRequest):
    """
    Handle a user's question about DMV processes.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant for DMV-related processes and documents."},
        {"role": "user", "content": request.question}
    ]

    # Call the model with tools enabled
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=[{"type": "function", "function": f} for f in functions],
        tool_choice="auto"
    )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call['function']['arguments'])

        if tool_call['function']['name'] == "get_document":
            try:
                document = get_document(arguments["document_type"])
                return [f"Here is the document you requested: {document['document_name']} - {document['url']}"]
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

    return [response.choices[0].message.content]

@app.post("/get-document", response_model=DocumentResponse)
def get_document_endpoint(request: DocumentRequest):
    """
    Retrieve a specific document related to DMV processes.
    """
    document = DOCUMENTS_DB.get(request.document_type)
    if not document:
        raise HTTPException(status_code=404, detail="Document type not found")
    return DocumentResponse(**document)


