from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
from pydantic import BaseModel
from io import BytesIO

app = FastAPI()
openai.api_key = "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"

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    count : int
    size: str

@app.post('/image_generator')
async def Image_Generator(input_data: TextInput):
    images = []
    response = openai.images.generate(
      model="dall-e-2",
      prompt= f"Generate a real image of : {input_data.text}",
      n = input_data.count,
      size=input_data.size,
      quality="standard",
      response_format = 'b64_json'
    )
    for i, image_data in enumerate(response.data):
      image_url = image_data.b64_json
      print(image_url)
      images.append(image_url)
    return images
