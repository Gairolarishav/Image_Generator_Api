from fastapi import FastAPI,HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
import requests

app = FastAPI()
openai.api_key = os.environ.get('api_key')

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    text: str
    count : int
    size: str

@app.post('/image_generator')
async def Image_Generator(input_data: ImageInput):
    if input_data.text:
        images = []
        response = openai.images.generate(
            model="dall-e-2",
            prompt=f"Generate a real image of : {input_data.text}",
            n=input_data.count,
            size=input_data.size,
            quality="standard",
            response_format='b64_json'
        )
        # print(response)
        try:
            for i, image_data in enumerate(response.data):
                image_url = image_data.b64_json
                images.append(image_url)
            return images
        except openai.AuthenticationError as e:
            raise HTTPException(status_code=401, detail="Incorrect API key")
        except Exception as e:
            raise HTTPException(status_code=500, detail="An error occurred while generating images")
    else:
        return "Please Enter something"
    


class ContentInput(BaseModel):
    text: str

@app.post('/content_generator')
async def Content_Generator(input_data: ContentInput):
    URL = "https://api.openai.com/v1/chat/completions"

    messages = [
        {'role': 'system', 'content': 'You are a kind helpful assistant'}
    ]

    while True:
      user_input = input_data.text
      if user_input:
        # Append the user input to the messages list
        messages.append({'role': 'user', 'content': user_input})

        payload = {
          "model": "gpt-3.5-turbo",
          "messages": messages,
          "temperature": 1.0,
          'top_p': 1.0,
          'n': 1,
          'stream': False,
          "presence_penalty": 0,
          "frequency_penalty": 0,
        }
        headers = {
        'Content-Type': "application/json",
        'Authorization': f"Bearer {openai.api_key}"
        }
        response = requests.post(URL, headers=headers, json=payload)
      else:
         return "Please Enter something"

      if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content']
        reply = reply.replace('\n','<br>')
        messages.append({"role": "assistant", "content": reply})
        return reply
      else:
         return response.json()['error']['message']



if __name__ == "__main__":
    uvicorn.run('main:app',host='localhost',port=8000)