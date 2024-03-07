from fastapi import FastAPI,HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
import requests
import base64
from BingImageCreator import ImageGen

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
    # count : int
    # size: str

@app.post('/image_generator')
def Image_Generator(input_data: ImageInput):
    try :    
       if input_data.text:
          images= [ ]
          image_gen = ImageGen(auth_cookie='1Vd9x_vBWsb_L3wJtLXMeKVWOKLCiIG-lPzivgKrqP24RpEfPy4JOoFRFWWTBEHkVA0JOxfWe6GiOeXnYhDRggtZokuRhF7BokBXzt6YMwzeTZS72VINEBgAbOCdLpDaLVIMNxzKVl_D3rLNzeeYYxQBp9itDoK93X8q1xMhxtLr0EwMPlXw2khy-i4FRCK1ZH-ZKohtC2vgYGkGIjKDAIw',auth_cookie_SRCHHPGUSR='1Vd9x_vBWsb_L3wJtLXMeKVWOKLCiIG-lPzivgKrqP24RpEfPy4JOoFRFWWTBEHkVA0JOxfWe6GiOeXnYhDRggtZokuRhF7BokBXzt6YMwzeTZS72VINEBgAbOCdLpDaLVIMNxzKVl_D3rLNzeeYYxQBp9itDoK93X8q1xMhxtLr0EwMPlXw2khy-i4FRCK1ZH-ZKohtC2vgYGkGIjKDAIw')

          # Provide a prompt (your text description)
          prompt = f'create and image of {input_data.text}'

          # Generate an image based on the prompt
          image_url = image_gen.get_images(prompt)
          for i in image_url:
            response = requests.get(i)
            if response.status_code == 200:
                image_bytes = response.content
                base64_bytes = base64.b64encode(image_bytes)
                base64_string = base64_bytes.decode('utf-8')
                print(base64_string)
                if base64_string.startswith('/'):
                   images.append(base64_string)
          return images
       else:
         return "Please Enter something"
    except Exception as e:
        # print(e.http_status)
        return (e)

# @app.post('/save_image')
# def save_image():
# save_images(links: list, output_dir: str, file_name: str = None) -> None



# def Image_Generator(input_data: ImageInput):
#     try :    
#        if input_data.text:
#            images = []
#            response = openai.images.generate(
#            model="dall-e-2",
#            prompt= input_data.text,
#            size= input_data.size,
#            quality="standard",
#            response_format = "b64_json",
#            n=input_data.count,
#            )
#            print(response)

#            for i, image_data in enumerate(response.data):
#                    image_url = image_data.b64_json
#                    images.append(image_url)
#                    print(images)
#            return images
#        else:
#          return "Please Enter something"
#     except openai.OpenAIError as e:
#         # print(e.http_status)
#         return (e.code)
    


class ContentInput(BaseModel):
    text: str

@app.post('/content_generator')
def Content_Generator(input_data: ContentInput):
    URL = "https://api.openai.com/v1/chat/completions"

    messages = [
        {'role': 'system', 'content': 'You are a kind helpful assistant'}
    ]
    print(input_data)

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
