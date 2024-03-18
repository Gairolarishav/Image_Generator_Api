from fastapi import FastAPI,HTTPException,File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
import requests
import base64
import cv2
import numpy as np
import pytesseract
from clarifai.client.model import Model

app = FastAPI()
openai.api_key = os.environ.get('api_key')

os.environ.get('CLARIFAI_PAT')

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

@app.post('/image_generator')
def Image_Generator(input_data: ImageInput):

    if not input_data.text:
        return "Please Enter something"
    
    try:
       images = []
       # Setting the inference parameters
       inference_params = dict(quality="standard" , size= "1024x1024")
       # Using the model Dall-e-3 to generate image
       # Passing the prompt and inference parameters
       for i in range(2):
           model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(input_data.text.encode(), input_type="text",inference_params = inference_params) 
        # Storing the output
           output = model_prediction.outputs[0].data.image.base64
           image_data = base64.b64encode(output).decode('utf-8')
           images.append(image_data)
       print(len(images))
       return images
    except Exception as exc:
         # Extracting the message from the exception
        error = str(exc)
        details = error.find('details')
        details = error[details:]
        badreq = details.find("BadRequestError-Error code:")
        badreq = details[badreq:]
        badreq = badreq.split(':')
        badreq = badreq[4]
        stop =  badreq.split("'")
        stop = stop[1] 
        # badreq = badreq[2:stop]
        return stop[0:-1]       



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

def read_file_as_image(data):
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post('/extractor')
async def Content_Generator(file: UploadFile = File(...)):
     img = read_file_as_image(await file.read())
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

     # ret, thresh = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
     # from PIL import Image, ImageEnhance
     # import numpy as np
     # kernel = np.ones((2, 2), np.uint8)
     # img_erosion = cv2.erode(thresh, kernel, iterations=1)
     # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
     # # Creating object of Brightness class
     # img1 = Image.fromarray(img_dilation)
     # im3 = ImageEnhance.Brightness(img1)
     # im3 = im3.enhance(1)
     # plt.imshow(img,cmap='gray')
     # plt.show()
     # # custom_config = r'-l eng+hin --psm 6'
     text = pytesseract.image_to_string(img,lang='hin+eng')
     print(text)
     return text

if __name__ == "__main__":
    uvicorn.run('main:app',host='localhost',port=8000)
