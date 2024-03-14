from fastapi import FastAPI,HTTPException,File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
import requests
import base64
from BingImageCreator import ImageGen
import cv2
import numpy as np
import pytesseract
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

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

@app.post('/image_generator')
def Image_Generator(input_data: ImageInput):
    images = []
    
    if not input_data.text:
        return "Please Enter something"
    
     # Your PAT (Personal Access Token) can be found in the portal under Authentification
    PAT = '7d91ea7a8cf84e54bd72e3579b6b210c'
    # Specify the correct user_id/app_id pairings
    # Since you're making inferences outside your app's scope
    USER_ID = 'openai'
    APP_ID = 'dall-e'
    # Change these to whatever model and text URL you want to use
    MODEL_ID = 'dall-e-3'
    MODEL_VERSION_ID = 'dc9dcb6ee67543cebc0b9a025861b868'
        
    # Initialize Clarifai API
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    # Make request to DALL-E model
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=input_data.text
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    
    # Handle response
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
    # Append generated image base64 to images list
    image_base64 = post_model_outputs_response.outputs[0].data.image.base64
    
    return {"image_base64": image_base64}

        # except Exception as e:
        #    # Check if the exception message matches the one from BingImageCreator.py
        #    if str(e) == "Your prompt has been blocked by Bing. Try to change any bad words and try again.":
        #        raise HTTPException(status_code=403, detail="blocked the prompt. Please revise it and try again.")
        #    else:
        #        raise HTTPException(status_code=500, detail="Internal Server Error")

# @app.post('/save_image')
# def save_image():
# save_images(links: list, output_dir: str, file_name: str = None) -> None

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
