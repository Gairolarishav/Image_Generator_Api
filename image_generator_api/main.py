from fastapi import FastAPI,HTTPException,File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
import requests
import base64
import json
import requests
from typing import List, Dict, Optional
from clarifai.client.model import Model
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

app = FastAPI()
openai.api_key = os.environ.get('api_key')

# CORS (Cross-Origin Resource Sharing) configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class DalleInput(BaseModel):
    text: str
    previous_image: Optional[str] = None

class ConversationState:
    def __init__(self):
        self.current_image = None
        self.conversation_history = []

conversation_state = ConversationState()

@app.post('/dalle_generator')
def dalle_generator(input_data: DalleInput):
    if not input_data.text:
        raise HTTPException(status_code=400, detail="Please enter a prompt")
    
    try:
        # Setting the inference parameters
        inference_params = dict(quality="standard", size="1024x1024")
        
        # Construct the prompt based on conversation history
        full_prompt = construct_prompt(input_data.text)
        
        # Using the model Dall-e-3 to generate image
        model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(
            full_prompt.encode(), 
            input_type="text",
            inference_params=inference_params
        )
        
        # Storing the output
        output = model_prediction.outputs[0].data.image.base64
        image_data = base64.b64encode(output).decode('utf-8')
        
        # Update conversation state
        conversation_state.current_image = image_data
        conversation_state.conversation_history.append(input_data.text)
        
        return image_data
    
    except Exception as exc:
        # Error handling (similar to your original code)
        error = str(exc)
        # ... (rest of your error handling code)
        return stop[0:-1]

def construct_prompt(new_input: str) -> str:
    if not conversation_state.conversation_history:
        return new_input
    else:
        previous_prompt = conversation_state.conversation_history[-1]
        return f"Based on the previous image of '{previous_prompt}', {new_input}"

@app.get('/reset_conversation')
def reset_conversation():
    conversation_state.current_image = None
    conversation_state.conversation_history = []
    return {"message": "Conversation reset successfully"}

# class DalleInput(BaseModel):
#     text: str

# @app.post('/dalle_generator')
# def Dalle_Generator(input_data: DalleInput):
#     if not input_data.text:
#         return "Please Enter something"
    
#     try:
#        images = []
#        # Setting the inference parameters
#        inference_params = dict(quality="standard" , size= "1024x1024")

#        # Using the model Dall-e-3 to generate image
#        # Passing the prompt and inference parameters
#        for i in range(2):
#            model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(input_data.text.encode(), input_type="text",inference_params = inference_params) 

#         # Storing the output
#            output = model_prediction.outputs[0].data.image.base64
#            image_data = base64.b64encode(output).decode('utf-8')
#            images.append(image_data)
#        print(len(images))
#        return images
#     except Exception as exc:
#          # Extracting the message from the exception
#         error = str(exc)
#         details = error.find('details')
#         details = error[details:]
#         badreq = details.find("BadRequestError-Error code:")
#         badreq = details[badreq:]
#         badreq = badreq.split(':')
#         badreq = badreq[4]
#         stop =  badreq.split("'")
#         stop = stop[1] 
#         # badreq = badreq[2:stop]
#         return stop[0:-1]  

class StableInput(BaseModel):
    text: str
    style: str

@app.post('/stable_generator')
def Stable_Generator(input_data: StableInput):
   engine_id = "stable-diffusion-v1-6"
   api_host = 'https://api.stability.ai'
   api_key = os.getenv("STABILITY_API_KEY")

   text = input_data.text
   style = input_data.style
   images = []

   if not input_data.text:
      return "Please Enter something"
   
   elif (api_key is None):   
      return("Missing Stability API key.")
   
   else:
      response = requests.post(   
          f"{api_host}/v1/generation/{engine_id}/text-to-image",
          headers={
          "Content-Type": "application/json",
          "Accept": "application/json",
          "Authorization": f"Bearer {api_key}"
          },
          json={
          "text_prompts": [
              {
                  "text": text
              }
          ],
          "cfg_scale": 7,
          "height": 512,
          "width": 512,
          "samples": 2,
          "steps": 30,
          "style_preset" : style
          },
      )

      if response.status_code != 200:
         error =  response.json()
         error = error['message']
         return error
         
         
        #  return result

      data = response.json()
      for i, image in enumerate(data["artifacts"]):
         image = image["base64"]
         image = images.append(image)
      return images




class ContentInput(BaseModel):
    text: str

# @app.post('/content_generator')
# def Content_Generator(input_data: ContentInput):
#     URL = "https://api.openai.com/v1/chat/completions"

#     messages = [
#         {'role': 'system', 'content': 'You are a kind helpful assistant'}
#     ]
#     print(input_data)

#     while True:
#       user_input = input_data.text
#       if user_input:
#         # Append the user input to the messages list
#         messages.append({'role': 'user', 'content': user_input})

#         payload = {
#           "model": "gpt-3.5-turbo",
#           "messages": messages,
#           "temperature": 1.0,
#           'top_p': 1.0,
#           'n': 1,
#           'stream': False,
#           "presence_penalty": 0,
#           "frequency_penalty": 0,
#         }
#         headers = {
#         'Content-Type': "application/json",
#         'Authorization': f"Bearer {openai.api_key}"
#         }
#         response = requests.post(URL, headers=headers, json=payload)
#       else:
#          return "Please Enter something"

#       if response.status_code == 200:
#         reply = response.json()['choices'][0]['message']['content']
#         reply = reply.replace('\n','<br>')
#         messages.append({"role": "assistant", "content": reply})
#         return reply
#       else:
#          return response.json()['error']['message']

# Clarifai API setup
PAT = "7d91ea7a8cf84e54bd72e3579b6b210c"
USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'gpt-4o'
MODEL_VERSION_ID = '1cd39c6a109f4f0e94f1ac3fe233c207'

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (('authorization', 'Key ' + PAT),)

userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

# Initialize an empty list to store the conversation history
conversation_history = [{'role': 'system', 'content': 'You are a kind helpful assistant'}]

@app.post('/content_generator')
def Content_Generator(input_data: ContentInput):
    print("input_data.text :", input_data.text)
    if not input_data.text:
        return {"error": "Please Enter something"}

    try:
        # Append the new user message to the conversation history
        conversation_history.append({'role': 'user', 'content': input_data.text})

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=userDataObject,
                model_id=MODEL_ID,
                version_id=MODEL_VERSION_ID,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            text=resources_pb2.Text(
                                raw=json.dumps(conversation_history)
                            )
                        )
                    )
                ]
            ),
            metadata=metadata
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

        output = post_model_outputs_response.outputs[0]
        reply = output.data.text.raw.replace('\n', '<br>')

        # Append the assistant's reply to the conversation history
        conversation_history.append({'role': 'assistant', 'content': reply})

        return reply
    
    except Exception as exc:
        error = str(exc)
        if "404" in error:
            return "Resource not found. Please check your Model ID and Version ID."
        elif "403" in error:
            return "Access forbidden. Please check your API key permissions."
        else:
            return "An unexpected error occurred: " + error
        
# def read_file_as_image(data):
#     nparr = np.fromstring(data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

@app.post('/extractor')
async def text_extractor(file: UploadFile = File(...)):
    try:
        # Get the file name
        # file_name = file.filename
        # print("File Name:", file_name)def read_file_as_image(data):
#     nparr = np.fromstring(data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

        # img = read_file_as_image(await file.read())
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(img)

        # ret, thresh = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((2, 2), np.uint8)
        # img_erosion = cv2.erode(thresh, kernel, iterations=1)
        # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        # img1 = Image.fromarray(img_dilation)
        # im3 = ImageEnhance.Brightness(img1)
        # im3 = im3.enhance(1)
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiOWY2ODVjZjQtNjcyZS00ZTVkLTk2N2YtZmVkYTIwYjBhNTQ3IiwidHlwZSI6ImFwaV90b2tlbiJ9.CKdmWLOx1AqzqeX_cL6WGCP8C5nRscR9sTf2dgXCbbc"}

        url = "https://api.edenai.run/v2/ocr/ocr"
        data = {
            "providers": "google",
            "language": "en",
            "fallback_providers": ""
        }
        files = {"file": (file.filename, file.file, file.content_type)}

        response = requests.post(url, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        print(result)
        return result["google"]["text"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)

class Entry(BaseModel):
    folder_id: int
    duplicate_id: int

@app.post("api/forms/entries/duplicate/{entryy_id}")
async def duplicate_entry(entryy_id: int, entry: Entry):

    folder_id = entry.folder_id
    duplicate_id = entry.duplicate_id

    if folder_id and duplicate_id: 
        folder = entry.folder_id
        
             
        return Response({'status': 'Entry Duplicated successfully inside a folder', 'new_entry':1}) 
    else:
        pass 
    
    
    return Response({'status': 'Entry Duplicated successfully', 'new_entry':2})

if __name__ == "__main__":
    uvicorn.run('main:app',host='localhost',port=8000)
