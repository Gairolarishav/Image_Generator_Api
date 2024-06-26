from fastapi import FastAPI,HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
import requests
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
    count : int
    size: str

@app.post('/image_generator')
def Image_Generator(input_data: ImageInput):
    if input_data.text:
        images = []
        response = openai.images.generate(
            model="dall-e-3",
            prompt=input_data.text,
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
    


# class ContentInput(BaseModel):
#     text: str

# @app.post('/content_generator')
# def Content_Generator(input_data: ContentInput):
#     URL = "https://api.openai.com/v1/chat/completions"

#     messages = [
#         {'role': 'system', 'content': 'You are a kind helpful assistant'}
#     ]

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

class ContentInput(BaseModel):
    text: str

PAT = '5e9f346a37a7460485dc8c549fc7fd4d'
USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'GPT-3_5-turbo'
MODEL_VERSION_ID = '4c0aec1853c24b4c83df8ba250f3b984'

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (('authorization', 'Key ' + PAT),)

userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

@app.post('/content_generator')
def Content_Generator(input_data: ContentInput):
    if not input_data.text:
        return {"error": "Please Enter something"}
    
    try:
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

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

        output = post_model_outputs_response.outputs[0]
        reply = output.data.text.raw.replace('\n', '<br>')

        return {"reply": reply}
    
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run('main:app',host='localhost',port=8000)
