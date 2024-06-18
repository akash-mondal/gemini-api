import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import google.generativeai as genai
import google.ai.generativelanguage as glm
import time
import threading
import uvicorn

# Configure your API key
API_KEY = 'AIzaSyAv7RXj23iVkQ6ZMjbTLLu5v1-_J1v09vY'
genai.configure(api_key=API_KEY)

app = FastAPI()

class RequestData(BaseModel):
    prompt: str
    image_base64: str

@app.post("/analyze")
async def analyze_project(data: RequestData, request: Request):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data.image_base64)
        image = Image.open(BytesIO(image_data))
        bytes_data = BytesIO()
        image.save(bytes_data, format='JPEG')
        bytes_data = bytes_data.getvalue()

        # Send request to Gemini API with the provided prompt and image
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text=data.prompt),
                    glm.Part(
                        inline_data=glm.Blob(
                            mime_type='image/jpeg',
                            data=bytes_data
                        )
                    ),
                ],
            ),
        )
        response.resolve()

        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def periodic_task():
    while True:
        # This is a dummy task that runs every 5 minutes
        print("Fetching some dummy values to keep the server alive...")
        time.sleep(300)  # Sleep for 5 minutes (300 seconds)

@app.on_event("startup")
async def startup_event():
    # Start the periodic task in a separate thread
    threading.Thread(target=periodic_task, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", limit_concurrency=10, timeout_keep_alive=300)

