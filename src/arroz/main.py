from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

model_name = "KaliumPotas/potas_recommend"
classifier = pipeline("text-classification", model=model_name)

class Request(BaseModel):
    audience: str
    category: str
    area: str
    sub_area: str

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.post("/ai")
async def ai(request: Request):
    try:
        text = (
            f"This ad is focused on: {request.audience}. "
            f"Category: {request.category}. "
            f"Area: {request.area}. "
            f"Sub-area: {request.sub_area}."
        )
        result = classifier(text)
        
        return {"message": result}
    except BaseException as error:
        print(error)
        
        