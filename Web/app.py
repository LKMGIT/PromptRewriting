from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_handler import rag_refine_prompt

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.post("/", response_class=HTMLResponse)
def handle_prompt(request: Request, user_input: str = Form(...)):
    result = rag_refine_prompt(user_input)
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "user_input": user_input})
