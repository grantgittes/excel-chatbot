import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from langserve import add_routes
import os
from fastapi.staticfiles import StaticFiles
import aiofiles
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import uvicorn
import asyncio
from app.custom_agent import AgentInputs, execute_agent

class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Clone the request body for logging
        body = await request.body()
        # Create a new request clone with the original body, since reading the body consumes the stream
        request = Request(request.scope, request.receive)
        request._body = body  # Reassign the cloned body for logging

        # Log request details
        print(f"Request method: {request.method}")
        print(f"Request URL: {request.url}")
        # Optional: decode body to string, be cautious for binary data
        # body_str = body.decode('utf-8') if body else "No body"
        # print(f"Request body: {body_str}")

        # Proceed with request processing
        response = await call_next(request)
        return response

app = FastAPI()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

from fastapi.middleware.cors import CORSMiddleware

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
from fastapi.staticfiles import StaticFiles

app.add_middleware(LogRequestsMiddleware)

# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
# from app.csv_agent import agent_executor as csv_agent_chain

# add_routes(app, csv_agent_chain, path="/csv-agent")

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    print("Trying to upload!")
    if not file.filename.endswith(('.csv','.xlsx','.xls')):
        raise HTTPException(status_code=400, detail="File extension not allowed. Please upload a .csv file.")
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    async with aiofiles.open(file_location, 'wb') as out_file:
        content = await file.read()  # read file content
        await out_file.write(content)  # write to file
    return {"info": f"File '{file.filename}' uploaded successfully.", "filename": file.filename}

@app.get("/download-csv/{filename}")
async def download_csv(filename: str):
    file_path = f"{UPLOAD_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='text/csv')
    raise HTTPException(status_code=404, detail="File not found.")



@app.post("/agent-query/")
async def agent_query(query: AgentInputs):
    print("Asking agent!")
    try:
        result = await execute_agent(query)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
        


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
