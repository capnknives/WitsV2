import os
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

file_router = APIRouter(tags=["files"])

# Ensure the upload directory exists
def get_upload_dir():
    upload_dir = os.path.join("data", "user_files")
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

@file_router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload a file to be processed by WITS-NEXUS.
    The file will be saved in the user_files directory.
    """
    try:
        # Generate a unique filename
        upload_dir = get_upload_dir()
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        
        # Write the file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info(f"File uploaded: {filename} for session {session_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"File uploaded successfully: {filename}",
                "file_path": file_path,
                "original_filename": file.filename,
                "session_id": session_id
            }
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@file_router.get("/api/files")
async def list_session_files(session_id: str):
    """
    List all files uploaded for a specific session.
    """
    try:
        upload_dir = get_upload_dir()
        all_files = os.listdir(upload_dir)
        
        # Parse the filenames to get session IDs if they're encoded in the filename
        session_files = []
        
        for filename in all_files:
            # Here you would implement your logic for tracking files by session
            # This is a simple example - you might have a database or other way to track this
            file_path = os.path.join(upload_dir, filename)
            stat = os.stat(file_path)
            
            session_files.append({
                "filename": filename,
                "file_path": file_path,
                "size": stat.st_size,
                "created_at": stat.st_ctime
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "session_id": session_id,
                "files": session_files
            }
        )
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
