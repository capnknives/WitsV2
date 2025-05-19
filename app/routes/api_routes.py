# app/routes/api_routes.py
from fastapi import APIRouter, HTTPException, Depends, Body, status, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import logging
import os
import json
from pathlib import Path # Added import for Path
from core.config import AppConfig, load_app_config # Corrected import for AppConfig and load_app_config
from core.schemas import (
    BookProjectCreate, BookProjectUpdate, BookProject,
)
from agents.book_writing_schemas import BookWritingState

router = APIRouter()

# Load application configuration
app_config: AppConfig = load_app_config()

BOOK_PROJECTS_DIR = Path(app_config.output_directory) / "book_projects"
BOOK_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

def get_project_path(project_name: str) -> Path:
    if not project_name or ".." in project_name or "/" in project_name or "\\\\" in project_name:
        raise HTTPException(status_code=400, detail="Invalid project name.")
    return BOOK_PROJECTS_DIR / f"{project_name.replace(' ', '_').lower()}.json"

@router.get("/api/book_projects", response_model=List[str])
async def list_book_projects():
    projects = []
    if not BOOK_PROJECTS_DIR.exists():
        return projects
    for f_name in os.listdir(BOOK_PROJECTS_DIR):
        if f_name.endswith(".json"):
            projects.append(Path(f_name).stem.replace('_', ' ').title())
    return projects

@router.post("/api/book_projects", response_model=BookWritingState)
async def create_book_project(payload: Dict[str, str] = Body(..., مثال="{\"project_name\": \"My New Book\"}")):
    project_name = payload.get("project_name")
    if not project_name or project_name.isspace():
        raise HTTPException(status_code=400, detail="Project name cannot be empty.")
    
    project_file = get_project_path(project_name)
    if project_file.exists():
        raise HTTPException(status_code=400, detail=f"Project '{project_name}' already exists.")
    
    new_book_state = BookWritingState(project_name=project_name)
    
    try:
        with open(project_file, "w") as f:
            json.dump(new_book_state.model_dump(), f, indent=2)
        return new_book_state
    except IOError as e:
        # Consider logging the error e
        raise HTTPException(status_code=500, detail=f"Failed to create project file.")
    except Exception as e:
        # Consider logging the error e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during project creation.")

@router.get("/api/book_projects/{project_name}", response_model=BookWritingState)
async def get_book_project(project_name: str):
    project_file = get_project_path(project_name)
    if not project_file.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    try:
        with open(project_file, "r") as f:
            data = json.load(f)
            return BookWritingState(**data)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read project file.")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse project file.")
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching project: {str(e)}")

@router.put("/api/book_projects/{project_name}", response_model=BookWritingState)
async def update_book_project(project_name: str, book_state: BookWritingState = Body(...)):
    project_file = get_project_path(project_name)
    if project_name != book_state.project_name:
        raise HTTPException(status_code=400, detail="Project name in URL and payload must match.")

    if not project_file.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found. Use POST to create.")

    try:
        with open(project_file, "w") as f:
            json.dump(book_state.model_dump(), f, indent=2)
        return book_state
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write project file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during project update: {str(e)}")

# The existing chat_stream, agent management, etc. routes should be in app/main.py or another route file.
# This file will focus on book project specific APIs.

# Example of how this router might be included in main.py:
# from app.routes import api_routes as book_api_routes
# app.include_router(book_api_routes.router)
