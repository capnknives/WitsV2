# tools/project_file_tools.py
import os
from typing import ClassVar, Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from .base_tool import BaseTool

class ProjectFileReaderArgs(BaseModel):
    """Arguments for reading a project file."""
    relative_project_path: str = Field(
        ..., 
        description="Path to the file relative to the project root."
    )

class ProjectFileReaderResponse(BaseModel):
    """Response from reading a project file."""
    content: str = Field("", description="Content of the file.")
    error: Optional[str] = Field(None, description="Error message if file reading failed.")
    file_path: str = Field(..., description="Path to the file that was read.")
    exists: bool = Field(False, description="Whether the file exists.")

class ProjectFileReaderTool(BaseTool):
    """Tool for securely reading project source files."""
    
    name: ClassVar[str] = "project_file_reader"
    description: ClassVar[str] = "Read the contents of a project source file. Path should be relative to project root."
    args_schema: ClassVar[Type[BaseModel]] = ProjectFileReaderArgs
    
    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        # Determine project root - this should be set in config or determined dynamically
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    async def execute(self, args: ProjectFileReaderArgs) -> ProjectFileReaderResponse:
        """
        Read the contents of a project file.
        
        Args:
            args: ProjectFileReaderArgs containing the relative file path
            
        Returns:
            ProjectFileReaderResponse: The file contents or error
        """
        try:
            # Normalize path and ensure it stays within project
            normalized_path = os.path.normpath(args.relative_project_path)
            if normalized_path.startswith('..') or normalized_path.startswith('/'):
                return ProjectFileReaderResponse(
                    content="",
                    error="Invalid path: Must be relative to project root",
                    file_path=args.relative_project_path,
                    exists=False
                )
            
            # Construct full path
            full_path = os.path.join(self.project_root, normalized_path)
            
            # Ensure path is still within project root
            if not os.path.abspath(full_path).startswith(os.path.abspath(self.project_root)):
                return ProjectFileReaderResponse(
                    content="",
                    error="Invalid path: Attempts to access file outside project",
                    file_path=args.relative_project_path,
                    exists=False
                )
            
            # Check if file exists
            if not os.path.exists(full_path) or not os.path.isfile(full_path):
                return ProjectFileReaderResponse(
                    content="",
                    error="File does not exist",
                    file_path=args.relative_project_path,
                    exists=False
                )
            
            # Read file content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return ProjectFileReaderResponse(
                content=content,
                error=None,
                file_path=args.relative_project_path,
                exists=True
            )
            
        except Exception as e:
            return ProjectFileReaderResponse(
                content="",
                error=f"Error reading file: {str(e)}",
                file_path=args.relative_project_path,
                exists=False
            )
