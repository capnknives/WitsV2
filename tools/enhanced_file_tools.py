# tools/enhanced_file_tools.py
"""
Enhanced File Access System for WITS Nexus v2.
Provides advanced file operations with security and validation.
For when you need ALL the file superpowers! \o/
"""

import os
import glob
import shutil
import aiofiles
import logging
import fnmatch
from typing import ClassVar, Type, Dict, Any, List, Optional, Union, Set
from pydantic import BaseModel, Field, validator

from core.autonomy.enhanced_tool_base import EnhancedTool
from tools.base_tool import ToolResponse

# Safe file operations with path validation and security checks

class EnhancedFileAccessConfig(BaseModel):
    """Configuration for enhanced file access tools."""
    base_directories: List[str] = Field(
        default_factory=lambda: ["output"], 
        description="List of allowed base directories for file operations"
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ["*"],
        description="List of allowed file extensions (* for any)"
    )
    blocked_extensions: List[str] = Field(
        default_factory=lambda: ["exe", "dll", "so", "dylib", "bat", "sh", "com"],
        description="List of blocked file extensions for security"
    )
    max_file_size: int = Field(
        10 * 1024 * 1024,  # 10 MB default
        description="Maximum file size in bytes"
    )

class FilePathArgs(BaseModel):
    """Base class for arguments requiring a file path."""
    file_path: str = Field(..., description="Path to the file (relative to an allowed directory)")
    
    @validator('file_path')
    def validate_path_characters(cls, v):
        """Validate file path to prevent path traversal attacks."""
        if '..' in v or '~' in v:
            raise ValueError("Path cannot contain '..' or '~' for security reasons")
        return v

# --- Enhanced File Search Tool ---

class FileSearchArgs(BaseModel):
    """Arguments for searching files."""
    pattern: str = Field(..., description="Glob pattern to search for files")
    recursive: bool = Field(True, description="Whether to search recursively")
    base_dir: Optional[str] = Field(None, description="Base directory to search in (must be in allowed list)")
    
    @validator('pattern')
    def validate_pattern(cls, v):
        """Validate search pattern for security."""
        if '..' in v:
            raise ValueError("Pattern cannot contain '..' for security reasons")
        return v

class FileSearchResult(BaseModel):
    """Results of a file search operation."""
    files: List[str] = Field(default_factory=list, description="List of matching file paths")
    error: Optional[str] = Field(None, description="Error message if the search failed")
    count: int = Field(0, description="Number of files found")

class EnhancedFileSearchTool(EnhancedTool):
    """
    Find files matching patterns across allowed directories.
    I'm like a detective for your files! \o/
    """
    
    name: ClassVar[str] = "enhanced_file_search"
    description: ClassVar[str] = "Search for files matching a pattern across allowed directories."
    args_schema: ClassVar[Type[BaseModel]] = FileSearchArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.file_config = EnhancedFileAccessConfig(**(self.config.get("file_access", {}) or {}))
        self.logger = logging.getLogger('WITS.EnhancedFileSearchTool')
    
    def _is_allowed_directory(self, base_dir: str) -> bool:
        """Check if a directory is in the allowed list."""
        return any(
            os.path.abspath(base_dir).startswith(os.path.abspath(allowed_dir))
            for allowed_dir in self.file_config.base_directories
        )
    
    async def _execute_impl(self, args: FileSearchArgs) -> ToolResponse[FileSearchResult]:
        """Implementation of file search."""
        try:
            # Determine base directory for search
            base_dir = args.base_dir or self.file_config.base_directories[0]
            
            # Security check
            if not self._is_allowed_directory(base_dir):
                return ToolResponse[FileSearchResult](
                    status_code=403,
                    error_message=f"Access denied: {base_dir} is not in the allowed directories",
                    output=FileSearchResult(
                        files=[],
                        error=f"Access denied: Directory not in allowed list",
                        count=0
                    )
                )
            
            # Build search pattern
            search_path = os.path.join(base_dir, args.pattern)
            
            # Perform the search
            matching_files = []
            if args.recursive:
                for root, _, files in os.walk(base_dir):
                    for filename in files:
                        if fnmatch.fnmatch(filename, args.pattern):
                            rel_path = os.path.join(root, filename)
                            # Convert to relative path from base_dir
                            rel_path = os.path.relpath(rel_path, base_dir)
                            matching_files.append(rel_path)
            else:
                matching_files = [
                    os.path.relpath(f, base_dir)
                    for f in glob.glob(search_path)
                    if os.path.isfile(f)
                ]
            
            # Filter out any files with blocked extensions
            filtered_files = []
            for file_path in matching_files:
                ext = os.path.splitext(file_path)[1].lstrip('.').lower()
                if ext not in self.file_config.blocked_extensions:
                    filtered_files.append(file_path)
            
            return ToolResponse[FileSearchResult](
                status_code=200,
                output=FileSearchResult(
                    files=filtered_files,
                    count=len(filtered_files)
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error in file search: {str(e)}")
            return ToolResponse[FileSearchResult](
                status_code=500,
                error_message=f"Error searching files: {str(e)}",
                output=FileSearchResult(
                    files=[],
                    error=str(e),
                    count=0
                )
            )

# --- Enhanced File Read Tool ---

class EnhancedReadFileArgs(FilePathArgs):
    """Arguments for enhanced file reading."""
    line_range: Optional[List[int]] = Field(
        None, 
        description="Optional start and end line numbers for partial file reading"
    )
    
class EnhancedReadFileResponse(BaseModel):
    """Response from enhanced file read operation."""
    content: str = Field("", description="File contents")
    file_path: str = Field(..., description="Path to the file that was read")
    exists: bool = Field(..., description="Whether the file exists")
    size_bytes: int = Field(0, description="Size of the file in bytes")
    line_count: int = Field(0, description="Number of lines in the file content")
    error: Optional[str] = Field(None, description="Error message if the read failed")

class EnhancedReadFileTool(EnhancedTool):
    """
    Enhanced file reading tool with security checks and line range support.
    I'm your file reading expert! Get exactly what you need! \o/
    """
    
    name: ClassVar[str] = "enhanced_read_file"
    description: ClassVar[str] = "Read the contents of a file with security validations and line range support."
    args_schema: ClassVar[Type[BaseModel]] = EnhancedReadFileArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.file_config = EnhancedFileAccessConfig(**(self.config.get("file_access", {}) or {}))
        self.logger = logging.getLogger('WITS.EnhancedReadFileTool')
    
    def _resolve_file_path(self, file_path: str) -> str:
        """Resolve and validate a file path against allowed directories."""
        # Try each allowed base directory
        for base_dir in self.file_config.base_directories:
            full_path = os.path.abspath(os.path.join(base_dir, file_path))
            base_abs = os.path.abspath(base_dir)
            
            # Security check: Make sure the path is within an allowed directory
            if os.path.commonpath([full_path, base_abs]) == base_abs and os.path.exists(full_path):
                return full_path
        
        # If we get here, the file isn't in any allowed directory
        raise ValueError(f"File path {file_path} is not in any allowed directory")
    
    async def _execute_impl(self, args: EnhancedReadFileArgs) -> ToolResponse[EnhancedReadFileResponse]:
        """Implementation of enhanced file reading."""
        try:
            # Resolve the file path
            try:
                full_path = self._resolve_file_path(args.file_path)
            except ValueError as e:
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=403,
                    error_message=str(e),
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=False,
                        size_bytes=0,
                        line_count=0,
                        error=str(e)
                    )
                )
            
            # Check if the file exists
            if not os.path.exists(full_path):
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=404,
                    error_message=f"File not found: {args.file_path}",
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=False,
                        size_bytes=0,
                        line_count=0,
                        error=f"File not found: {args.file_path}"
                    )
                )
            
            # Check if it's a directory
            if os.path.isdir(full_path):
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=400,
                    error_message=f"Path is a directory, not a file: {args.file_path}",
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=True,
                        size_bytes=0,
                        line_count=0,
                        error=f"Path is a directory, not a file: {args.file_path}"
                    )
                )
            
            # Check file size
            file_size = os.path.getsize(full_path)
            if file_size > self.file_config.max_file_size:
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=413,
                    error_message=f"File too large: {file_size} bytes (max: {self.file_config.max_file_size})",
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=True,
                        size_bytes=file_size,
                        line_count=0,
                        error=f"File too large: {file_size} bytes (max: {self.file_config.max_file_size})"
                    )
                )
            
            # Check file extension
            ext = os.path.splitext(full_path)[1].lstrip('.').lower()
            if (self.file_config.allowed_extensions != ["*"] and 
                ext not in self.file_config.allowed_extensions):
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=403,
                    error_message=f"File extension not allowed: {ext}",
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=True,
                        size_bytes=file_size,
                        line_count=0,
                        error=f"File extension not allowed: {ext}"
                    )
                )
            
            if ext in self.file_config.blocked_extensions:
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=403,
                    error_message=f"File extension blocked for security: {ext}",
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=True,
                        size_bytes=file_size,
                        line_count=0,
                        error=f"File extension blocked for security: {ext}"
                    )
                )
            
            # Read file
            try:
                if args.line_range:
                    # Partial reading with line ranges
                    start_line = args.line_range[0] if len(args.line_range) > 0 else 0
                    end_line = args.line_range[1] if len(args.line_range) > 1 else None
                    
                    lines = []
                    line_count = 0
                    
                    async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                        i = 0
                        async for line in f:
                            if i >= start_line and (end_line is None or i <= end_line):
                                lines.append(line)
                            i += 1
                            line_count += 1
                    
                    content = "".join(lines)
                    read_line_count = len(lines)
                    
                else:
                    # Full file reading
                    async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                    line_count = content.count('\n') + (1 if content else 0)
                    read_line_count = line_count
                
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=200,
                    output=EnhancedReadFileResponse(
                        content=content,
                        file_path=args.file_path,
                        exists=True,
                        size_bytes=file_size,
                        line_count=read_line_count
                    )
                )
            
            except UnicodeDecodeError:
                return ToolResponse[EnhancedReadFileResponse](
                    status_code=415,
                    error_message=f"File appears to be binary and cannot be read as text: {args.file_path}",
                    output=EnhancedReadFileResponse(
                        content="",
                        file_path=args.file_path,
                        exists=True,
                        size_bytes=file_size,
                        line_count=0,
                        error=f"File appears to be binary and cannot be read as text: {args.file_path}"
                    )
                )
                
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            return ToolResponse[EnhancedReadFileResponse](
                status_code=500,
                error_message=f"Error reading file: {str(e)}",
                output=EnhancedReadFileResponse(
                    content="",
                    file_path=args.file_path, 
                    exists=False,
                    size_bytes=0,
                    line_count=0,
                    error=f"Error reading file: {str(e)}"
                )
            )

# --- Enhanced File Write Tool ---

class EnhancedWriteFileArgs(FilePathArgs):
    """Arguments for enhanced file writing."""
    content: str = Field(..., description="Content to write to the file")
    mode: str = Field("w", description="Write mode: 'w' (overwrite) or 'a' (append)")
    
    @validator('mode')
    def validate_mode(cls, v):
        """Validate write mode."""
        if v not in ('w', 'a'):
            raise ValueError("Mode must be 'w' (overwrite) or 'a' (append)")
        return v

class EnhancedWriteFileResponse(BaseModel):
    """Response from enhanced file write operation."""
    success: bool = Field(..., description="Whether the write was successful")
    file_path: str = Field(..., description="Path to the file that was written")
    bytes_written: int = Field(0, description="Number of bytes written")
    error: Optional[str] = Field(None, description="Error message if the write failed")

class EnhancedWriteFileTool(EnhancedTool):
    """
    Enhanced file writing tool with security checks and flexible modes.
    I can write and update files with precision and safety! \o/
    """
    
    name: ClassVar[str] = "enhanced_write_file"
    description: ClassVar[str] = "Write content to a file with security validations and multiple modes."
    args_schema: ClassVar[Type[BaseModel]] = EnhancedWriteFileArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.file_config = EnhancedFileAccessConfig(**(self.config.get("file_access", {}) or {}))
        self.logger = logging.getLogger('WITS.EnhancedWriteFileTool')
    
    def _resolve_write_path(self, file_path: str) -> str:
        """Resolve and validate a file path for writing against allowed directories."""
        # Try each allowed base directory
        for base_dir in self.file_config.base_directories:
            # Create the directory if it doesn't exist
            os.makedirs(base_dir, exist_ok=True)
            
            full_path = os.path.abspath(os.path.join(base_dir, file_path))
            base_abs = os.path.abspath(base_dir)
            
            # Security check: Make sure the path is within an allowed directory
            if os.path.commonpath([full_path, base_abs]) == base_abs:
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                return full_path
        
        # If we get here, the file isn't in any allowed directory
        raise ValueError(f"File path {file_path} is not in any allowed directory")
    
    async def _execute_impl(self, args: EnhancedWriteFileArgs) -> ToolResponse[EnhancedWriteFileResponse]:
        """Implementation of enhanced file writing."""
        try:
            # Resolve the file path
            try:
                full_path = self._resolve_write_path(args.file_path)
            except ValueError as e:
                return ToolResponse[EnhancedWriteFileResponse](
                    status_code=403,
                    error_message=str(e),
                    output=EnhancedWriteFileResponse(
                        success=False,
                        file_path=args.file_path,
                        bytes_written=0,
                        error=str(e)
                    )
                )
            
            # Check file extension
            ext = os.path.splitext(full_path)[1].lstrip('.').lower()
            if (self.file_config.allowed_extensions != ["*"] and 
                ext not in self.file_config.allowed_extensions):
                return ToolResponse[EnhancedWriteFileResponse](
                    status_code=403,
                    error_message=f"File extension not allowed: {ext}",
                    output=EnhancedWriteFileResponse(
                        success=False,
                        file_path=args.file_path,
                        bytes_written=0,
                        error=f"File extension not allowed: {ext}"
                    )
                )
            
            if ext in self.file_config.blocked_extensions:
                return ToolResponse[EnhancedWriteFileResponse](
                    status_code=403,
                    error_message=f"File extension blocked for security: {ext}",
                    output=EnhancedWriteFileResponse(
                        success=False,
                        file_path=args.file_path,
                        bytes_written=0,
                        error=f"File extension blocked for security: {ext}"
                    )
                )
            
            # Check content size
            content_size = len(args.content.encode('utf-8'))
            if content_size > self.file_config.max_file_size:
                return ToolResponse[EnhancedWriteFileResponse](
                    status_code=413,
                    error_message=f"Content too large: {content_size} bytes (max: {self.file_config.max_file_size})",
                    output=EnhancedWriteFileResponse(
                        success=False,
                        file_path=args.file_path,
                        bytes_written=0,
                        error=f"Content too large: {content_size} bytes (max: {self.file_config.max_file_size})"
                    )
                )
            
            # Write file
            async with aiofiles.open(full_path, mode=f"{args.mode}", encoding='utf-8') as f:
                await f.write(args.content)
            
            return ToolResponse[EnhancedWriteFileResponse](
                status_code=200,
                output=EnhancedWriteFileResponse(
                    success=True,
                    file_path=args.file_path,
                    bytes_written=content_size
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error writing file: {str(e)}")
            return ToolResponse[EnhancedWriteFileResponse](
                status_code=500,
                error_message=f"Error writing file: {str(e)}",
                output=EnhancedWriteFileResponse(
                    success=False,
                    file_path=args.file_path,
                    bytes_written=0,
                    error=f"Error writing file: {str(e)}"
                )
            )

# --- Enhanced File Delete Tool ---

class EnhancedDeleteFileArgs(FilePathArgs):
    """Arguments for enhanced file deletion."""
    recursive: bool = Field(False, description="Whether to recursively delete directories")

class EnhancedDeleteFileResponse(BaseModel):
    """Response from enhanced file delete operation."""
    success: bool = Field(..., description="Whether the delete was successful")
    file_path: str = Field(..., description="Path to the file that was deleted")
    error: Optional[str] = Field(None, description="Error message if the deletion failed")

class EnhancedDeleteFileTool(EnhancedTool):
    """
    Enhanced file deletion tool with security checks.
    I'm the cleanup crew! I'll remove files safely! \o/
    """
    
    name: ClassVar[str] = "enhanced_delete_file"
    description: ClassVar[str] = "Delete a file or directory with security validations."
    args_schema: ClassVar[Type[BaseModel]] = EnhancedDeleteFileArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.file_config = EnhancedFileAccessConfig(**(self.config.get("file_access", {}) or {}))
        self.logger = logging.getLogger('WITS.EnhancedDeleteFileTool')
        
        # Define protected directories that should never be deleted
        self.protected_directories = set(self.config.get("protected_directories", []))
        
    def _resolve_delete_path(self, file_path: str) -> str:
        """Resolve and validate a file path for deletion against allowed directories."""
        # Try each allowed base directory
        for base_dir in self.file_config.base_directories:
            full_path = os.path.abspath(os.path.join(base_dir, file_path))
            base_abs = os.path.abspath(base_dir)
            
            # Never delete the base directory itself
            if full_path == base_abs:
                raise ValueError(f"Cannot delete base directory {base_dir}")
                
            # Never delete protected directories
            for protected_dir in self.protected_directories:
                protected_abs = os.path.abspath(protected_dir)
                if full_path == protected_abs:
                    raise ValueError(f"Cannot delete protected directory {protected_dir}")
            
            # Security check: Make sure the path is within an allowed directory
            if os.path.commonpath([full_path, base_abs]) == base_abs and os.path.exists(full_path):
                return full_path
        
        # If we get here, the file isn't in any allowed directory
        raise ValueError(f"File path {file_path} is not in any allowed directory or does not exist")
    
    async def _execute_impl(self, args: EnhancedDeleteFileArgs) -> ToolResponse[EnhancedDeleteFileResponse]:
        """Implementation of enhanced file deletion."""
        try:
            # Resolve the file path
            try:
                full_path = self._resolve_delete_path(args.file_path)
            except ValueError as e:
                return ToolResponse[EnhancedDeleteFileResponse](
                    status_code=403,
                    error_message=str(e),
                    output=EnhancedDeleteFileResponse(
                        success=False,
                        file_path=args.file_path,
                        error=str(e)
                    )
                )
            
            # Check if the path exists
            if not os.path.exists(full_path):
                return ToolResponse[EnhancedDeleteFileResponse](
                    status_code=404,
                    error_message=f"Path not found: {args.file_path}",
                    output=EnhancedDeleteFileResponse(
                        success=False,
                        file_path=args.file_path,
                        error=f"Path not found: {args.file_path}"
                    )
                )
            
            # Delete file or directory
            if os.path.isdir(full_path):
                if not args.recursive:
                    return ToolResponse[EnhancedDeleteFileResponse](
                        status_code=400,
                        error_message=f"Cannot delete directory without recursive=True: {args.file_path}",
                        output=EnhancedDeleteFileResponse(
                            success=False,
                            file_path=args.file_path,
                            error=f"Cannot delete directory without recursive=True: {args.file_path}"
                        )
                    )
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            
            return ToolResponse[EnhancedDeleteFileResponse](
                status_code=200,
                output=EnhancedDeleteFileResponse(
                    success=True,
                    file_path=args.file_path
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error deleting file: {str(e)}")
            return ToolResponse[EnhancedDeleteFileResponse](
                status_code=500,
                error_message=f"Error deleting file: {str(e)}",
                output=EnhancedDeleteFileResponse(
                    success=False,
                    file_path=args.file_path,
                    error=f"Error deleting file: {str(e)}"
                )
            )
