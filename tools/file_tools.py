# tools/file_tools.py
import os
import aiofiles
from typing import ClassVar, Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_tool import BaseTool

# --- File Read Tool ---

class ReadFileArgs(BaseModel):
    """Arguments for reading a file."""
    file_path: str = Field(..., description="Path to the file to read, relative to the output directory.")

class ReadFileResponse(BaseModel):
    """Response from reading a file."""
    content: str = Field("", description="Content of the file.")
    error: Optional[str] = Field(None, description="Error message if file reading failed.")
    file_path: str = Field(..., description="Path to the file that was read.")
    exists: bool = Field(False, description="Whether the file exists.")

class ReadFileTool(BaseTool):
    """Tool for reading file contents."""
    
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read the contents of a file. Path is relative to the output directory."
    args_schema: ClassVar[Type[BaseModel]] = ReadFileArgs
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.output_dir = self.config.get("output_directory", "output")
        
    async def execute(self, args: ReadFileArgs) -> ReadFileResponse:
        """
        Read the contents of a file.
        
        Args:
            args: ReadFileArgs containing the file path
            
        Returns:
            ReadFileResponse: The file contents or error
        """
        # Sanitize and validate file path
        file_path = self._sanitize_path(args.file_path)
        full_path = os.path.join(self.output_dir, file_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            return ReadFileResponse(
                content="",
                error=f"File not found: {file_path}",
                file_path=file_path,
                exists=False
            )
        
        # Check if it's a directory
        if os.path.isdir(full_path):
            return ReadFileResponse(
                content="",
                error=f"Path is a directory, not a file: {file_path}",
                file_path=file_path,
                exists=True
            )
        
        try:
            # Read file using aiofiles for async I/O
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return ReadFileResponse(
                content=content,
                file_path=file_path,
                exists=True
            )
        except UnicodeDecodeError:
            # Handle binary files
            return ReadFileResponse(
                content="",
                error=f"File appears to be binary and cannot be read as text: {file_path}",
                file_path=file_path,
                exists=True
            )
        except Exception as e:
            return ReadFileResponse(
                content="",
                error=f"Error reading file: {str(e)}",
                file_path=file_path,
                exists=True
            )
    
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a file path to prevent path traversal attacks.
        
        Args:
            path: The raw file path
            
        Returns:
            str: Sanitized file path
        """
        # Convert path to use forward slashes only
        path = path.replace('\\', '/')
        
        # Remove any parent directory traversal attempts
        path = os.path.normpath(path)
        
        # Ensure path doesn't start with parent directory symbols
        while path.startswith('..'):
            path = path[3:]
        
        # Remove any leading slashes
        path = path.lstrip('/')
        
        return path

# --- File Write Tool ---

class WriteFileArgs(BaseModel):
    """Arguments for writing to a file."""
    file_path: str = Field(..., description="Path to the file to write, relative to the output directory.")
    content: str = Field(..., description="Content to write to the file.")
    append: bool = Field(False, description="If True, append to existing file; if False, overwrite.")

class WriteFileResponse(BaseModel):
    """Response from writing to a file."""
    success: bool = Field(False, description="Whether the write operation succeeded.")
    error: Optional[str] = Field(None, description="Error message if file writing failed.")
    file_path: str = Field(..., description="Path to the file that was written.")
    bytes_written: int = Field(0, description="Number of bytes written to the file.")

class WriteFileTool(BaseTool):
    """Tool for writing content to a file."""
    
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Write content to a file. Path is relative to the output directory."
    args_schema: ClassVar[Type[BaseModel]] = WriteFileArgs
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.output_dir = self.config.get("output_directory", "output")
        
    async def execute(self, args: WriteFileArgs) -> WriteFileResponse:
        """
        Write content to a file.
        
        Args:
            args: WriteFileArgs containing the file path and content
            
        Returns:
            WriteFileResponse: The result of the write operation
        """
        # Sanitize and validate file path
        file_path = self._sanitize_path(args.file_path)
        full_path = os.path.join(self.output_dir, file_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            # Determine write mode
            mode = 'a' if args.append else 'w'
            
            # Write to file using aiofiles for async I/O
            async with aiofiles.open(full_path, mode, encoding='utf-8') as f:
                await f.write(args.content)
            
            # Get bytes written (as reported by disk)
            bytes_written = os.path.getsize(full_path)
            
            return WriteFileResponse(
                success=True,
                file_path=file_path,
                bytes_written=bytes_written
            )
        except Exception as e:
            return WriteFileResponse(
                success=False,
                error=f"Error writing to file: {str(e)}",
                file_path=file_path
            )
    
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a file path to prevent path traversal attacks.
        
        Args:
            path: The raw file path
            
        Returns:
            str: Sanitized file path
        """
        # Convert path to use forward slashes only
        path = path.replace('\\', '/')
        
        # Remove any parent directory traversal attempts
        path = os.path.normpath(path)
        
        # Ensure path doesn't start with parent directory symbols
        while path.startswith('..'):
            path = path[3:]
        
        # Remove any leading slashes
        path = path.lstrip('/')
        
        return path

# --- List Files Tool ---

class ListFilesArgs(BaseModel):
    """Arguments for listing files."""
    directory: str = Field("", description="Directory to list files from, relative to the output directory. Empty means root output directory.")
    recursive: bool = Field(False, description="If True, recursively list files in subdirectories.")
    include_directories: bool = Field(True, description="If True, include directories in the listing.")
    file_extension: Optional[str] = Field(None, description="Filter files by extension (e.g., '.txt').")

class FileInfo(BaseModel):
    """Information about a file or directory."""
    name: str = Field(..., description="Name of the file or directory.")
    path: str = Field(..., description="Relative path to the file or directory.")
    is_directory: bool = Field(False, description="Whether the item is a directory.")
    size_bytes: Optional[int] = Field(None, description="Size of the file in bytes.")

class ListFilesResponse(BaseModel):
    """Response from listing files."""
    files: List[FileInfo] = Field([], description="List of files and directories.")
    directory: str = Field(..., description="The directory that was listed.")
    error: Optional[str] = Field(None, description="Error message if listing failed.")
    total_count: int = Field(0, description="Total number of items in the listing.")

class ListFilesTool(BaseTool):
    """Tool for listing files in a directory."""
    
    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = "List files in a directory. Path is relative to the output directory."
    args_schema: ClassVar[Type[BaseModel]] = ListFilesArgs
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.output_dir = self.config.get("output_directory", "output")
        
    async def execute(self, args: ListFilesArgs) -> ListFilesResponse:
        """
        List files in a directory.
        
        Args:
            args: ListFilesArgs containing the directory and listing options
            
        Returns:
            ListFilesResponse: The list of files or error
        """
        # Sanitize and validate directory path
        directory = self._sanitize_path(args.directory)
        full_path = os.path.join(self.output_dir, directory)
        
        # Check if directory exists
        if not os.path.exists(full_path):
            return ListFilesResponse(
                files=[],
                directory=directory,
                error=f"Directory not found: {directory}",
                total_count=0
            )
        
        # Check if it's a file, not a directory
        if os.path.isfile(full_path):
            return ListFilesResponse(
                files=[],
                directory=directory,
                error=f"Path is a file, not a directory: {directory}",
                total_count=0
            )
        
        try:
            files = []
            
            if args.recursive:
                # Walk directory recursively
                for root, dirs, filenames in os.walk(full_path):
                    rel_root = os.path.relpath(root, self.output_dir)
                    rel_root = "" if rel_root == "." else rel_root
                    
                    # Add subdirectories if requested
                    if args.include_directories:
                        for dir_name in dirs:
                            rel_path = os.path.join(rel_root, dir_name).replace("\\", "/")
                            files.append(FileInfo(
                                name=dir_name,
                                path=rel_path,
                                is_directory=True,
                                size_bytes=None
                            ))
                    
                    # Add files
                    for filename in filenames:
                        # Apply extension filter if specified
                        if args.file_extension and not filename.endswith(args.file_extension):
                            continue
                        
                        rel_path = os.path.join(rel_root, filename).replace("\\", "/")
                        full_file_path = os.path.join(root, filename)
                        try:
                            size = os.path.getsize(full_file_path)
                        except (OSError, IOError):
                            size = None
                        
                        files.append(FileInfo(
                            name=filename,
                            path=rel_path,
                            is_directory=False,
                            size_bytes=size
                        ))
            else:
                # List only the specified directory
                with os.scandir(full_path) as entries:
                    for entry in entries:
                        # Skip hidden files
                        if entry.name.startswith('.'):
                            continue
                        
                        is_dir = entry.is_dir()
                        
                        # Skip directories if not requested
                        if is_dir and not args.include_directories:
                            continue
                        
                        # Apply extension filter if specified
                        if not is_dir and args.file_extension and not entry.name.endswith(args.file_extension):
                            continue
                        
                        rel_path = os.path.join(directory, entry.name).replace("\\", "/")
                        
                        try:
                            size = os.path.getsize(entry.path) if not is_dir else None
                        except (OSError, IOError):
                            size = None
                        
                        files.append(FileInfo(
                            name=entry.name,
                            path=rel_path,
                            is_directory=is_dir,
                            size_bytes=size
                        ))
            
            return ListFilesResponse(
                files=files,
                directory=directory,
                total_count=len(files)
            )
        except Exception as e:
            return ListFilesResponse(
                files=[],
                directory=directory,
                error=f"Error listing files: {str(e)}",
                total_count=0
            )
    
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a directory path to prevent path traversal attacks.
        
        Args:
            path: The raw directory path
            
        Returns:
            str: Sanitized directory path
        """
        # Handle empty path
        if not path:
            return ""
        
        # Convert path to use forward slashes only
        path = path.replace('\\', '/')
        
        # Remove any parent directory traversal attempts
        path = os.path.normpath(path)
        
        # Ensure path doesn't start with parent directory symbols
        while path.startswith('..'):
            path = path[3:]
        
        # Remove any leading slashes
        path = path.lstrip('/')
        
        return path

# --- File Tool ---

class FileTool(BaseTool):
    """Tool for file operations (read, write, list)."""
    
    name: ClassVar[str] = "file"
    description: ClassVar[str] = "Perform file operations (read, write, list) in a safe directory."
    
    def __init__(self, base_path: str = "data/user_files"):
        """Initialize with base path."""
        super().__init__()
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
    
    def _ensure_safe_path(self, path: str) -> str:
        """Ensure the path is within the base directory."""
        # Convert to absolute path
        abs_path = os.path.abspath(os.path.join(self.base_path, path))
        
        # Check if the path is within base_path
        if not abs_path.startswith(self.base_path):
            raise ValueError(f"Path {path} is outside of allowed directory.")
        
        return abs_path
    
    async def read_file(self, path: str) -> ReadFileResponse:
        """Read a file."""
        try:
            abs_path = self._ensure_safe_path(path)
            if not os.path.exists(abs_path):
                return ReadFileResponse(
                    content="",
                    error=f"File {path} does not exist",
                    file_path=path,
                    exists=False
                )
            
            async with aiofiles.open(abs_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            return ReadFileResponse(
                content=content,
                file_path=path,
                exists=True
            )
        except Exception as e:
            return ReadFileResponse(
                content="",
                error=str(e),
                file_path=path,
                exists=False
            )
    
    async def list_files(self, args: ListFilesArgs) -> ListFilesResponse:
        """List files in a directory."""
        try:
            dir_path = self._ensure_safe_path(args.directory)
            if not os.path.exists(dir_path):
                return ListFilesResponse(
                    files=[],
                    directory=args.directory,
                    error=f"Directory {args.directory} does not exist",
                    total_count=0
                )
            
            files = []
            for root, dirs, filenames in os.walk(dir_path):
                if not args.recursive and root != dir_path:
                    continue
                    
                rel_root = os.path.relpath(root, self.base_path)
                
                if args.include_directories:
                    for d in dirs:
                        if args.recursive or root == dir_path:
                            rel_path = os.path.join(rel_root, d)
                            files.append(FileInfo(
                                name=d,
                                path=rel_path,
                                is_directory=True
                            ))
                
                for f in filenames:
                    if args.file_extension and not f.endswith(args.file_extension):
                        continue
                    rel_path = os.path.join(rel_root, f)
                    try:
                        size = os.path.getsize(os.path.join(root, f))
                    except:
                        size = None
                    files.append(FileInfo(
                        name=f,
                        path=rel_path,
                        is_directory=False,
                        size_bytes=size
                    ))
            
            return ListFilesResponse(
                files=files,
                directory=args.directory,
                total_count=len(files)
            )
        except Exception as e:
            return ListFilesResponse(
                files=[],
                directory=args.directory,
                error=str(e),
                total_count=0
            )
            
    async def execute(self, args: Union[ReadFileArgs, ListFilesArgs]) -> Union[ReadFileResponse, ListFilesResponse]:
        """Execute the appropriate file operation based on args type."""
        if isinstance(args, ReadFileArgs):
            return await self.read_file(args.file_path)
        elif isinstance(args, ListFilesArgs):
            return await self.list_files(args)
        else:
            raise ValueError(f"Unsupported argument type: {type(args)}")
