# Welcome to the file playground! Where bytes become dreams! \\o/
"""
Your friendly file operations toolkit! =D
We've got everything you need:
- Reading files (for those curious minds! ^_^)
- Writing files (creating digital art! \\o/)
- Listing files (like a really organized treasure hunt! O.o)
"""

import os
import aiofiles
import logging
from typing import ClassVar, Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_tool import BaseTool

# Let's play with files like it's 1999! \\o/

# --- File Read Tool ---

class ReadFileArgs(BaseModel):
    """What are we reading today? =D"""
    file_path: str = Field(..., description="Where's our target file hiding? ^_^")

class ReadFileResponse(BaseModel):
    """The treasures we found in the file! \\o/"""
    content: str = Field("", description="All the juicy contents! =P")
    error: Optional[str] = Field(None, description="Oops moments... >.>")
    exists: bool = Field(..., description="Is it real or just fantasy? O.o")
    file_path: str = Field(..., description="The path we looked in! \\o/")

class ReadFileTool(BaseTool):
    """
    Your personal file detective! I find and read files so you don't have to! ^_^
    Think of me as a very eager librarian with safety goggles! \\o/
    """
    
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read the contents of a file. Path is relative to the output directory."
    args_schema: ClassVar[Type[BaseModel]] = ReadFileArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
    """The recipe for our file creation magic! =D"""
    file_path: str = Field(..., description="Where shall we create our masterpiece? \\o/")
    content: str = Field(..., description="The stuff of legends (or just text, that works too ^_^)")
    create_dirs: bool = Field(False, description="Should we make folders too? Careful with this power! >.>")
    append: bool = Field(False, description="Add to existing file? Like a story that never ends! \\o/")

class WriteFileResponse(BaseModel):
    """How did our file writing adventure go? O.o"""
    success: bool = Field(..., description="Did we do the thing? =D")
    error: Optional[str] = Field(None, description="When things go whoopsie! x.x")
    file_path: str = Field(..., description="Where we left our masterpiece! \\o/")
    bytes_written: Optional[int] = Field(None, description="How many bytes did we write? So many numbers! @.@")

class WriteFileTool(BaseTool):
    """
    The artist formerly known as 'file writer'! \\o/
    I create and edit files like Bob Ross paints happy little trees! ^_^
    Just don't ask me to write system files... I have standards! =P
    """
    
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Write content to a file. Path is relative to the output directory."
    args_schema: ClassVar[Type[BaseModel]] = WriteFileArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
            
            # Get bytes_written (as reported by disk)
            bytes_written = os.path.getsize(full_path)
            
            return WriteFileResponse(
                success=True,
                file_path=file_path,
                bytes_written=bytes_written,
                error=None
            )
        except Exception as e:
            return WriteFileResponse(
                success=False,
                error=f"Error writing to file: {str(e)}",
                file_path=file_path,
                bytes_written=None
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
    """How shall we explore this digital forest? O.o"""
    directory: str = Field(..., description="Which folder should we peek into? \\o/")
    recursive: bool = Field(False, description="Go deeper? Like inception, but with folders! =D")
    include_directories: bool = Field(True, description="Should we include folders too? The more the merrier! ^_^")
    file_extension: Optional[str] = Field(None, description="Only files with this extension? Picky picky! >.>")

class FileInfo(BaseModel):
    """Everything you wanted to know about a file (but were afraid to ask >.>)"""
    name: str = Field(..., description="The file's stage name ^_^")
    path: str = Field(..., description="Where to find this superstar! \\o/")
    is_dir: bool = Field(..., description="Is it a folder in disguise? O.o")
    size: Optional[int] = Field(None, description="How chunky is it? (in bytes, not cookies x.x)")

class ListFilesResponse(BaseModel):
    """The results of our file hunting expedition! =D"""
    files: List[FileInfo] = Field([], description="All the files we rounded up! \\o/")
    directory: str = Field(..., description="The home base of our expedition! \\o/")
    error: Optional[str] = Field(None, description="When the file hunt goes wrong >.>")
    total_count: int = Field(0, description="How many treasures did we find? =D")

class ListFilesTool(BaseTool):
    """
    Your file system tour guide! ^_^
    I explore folders like Indiana Jones explores temples!
    Just with less booby traps and more error handling! =P
    """
    
    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = "List files in a directory. Path is relative to the output directory."
    args_schema: ClassVar[Type[BaseModel]] = ListFilesArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
                                is_dir=True,
                                size=None
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
                            is_dir=False,
                            size=size
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
                            is_dir=is_dir,
                            size=size
                        ))
            
            return ListFilesResponse(
                files=files,
                directory=directory,
                error=None,
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
    """
    The Swiss Army Knife of file operations! \\o/
    I'm like a file ninja - reading, writing, and listing with style! ^_^
    Safety first though, we keep everything in a cozy directory! =D
    """
    
    name: ClassVar[str] = "file"
    description: ClassVar[str] = "Perform file operations (read, write, list) in a safe directory. Like a responsible file DJ! \\o/"
    
    def __init__(self, base_path: str = "data/user_files"):
        """Initialize with base path."""
        super().__init__()
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
    
    def _ensure_safe_path(self, path: str) -> str:
        """
        Our file system bouncer! (•_•)
        Making sure no paths try to sneak past the velvet rope! >.>
        """
        # Convert to absolute path
        abs_path = os.path.abspath(os.path.join(self.base_path, path))
        
        # Check if the path is within base_path
        if not abs_path.startswith(self.base_path):
            raise ValueError(f"Path {path} is outside of allowed directory.")
        
        return abs_path
    
    async    def read_file(self, path: str) -> ReadFileResponse:
        """Story time! Let's see what secrets this file holds! ^o^"""
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
                exists=True,
                error=None
            )
        except Exception as e:
            return ReadFileResponse(
                content="",
                error=str(e),
                file_path=path,
                exists=False
            )
    
    async    def list_files(self, args: ListFilesArgs) -> ListFilesResponse:
        """Time for a digital scavenger hunt! Let's find those files! \\o/"""
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
                                is_dir=True,
                                size=None
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
                        is_dir=False,
                        size=size
                    ))
            
            return ListFilesResponse(
                files=files,
                directory=args.directory,
                error=None,
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
        """
        Time to do the thing! =D
        Like a file operations traffic cop, directing your requests to the right place! ^_^
        """
        if isinstance(args, ReadFileArgs):
            return await self.read_file(args.file_path)
        elif isinstance(args, ListFilesArgs):
            return await self.list_files(args)
        else:
            raise ValueError(f"Unsupported argument type: {type(args)}")
