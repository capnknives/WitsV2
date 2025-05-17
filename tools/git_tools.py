# tools/git_tools.py
import os
import subprocess
from typing import ClassVar, Type, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from .base_tool import BaseTool

class GitToolArgs(BaseModel):
    """Arguments for Git operations."""
    command: str = Field(..., description="Git command to execute (e.g., 'status', 'branch', 'add', 'commit')")
    branch_name: Optional[str] = Field(None, description="Branch name for branch-related operations")
    commit_message: Optional[str] = Field(None, description="Commit message for commit operations")
    file_paths: Optional[List[str]] = Field(None, description="List of file paths for git operations")

class GitToolResponse(BaseModel):
    """Response from Git operations."""
    success: bool = Field(False, description="Whether the operation was successful")
    output: str = Field("", description="Output from the git command")
    error: Optional[str] = Field(None, description="Error message if the operation failed")
    command: str = Field(..., description="The git command that was executed")

class GitTool(BaseTool):
    """Tool for safely executing Git operations."""
    
    name: ClassVar[str] = "git_tool"
    description: ClassVar[str] = "Execute Git operations safely within the project repository."
    args_schema: ClassVar[Type[BaseModel]] = GitToolArgs
    
    # List of allowed Git commands for safety
    ALLOWED_COMMANDS = {
        "status": "Get repository status",
        "branch": "Create or list branches",
        "add": "Stage files",
        "commit": "Commit changes",
        "diff": "Show changes",
        "log": "Show commit history",
        "checkout": "Switch branches (restricted to new branches only)",
        "pull": "Pull changes from remote",
        "push": "Push changes to remote"
    }
    
    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize with configuration."""
        super().__init__()
        self.config = config
        # Determine project root - this should be where .git is located
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    async def execute(self, args: GitToolArgs) -> GitToolResponse:
        """
        Execute a Git command safely.
        
        Args:
            args: GitToolArgs containing the command and parameters
            
        Returns:
            GitToolResponse: The result of the git operation
        """
        # Extract base command (e.g., "checkout" from "checkout -b branch-name")
        base_command = args.command.split()[0].lower()
        
        # Validate command is allowed
        if base_command not in self.ALLOWED_COMMANDS:
            return GitToolResponse(
                success=False,
                output="",
                error=f"Git command '{base_command}' is not allowed. Allowed commands: {', '.join(self.ALLOWED_COMMANDS.keys())}",
                command=args.command
            )
        
        try:
            # Build the git command
            git_cmd = ["git"]
            
            # Handle specific commands
            if base_command == "checkout":
                # Only allow creating new branches
                if not args.branch_name:
                    return GitToolResponse(
                        success=False,
                        output="",
                        error="Branch name is required for checkout operation",
                        command=args.command
                    )
                git_cmd.extend(["checkout", "-b", args.branch_name])
            
            elif base_command == "commit":
                if not args.commit_message:
                    return GitToolResponse(
                        success=False,
                        output="",
                        error="Commit message is required for commit operation",
                        command=args.command
                    )
                git_cmd.extend(["commit", "-m", args.commit_message])
            
            elif base_command == "add":
                git_cmd.append("add")
                if args.file_paths:
                    git_cmd.extend(args.file_paths)
                else:
                    return GitToolResponse(
                        success=False,
                        output="",
                        error="File paths are required for add operation",
                        command=args.command
                    )
            
            else:
                # For other commands, split and add all parts
                git_cmd.extend(args.command.split())
            
            # Execute git command
            process = subprocess.run(
                git_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on git errors
            )
            
            # Check for git errors
            if process.returncode != 0:
                return GitToolResponse(
                    success=False,
                    output=process.stdout,
                    error=process.stderr,
                    command=" ".join(git_cmd)
                )
            
            return GitToolResponse(
                success=True,
                output=process.stdout,
                error=None,  # Added error=None
                command=" ".join(git_cmd)
            )
            
        except Exception as e:
            return GitToolResponse(
                success=False,
                output="",
                error=f"Error executing git command: {str(e)}",
                command=args.command
            )
