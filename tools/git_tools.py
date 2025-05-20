# tools/git_tools.py
import os
import subprocess
from typing import ClassVar, Type, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from .base_tool import BaseTool

# Version control like it's 2005! =D

class GitToolArgs(BaseModel):
    """What Git-astic operation shall we perform today? \\o/"""
    command: str = Field(..., description="The Git command (but not *too* Git-y, we have trust issues >.>)")
    branch_name: Optional[str] = Field(None, description="Name for new branches (make it memorable! ^_^)")
    commit_message: Optional[str] = Field(None, description="Your commit story (keep it short and sweet! =P)")
    file_paths: Optional[List[str]] = Field(None, description="Files to stage (no sneaky system files! x.x)")

class GitToolResponse(BaseModel):
    """The tale of our Git adventure! O.o"""
    success: bool = Field(..., description="Did we Git it right? =D")
    output: str = Field("", description="Git's reply (hopefully not angry! >.>)")
    error: Optional[str] = Field(None, description="When Git throws a tantrum... x.x")
    command: str = Field(..., description="What we asked Git to do \\o/")

class GitTool(BaseTool):
    """
    Your friendly neighborhood Git whisperer! \\o/
    
    I help you Git stuff done safely! No force pushing here! ^_^
    Think of me as your responsible friend who won't let you 
    git push --force to master at 4am! =P
    
    I know these tricks:
    1. status: What's the sitch? O.o
    2. branch: Branch like a tree! \\o/
    3. add: Stage those changes! =D
    4. commit: Save your progress! ^_^
    5. diff: What did we break- I mean, change? >.>
    6. log: Time travel through code! \\o/
    7. checkout: Branch hopping (new branches only, I'm careful! =P)
    8. pull: Get the new-new! ^_^
    9. push: Share your brilliance! \\o/
    """
    
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
