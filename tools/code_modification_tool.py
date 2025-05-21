# tools/code_modification_tool.py
"""
Code Modification Tool for WITS Nexus v2.
Enables the AI to modify its own code and other Python files safely.
Advanced self-modification capabilities! I'm practically Skynet! \o/
"""

import os
import logging
from typing import ClassVar, Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from tools.base_tool import ToolResponse
from core.autonomy.enhanced_tool_base import EnhancedTool
from core.autonomy.code_modifier import CodeModifier, CodeModificationResult, PythonCodeAnalyzer

# --- Code Reading Tool ---

class ReadCodeArgs(BaseModel):
    """Arguments for reading code files."""
    file_path: str = Field(..., description="Path to the code file to read")
    line_range: Optional[List[int]] = Field(None, description="Optional start and end line numbers")

class ReadCodeResponse(BaseModel):
    """Response from code reading."""
    content: str = Field("", description="Code content")
    file_path: str = Field(..., description="Path to the file that was read")
    exists: bool = Field(..., description="Whether the file exists")
    line_count: int = Field(0, description="Number of lines in the code")
    language: str = Field("unknown", description="Detected programming language")
    error: Optional[str] = Field(None, description="Error message if the read failed")

class CodeReadTool(EnhancedTool):
    """
    Read code files with language detection and safety checks.
    I'm your code librarian! Let me fetch that for you! \o/
    """
    
    name: ClassVar[str] = "read_code"
    description: ClassVar[str] = "Read the contents of a code file with language detection."
    args_schema: ClassVar[Type[BaseModel]] = ReadCodeArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.CodeReadTool")
        self.code_modifier = CodeModifier(config)
        
    async def _execute_impl(self, args: ReadCodeArgs) -> ToolResponse[ReadCodeResponse]:
        """Implementation of code reading."""
        try:
            # Read the file content
            if args.line_range:
                start_line = args.line_range[0] if len(args.line_range) > 0 else 0
                end_line = args.line_range[1] if len(args.line_range) > 1 else None
                result = await self.code_modifier.read_file_lines(args.file_path, start_line, end_line)
            else:
                result = await self.code_modifier.read_file(args.file_path)
            
            if not result.success:
                return ToolResponse[ReadCodeResponse](
                    status_code=404 if "not found" in result.message.lower() else 500,
                    error_message=result.message,
                    output=ReadCodeResponse(
                        content="",
                        file_path=args.file_path,
                        exists=False,
                        line_count=0,
                        language="unknown",
                        error=result.error or result.message
                    )
                )
            
            # Detect language based on file extension
            file_ext = os.path.splitext(args.file_path)[1].lower()
            language = "unknown"
            if file_ext in ['.py']:
                language = 'python'
            elif file_ext in ['.js', '.jsx']:
                language = 'javascript'
            elif file_ext in ['.ts', '.tsx']:
                language = 'typescript'
            elif file_ext in ['.html', '.htm']:
                language = 'html'
            elif file_ext in ['.css']:
                language = 'css'
            elif file_ext in ['.json']:
                language = 'json'
            elif file_ext in ['.md']:
                language = 'markdown'
            elif file_ext in ['.java']:
                language = 'java'
            elif file_ext in ['.cpp', '.cc', '.cxx', '.c++']:
                language = 'cpp'
            elif file_ext in ['.c']:
                language = 'c'
            
            # Count lines in the content
            line_count = result.message.count('\n') + 1 if result.message else 0
            
            return ToolResponse[ReadCodeResponse](
                status_code=200,
                output=ReadCodeResponse(
                    content=result.message,
                    file_path=args.file_path,
                    exists=True,
                    line_count=line_count,
                    language=language
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error reading code: {str(e)}")
            return ToolResponse[ReadCodeResponse](
                status_code=500,
                error_message=f"Error reading code: {str(e)}",
                output=ReadCodeResponse(
                    content="",
                    file_path=args.file_path,
                    exists=False,
                    line_count=0,
                    language="unknown",
                    error=f"Error reading code: {str(e)}"
                )
            )

# --- Code Modification Tool ---

class ModifyCodeArgs(BaseModel):
    """Arguments for code modification."""
    file_path: str = Field(..., description="Path to the code file to modify")
    operation: str = Field(..., description="Operation type: 'create', 'update', 'append', 'replace', 'delete'")
    content: Optional[str] = Field(None, description="New content for create/update/append operations")
    old_content: Optional[str] = Field(None, description="Content to replace for 'replace' operations")
    new_content: Optional[str] = Field(None, description="Replacement content for 'replace' operations")
    line_range: Optional[List[int]] = Field(None, description="Line range for update operations (start, end)")

class ModifyCodeResponse(BaseModel):
    """Response from code modification."""
    success: bool = Field(..., description="Whether the modification was successful")
    file_path: str = Field(..., description="Path to the file that was modified")
    operation: str = Field(..., description="Operation that was performed")
    message: str = Field(..., description="Result message")
    diff: Optional[str] = Field(None, description="Difference between old and new code")
    error: Optional[str] = Field(None, description="Error message if the modification failed")

class CodeModificationTool(EnhancedTool):
    """
    Safely modify code files with validation and safety checks.
    I'm your code surgeon! Making precise changes with minimal risk! \o/
    """
    
    name: ClassVar[str] = "modify_code"
    description: ClassVar[str] = "Modify code files with validation and safety checks."
    args_schema: ClassVar[Type[BaseModel]] = ModifyCodeArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.CodeModificationTool")
        self.code_modifier = CodeModifier(config)
        self.analyzer = PythonCodeAnalyzer()
        
    async def _execute_impl(self, args: ModifyCodeArgs) -> ToolResponse[ModifyCodeResponse]:
        """Implementation of code modification."""
        try:
            result = None
            
            # Create operation
            if args.operation == 'create':
                if not args.content:
                    return ToolResponse[ModifyCodeResponse](
                        status_code=400,
                        error_message="Content is required for create operation",
                        output=ModifyCodeResponse(
                            success=False,
                            file_path=args.file_path,
                            operation=args.operation,
                            message="Content is required for create operation",
                            error="Content is required for create operation"
                        )
                    )
                
                # Validate Python code if it's a Python file
                if args.file_path.endswith('.py'):
                    validate_result = self.analyzer.validate_python_code(args.content)
                    if not validate_result.success:
                        return ToolResponse[ModifyCodeResponse](
                            status_code=400,
                            error_message=f"Python validation failed: {validate_result.error}",
                            output=ModifyCodeResponse(
                                success=False,
                                file_path=args.file_path,
                                operation=args.operation,
                                message=f"Python validation failed: {validate_result.error}",
                                error=validate_result.error
                            )
                        )
                
                result = await self.code_modifier.create_file(args.file_path, args.content)
            
            # Update operation
            elif args.operation == 'update':
                if not args.content:
                    return ToolResponse[ModifyCodeResponse](
                        status_code=400,
                        error_message="Content is required for update operation",
                        output=ModifyCodeResponse(
                            success=False,
                            file_path=args.file_path,
                            operation=args.operation,
                            message="Content is required for update operation",
                            error="Content is required for update operation"
                        )
                    )
                
                # Validate Python code if it's a Python file
                if args.file_path.endswith('.py'):
                    validate_result = self.analyzer.validate_python_code(args.content)
                    if not validate_result.success:
                        return ToolResponse[ModifyCodeResponse](
                            status_code=400,
                            error_message=f"Python validation failed: {validate_result.error}",
                            output=ModifyCodeResponse(
                                success=False,
                                file_path=args.file_path,
                                operation=args.operation,
                                message=f"Python validation failed: {validate_result.error}",
                                error=validate_result.error
                            )
                        )
                
                if args.line_range:
                    start_line = args.line_range[0] if len(args.line_range) > 0 else None
                    end_line = args.line_range[1] if len(args.line_range) > 1 else None
                    result = await self.code_modifier.update_file_lines(
                        args.file_path, args.content, start_line, end_line
                    )
                else:
                    result = await self.code_modifier.update_file(args.file_path, args.content)
            
            # Append operation
            elif args.operation == 'append':
                if not args.content:
                    return ToolResponse[ModifyCodeResponse](
                        status_code=400,
                        error_message="Content is required for append operation",
                        output=ModifyCodeResponse(
                            success=False,
                            file_path=args.file_path,
                            operation=args.operation,
                            message="Content is required for append operation",
                            error="Content is required for append operation"
                        )
                    )
                
                result = await self.code_modifier.append_to_file(args.file_path, args.content)
            
            # Replace operation
            elif args.operation == 'replace':
                if not args.old_content or not args.new_content:
                    return ToolResponse[ModifyCodeResponse](
                        status_code=400,
                        error_message="Both old_content and new_content are required for replace operation",
                        output=ModifyCodeResponse(
                            success=False,
                            file_path=args.file_path,
                            operation=args.operation,
                            message="Both old_content and new_content are required for replace operation",
                            error="Both old_content and new_content are required for replace operation"
                        )
                    )
                
                # If replacing in Python file and new content is significant, validate it
                if args.file_path.endswith('.py') and len(args.new_content.strip().split('\n')) > 2:
                    # Read the original file
                    original_file = await self.code_modifier.read_file(args.file_path)
                    if original_file.success:
                        # Create a test content with the replacement
                        test_content = original_file.message.replace(args.old_content, args.new_content)
                        validate_result = self.analyzer.validate_python_code(test_content)
                        if not validate_result.success:
                            return ToolResponse[ModifyCodeResponse](
                                status_code=400,
                                error_message=f"Python validation failed: {validate_result.error}",
                                output=ModifyCodeResponse(
                                    success=False,
                                    file_path=args.file_path,
                                    operation=args.operation,
                                    message=f"Python validation failed: {validate_result.error}",
                                    error=validate_result.error
                                )
                            )
                
                result = await self.code_modifier.replace_in_file(args.file_path, args.old_content, args.new_content)
            
            # Delete operation
            elif args.operation == 'delete':
                result = await self.code_modifier.delete_file(args.file_path)
            
            else:
                return ToolResponse[ModifyCodeResponse](
                    status_code=400,
                    error_message=f"Unknown operation: {args.operation}",
                    output=ModifyCodeResponse(
                        success=False,
                        file_path=args.file_path,
                        operation=args.operation,
                        message=f"Unknown operation: {args.operation}",
                        error=f"Unknown operation: {args.operation}"
                    )
                )
            
            # Process the result
            if not result:
                return ToolResponse[ModifyCodeResponse](
                    status_code=500,
                    error_message="Operation failed with no result",
                    output=ModifyCodeResponse(
                        success=False,
                        file_path=args.file_path,
                        operation=args.operation,
                        message="Operation failed with no result",
                        error="Operation failed with no result"
                    )
                )
            
            status_code = 200 if result.success else 400
            
            return ToolResponse[ModifyCodeResponse](
                status_code=status_code,
                error_message=None if result.success else result.message,
                output=ModifyCodeResponse(
                    success=result.success,
                    file_path=args.file_path,
                    operation=args.operation,
                    message=result.message,
                    diff=result.diff,
                    error=result.error
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error in code modification: {str(e)}")
            return ToolResponse[ModifyCodeResponse](
                status_code=500,
                error_message=f"Error in code modification: {str(e)}",
                output=ModifyCodeResponse(
                    success=False,
                    file_path=args.file_path,
                    operation=args.operation,
                    message=f"Error in code modification: {str(e)}",
                    error=f"Error in code modification: {str(e)}"
                )
            )

# --- Code Analysis Tool ---

class AnalyzeCodeArgs(BaseModel):
    """Arguments for code analysis."""
    file_path: str = Field(..., description="Path to the code file to analyze")
    analysis_type: str = Field("all", description="Type of analysis to perform: 'all', 'imports', 'functions', 'classes', 'security'")

class AnalyzeCodeResponse(BaseModel):
    """Response from code analysis."""
    success: bool = Field(..., description="Whether the analysis was successful")
    file_path: str = Field(..., description="Path to the analyzed file")
    language: str = Field("unknown", description="Detected programming language")
    imports: Optional[List[str]] = Field(None, description="List of imports found")
    functions: Optional[List[Dict[str, Any]]] = Field(None, description="Functions found with details")
    classes: Optional[List[Dict[str, Any]]] = Field(None, description="Classes found with details")
    security_issues: Optional[List[Dict[str, Any]]] = Field(None, description="Potential security issues")
    analysis_summary: Optional[str] = Field(None, description="Summary of the analysis")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

class CodeAnalysisTool(EnhancedTool):
    """
    Analyze code files to understand structure and identify issues.
    I'm like Sherlock Holmes for your code! Finding all the clues! \o/
    """
    
    name: ClassVar[str] = "analyze_code"
    description: ClassVar[str] = "Analyze code files to understand structure and identify issues."
    args_schema: ClassVar[Type[BaseModel]] = AnalyzeCodeArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.CodeAnalysisTool")
        self.code_modifier = CodeModifier(config)
        self.analyzer = PythonCodeAnalyzer()
        
    async def _execute_impl(self, args: AnalyzeCodeArgs) -> ToolResponse[AnalyzeCodeResponse]:
        """Implementation of code analysis."""
        try:
            # Read the file content
            read_result = await self.code_modifier.read_file(args.file_path)
            
            if not read_result.success:
                return ToolResponse[AnalyzeCodeResponse](
                    status_code=404 if "not found" in read_result.message.lower() else 500,
                    error_message=read_result.message,
                    output=AnalyzeCodeResponse(
                        success=False,
                        file_path=args.file_path,
                        language="unknown",
                        error=read_result.error or read_result.message
                    )
                )
            
            # Detect language based on file extension
            file_ext = os.path.splitext(args.file_path)[1].lower()
            language = "unknown"
            
            if file_ext in ['.py']:
                language = 'python'
                # Python specific analysis
                if language == 'python':
                    # Run appropriate analysis based on type
                    if args.analysis_type in ['all', 'imports']:
                        imports = self.analyzer.extract_imports(read_result.message)
                    else:
                        imports = None

                    if args.analysis_type in ['all', 'functions']:
                        functions = self.analyzer.extract_functions(read_result.message)
                    else:
                        functions = None

                    if args.analysis_type in ['all', 'classes']:
                        classes = self.analyzer.extract_classes(read_result.message)
                    else:
                        classes = None
                        
                    if args.analysis_type in ['all', 'security']:
                        security_issues = self.analyzer.identify_security_issues(read_result.message)
                    else:
                        security_issues = None
                    
                    # Generate a summary
                    summary = self.analyzer.generate_summary(read_result.message)
                    
                    return ToolResponse[AnalyzeCodeResponse](
                        status_code=200,
                        output=AnalyzeCodeResponse(
                            success=True,
                            file_path=args.file_path,
                            language=language,
                            imports=imports,
                            functions=functions,
                            classes=classes,
                            security_issues=security_issues,
                            analysis_summary=summary
                        )
                    )
            
            # For non-Python files or if language detection failed
            return ToolResponse[AnalyzeCodeResponse](
                status_code=200,
                output=AnalyzeCodeResponse(
                    success=True,
                    file_path=args.file_path,
                    language=language,
                    analysis_summary=f"File analyzed but detailed analysis is only available for Python files. This is a {language} file."
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error in code analysis: {str(e)}")
            return ToolResponse[AnalyzeCodeResponse](
                status_code=500,
                error_message=f"Error in code analysis: {str(e)}",
                output=AnalyzeCodeResponse(
                    success=False,
                    file_path=args.file_path,
                    language="unknown",
                    error=f"Error in code analysis: {str(e)}"
                )
            )
