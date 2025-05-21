# core/autonomy/code_modifier.py
"""
Code Modification utilities for WITS Nexus v2.
Enables the AI to modify its own code and other project files safely.
Self-awareness and adaptation in action! The future is now! \o/
"""

import os
import logging
import ast
import re
import difflib
from typing import List, Dict, Any, Optional, Tuple, Union
import aiofiles

class CodeModificationResult:
    """Result of a code modification operation."""
    
    def __init__(
        self, 
        success: bool,
        file_path: str,
        message: str,
        diff: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Initialize result with modification details."""
        self.success = success
        self.file_path = file_path
        self.message = message
        self.diff = diff
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "message": self.message,
            "diff": self.diff,
            "error": self.error
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "Success" if self.success else "Failed"
        return f"{status}: {self.message} ({self.file_path})"

class CodeModifier:
    """
    Safely modifies code files with validation and error checking.
    Like a responsible code surgeon - precise, cautious, and professional! ^_^
    """
    
    def __init__(self, 
                 workspace_root: str, 
                 allowed_paths: Optional[List[str]] = None,
                 restricted_paths: Optional[List[str]] = None):
        """
        Initialize the code modifier with safety constraints.
        
        Args:
            workspace_root: Root directory for code modifications
            allowed_paths: Specific paths allowed for modification (under workspace_root)
            restricted_paths: Paths that are restricted from modification
        """
        self.workspace_root = os.path.abspath(workspace_root)
        self.allowed_paths = [os.path.join(self.workspace_root, p) for p in (allowed_paths or [])]
        self.restricted_paths = [os.path.join(self.workspace_root, p) for p in (restricted_paths or [])]
        self.logger = logging.getLogger("WITS.Autonomy.CodeModifier")
    
    def _validate_path(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that a file path is allowed for modification.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Normalize path
        abs_path = os.path.abspath(file_path)
        
        # Check that it's under workspace_root
        if not abs_path.startswith(self.workspace_root):
            return False, f"Path must be within workspace: {self.workspace_root}"
        
        # Check if it's in restricted_paths
        for restricted in self.restricted_paths:
            if abs_path.startswith(restricted):
                return False, f"Path is in a restricted directory: {restricted}"
        
        # If allowed_paths is specified, check that the path is allowed
        if self.allowed_paths and not any(abs_path.startswith(allowed) for allowed in self.allowed_paths):
            allowed_desc = ", ".join(p.replace(self.workspace_root, "") for p in self.allowed_paths)
            return False, f"Path is not in an allowed directory. Allowed: {allowed_desc}"
        
        # Check if the directory exists (for new files)
        dir_path = os.path.dirname(abs_path)
        if not os.path.exists(dir_path):
            return False, f"Directory does not exist: {dir_path}"
        
        return True, ""
    
    async def _read_file(self, file_path: str) -> Tuple[bool, Union[str, Exception]]:
        """
        Read a file with error handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple[bool, Union[str, Exception]]: (success, content_or_error)
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return True, content
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return False, e
    
    async def _write_file(self, file_path: str, content: str) -> Tuple[bool, Optional[Exception]]:
        """
        Write content to a file with error handling.
        
        Args:
            file_path: Path to the file
            content: Content to write
            
        Returns:
            Tuple[bool, Optional[Exception]]: (success, error)
        """
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
                return True, None
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {str(e)}")
            return False, e
    
    def _validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax before applying changes.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _generate_diff(self, original: str, modified: str, file_path: str) -> str:
        """
        Generate a unified diff between original and modified code.
        
        Args:
            original: Original code
            modified: Modified code
            file_path: Path to the file (for diff header)
            
        Returns:
            str: Unified diff
        """
        original_lines = original.splitlines(True)
        modified_lines = modified.splitlines(True)
        diff = difflib.unified_diff(
            original_lines, 
            modified_lines, 
            fromfile=f"a/{file_path}", 
            tofile=f"b/{file_path}"
        )
        return "".join(diff)
    
    async def create_file(self, file_path: str, content: str) -> CodeModificationResult:
        """
        Create a new file with the provided content.
        
        Args:
            file_path: Path to create the file at
            content: Content for the new file
            
        Returns:
            CodeModificationResult: Result of the operation
        """
        # Validate path
        valid, message = self._validate_path(file_path)
        if not valid:
            return CodeModificationResult(False, file_path, "Path validation failed", None, message)
        
        # Check if file already exists
        if os.path.exists(file_path):
            return CodeModificationResult(False, file_path, "File already exists", None, "Cannot create existing file")
        
        # Validate syntax if it's a Python file
        if file_path.endswith('.py'):
            valid, error = self._validate_python_syntax(content)
            if not valid:
                return CodeModificationResult(False, file_path, "Syntax validation failed", None, error)
        
        # Write the file
        success, error = await self._write_file(file_path, content)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to write file", None, str(error))
        
        return CodeModificationResult(True, file_path, "File created successfully", None, None)
    
    async def modify_file(self, file_path: str, new_content: str) -> CodeModificationResult:
        """
        Replace a file's entire content.
        
        Args:
            file_path: Path to the file to modify
            new_content: New content for the file
            
        Returns:
            CodeModificationResult: Result of the operation
        """
        # Validate path
        valid, message = self._validate_path(file_path)
        if not valid:
            return CodeModificationResult(False, file_path, "Path validation failed", None, message)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return CodeModificationResult(False, file_path, "File does not exist", None, "Cannot modify non-existent file")
        
        # Read original content
        success, result = await self._read_file(file_path)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to read file", None, str(result))
        
        original_content = result
        
        # Validate syntax if it's a Python file
        if file_path.endswith('.py'):
            valid, error = self._validate_python_syntax(new_content)
            if not valid:
                return CodeModificationResult(False, file_path, "Syntax validation failed", None, error)
        
        # Generate diff
        diff = self._generate_diff(original_content, new_content, os.path.basename(file_path))
        
        # Write the file
        success, error = await self._write_file(file_path, new_content)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to write file", diff, str(error))
        
        return CodeModificationResult(True, file_path, "File modified successfully", diff, None)
    
    async def update_file_section(
        self, 
        file_path: str, 
        old_section: str, 
        new_section: str
    ) -> CodeModificationResult:
        """
        Update a specific section of a file while preserving the rest.
        
        Args:
            file_path: Path to the file to modify
            old_section: Section to replace
            new_section: New section content
            
        Returns:
            CodeModificationResult: Result of the operation
        """
        # Validate path
        valid, message = self._validate_path(file_path)
        if not valid:
            return CodeModificationResult(False, file_path, "Path validation failed", None, message)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return CodeModificationResult(False, file_path, "File does not exist", None, "Cannot modify non-existent file")
        
        # Read original content
        success, result = await self._read_file(file_path)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to read file", None, str(result))
        
        original_content = result
        
        # Ensure the section exists in the file
        if old_section not in original_content:
            return CodeModificationResult(
                False, 
                file_path, 
                "Section not found", 
                None, 
                "The specified section was not found in the file"
            )
        
        # Replace the section
        new_content = original_content.replace(old_section, new_section)
        
        # Validate syntax if it's a Python file
        if file_path.endswith('.py'):
            valid, error = self._validate_python_syntax(new_content)
            if not valid:
                return CodeModificationResult(False, file_path, "Syntax validation failed", None, error)
        
        # Generate diff
        diff = self._generate_diff(original_content, new_content, os.path.basename(file_path))
        
        # Write the file
        success, error = await self._write_file(file_path, new_content)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to write file", diff, str(error))
        
        return CodeModificationResult(True, file_path, "File section updated successfully", diff, None)
    
    async def insert_code(
        self, 
        file_path: str, 
        new_code: str, 
        after_line: Optional[int] = None,
        anchor: Optional[str] = None
    ) -> CodeModificationResult:
        """
        Insert code at a specific position in a file.
        
        Args:
            file_path: Path to the file to modify
            new_code: New code to insert
            after_line: Line number to insert after (0-indexed)
            anchor: String to insert after (use either this or after_line)
            
        Returns:
            CodeModificationResult: Result of the operation
        """
        # Validate path
        valid, message = self._validate_path(file_path)
        if not valid:
            return CodeModificationResult(False, file_path, "Path validation failed", None, message)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return CodeModificationResult(False, file_path, "File does not exist", None, "Cannot modify non-existent file")
        
        # Read original content
        success, result = await self._read_file(file_path)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to read file", None, str(result))
        
        original_content = result
        lines = original_content.splitlines(True)  # Keep line endings
        
        # Determine insertion point
        if anchor is not None:
            # Find the anchor text
            insertion_point = original_content.find(anchor)
            if insertion_point == -1:
                return CodeModificationResult(
                    False, 
                    file_path, 
                    "Anchor not found", 
                    None, 
                    f"The anchor text '{anchor}' was not found in the file"
                )
            
            # Find the end of the line containing the anchor
            insertion_point = original_content.find('\n', insertion_point)
            if insertion_point == -1:  # No newline after anchor
                insertion_point = len(original_content)
            else:
                insertion_point += 1  # Include the newline
        elif after_line is not None:
            # Validate line number
            if after_line < 0 or after_line >= len(lines):
                return CodeModificationResult(
                    False, 
                    file_path, 
                    "Invalid line number", 
                    None, 
                    f"Line number {after_line} is out of range (0-{len(lines)-1})"
                )
            
            # Calculate insertion point
            insertion_point = sum(len(lines[i]) for i in range(after_line + 1))
        else:
            # If neither is specified, append to end of file
            insertion_point = len(original_content)
        
        # Insert the new code
        new_content = (
            original_content[:insertion_point] + 
            (new_code if new_code.endswith('\n') else new_code + '\n') +
            original_content[insertion_point:]
        )
        
        # Validate syntax if it's a Python file
        if file_path.endswith('.py'):
            valid, error = self._validate_python_syntax(new_content)
            if not valid:
                return CodeModificationResult(False, file_path, "Syntax validation failed", None, error)
        
        # Generate diff
        diff = self._generate_diff(original_content, new_content, os.path.basename(file_path))
        
        # Write the file
        success, error = await self._write_file(file_path, new_content)
        if not success:
            return CodeModificationResult(False, file_path, "Failed to write file", diff, str(error))
        
        return CodeModificationResult(True, file_path, "Code inserted successfully", diff, None)
    
    async def delete_file(self, file_path: str) -> CodeModificationResult:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            CodeModificationResult: Result of the operation
        """
        # Validate path
        valid, message = self._validate_path(file_path)
        if not valid:
            return CodeModificationResult(False, file_path, "Path validation failed", None, message)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return CodeModificationResult(False, file_path, "File does not exist", None, "Cannot delete non-existent file")
        
        try:
            # Delete the file
            os.remove(file_path)
            return CodeModificationResult(True, file_path, "File deleted successfully", None, None)
        except Exception as e:
            return CodeModificationResult(False, file_path, "Failed to delete file", None, str(e))


class PythonCodeAnalyzer:
    """
    Analyzes Python code for safety and quality.
    Your friendly code quality assurance department! =D
    """
    
    def __init__(self):
        """Initialize the analyzer with default rules."""
        self.logger = logging.getLogger("WITS.Autonomy.PythonCodeAnalyzer")
    
    def check_for_unsafe_operations(self, code: str) -> List[str]:
        """
        Check for potentially unsafe operations in Python code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List[str]: Warnings about potentially unsafe operations
        """
        warnings = []
        
        # Check for common unsafe operations
        unsafe_patterns = [
            (r"os\.system\s*\(", "Direct system command execution"),
            (r"subprocess\.(call|run|Popen)", "Subprocess execution"),
            (r"exec\s*\(", "Dynamic code execution via exec()"),
            (r"eval\s*\(", "Dynamic code evaluation via eval()"),
            (r"__import__\s*\(", "Dynamic module import"),
            (r"open\s*\(.+['\"]w['\"]", "File write operation"),
            (r"shutil\.(rm|rmtree)", "File deletion operation"),
            (r"socket\.", "Socket operations"),
            (r"requests\.", "Network requests"),
            (r"urllib\.", "Network operations"),
        ]
        
        for pattern, warning in unsafe_patterns:
            if re.search(pattern, code):
                warnings.append(f"Warning: {warning} detected")
        
        return warnings
    
    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """
        Analyze the structure of Python code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            tree = ast.parse(code)
            
            # Count functions, classes, etc.
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            analysis = {
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports),
                "function_names": [func.name for func in functions],
                "class_names": [cls.name for cls in classes],
                "imported_modules": []
            }
            
            # Extract imported modules
            for imp in imports:
                if isinstance(imp, ast.Import):
                    for name in imp.names:
                        analysis["imported_modules"].append(name.name)
                elif isinstance(imp, ast.ImportFrom):
                    module = imp.module if imp.module else ''
                    for name in imp.names:
                        analysis["imported_modules"].append(f"{module}.{name.name}")
            
            return analysis
        
        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}
    
    def suggest_improvements(self, code: str) -> List[str]:
        """
        Suggest code improvements.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List[str]: Improvement suggestions
        """
        suggestions = []
        
        # Very basic linting
        if "import *" in code:
            suggestions.append("Avoid using 'import *' as it can pollute the namespace")
        
        if "except:" in code and "except Exception:" not in code:
            suggestions.append("Avoid bare 'except:' clauses; use 'except Exception:' instead")
        
        if "print(" in code and not code.startswith("#!"):
            suggestions.append("Consider using logging instead of print statements in production code")
        
        return suggestions
