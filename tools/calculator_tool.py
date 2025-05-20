# tools/calculator_tool.py
from typing import ClassVar, Type, Optional, Union
from pydantic import BaseModel, Field
import re
import ast
import math
from .base_tool import BaseTool

class CalculatorArgs(BaseModel):
    """Arguments for the CalculatorTool."""
    expression: str = Field(..., description="Mathematical expression to evaluate. Can use basic operators (+, -, *, /, **, %), functions (sin, cos, tan, sqrt, log, etc.), and constants (pi, e).")

class CalculatorResponse(BaseModel):
    """Response from the CalculatorTool."""
    result: Optional[Union[int, float]] = Field(None, description="Result of the expression evaluation.")
    expression: str = Field(..., description="The original expression that was evaluated.")
    error: Optional[str] = Field(None, description="Error message if evaluation failed.")

class CalculatorTool(BaseTool):
    """
    Your friendly neighborhood math wizard! \\o/
    
    I safely evaluate expressions so you don't have to math by hand!
    Basic arithmetic? Ez pz! ^_^
    Scientific functions? No problemo! =D
    Complex calculations? Hold my coffee! O.o
    
    Just don't try to divide by zero... I'm looking at you >.>
    """
    
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = "Math magician at your service! \\o/ Let me calculate stuff for you - basic ops (+, -, *, /, **, %), cool functions (sin, cos, tan, sqrt, log), and some tasty constants (pi, e)"
    args_schema: ClassVar[Type[BaseModel]] = CalculatorArgs
    
    # Our trusted math buddies (they passed the background check! =P)
    _ALLOWED_NAMES = {
        'sin': math.sin,  # For when you need those waves ~
        'cos': math.cos,  # The companion to sin ^_^
        'tan': math.tan,  # The wild child O.o
        'sqrt': math.sqrt,  # Square roots are just math trees! \\o/
        'log': math.log,  # Natural logs are nature's calculator =D
        'log10': math.log10,  # When you need that base 10 goodness
        'exp': math.exp,  # e to the power of AWESOME! ^_^
        'abs': abs,  # No negativity allowed here! =P
        'ceil': math.ceil,  # Rounds up (like my hopes and dreams!)
        'floor': math.floor,  # Rounds down (like my debugging skills x.x)
        'round': round,  # For when you can't decide up or down >.>
        'pi': math.pi,  # Mmm... π! *om nom nom*
        'e': math.e  # Not just for emails anymore! =D
    }
    
    def _sanitize_expression(self, expression: str) -> str:
        """
        Clean and verify the expression is safe for evaluation.
        
        Args:
            expression: The mathematical expression to sanitize
            
        Returns:
            str: The sanitized expression
        """
        # Remove any whitespace
        expression = expression.strip()
        
        # Reject expressions that might be code injection attempts
        if any(keyword in expression for keyword in [
            'import', 'eval', 'exec', 'compile', 'getattr', 'setattr', 
            'globals', 'locals', 'delattr', '__', 'os.', 'sys.'
        ]):
            raise ValueError("Expression contains forbidden keywords")
        
        # Replace function names with math.function
        for name in self._ALLOWED_NAMES:
            if name in ['pi', 'e']:  # These are constants
                continue
            # Replace full word matches of function names
            expression = re.sub(r'\b' + name + r'\b', 'math.' + name, expression)
        
        # Replace constants
        expression = re.sub(r'\bpi\b', 'math.pi', expression)
        expression = re.sub(r'\be\b', 'math.e', expression)
        
        return expression
    
    def _safe_eval(self, sanitized_expression: str) -> Union[int, float]:
        """
        Safely evaluate a sanitized mathematical expression.
        
        Args:
            sanitized_expression: The sanitized expression to evaluate
            
        Returns:
            Union[int, float]: The result of the evaluation
        """
        # Parse the expression into an AST (Abstract Syntax Tree)
        node = ast.parse(sanitized_expression, mode='eval')
        
        # Verify the AST only contains safe nodes
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if not isinstance(subnode.func, ast.Attribute):
                    raise ValueError("Function calls not allowed")
                
                if not isinstance(subnode.func.value, ast.Name) or not subnode.func.value.id == 'math':
                    raise ValueError("Only math module functions are allowed")
                
                if subnode.func.attr not in self._ALLOWED_NAMES:
                    raise ValueError(f"Math function not allowed: {subnode.func.attr}")
            
            elif isinstance(subnode, ast.Name) and subnode.id not in ['math']:
                raise ValueError(f"Variable not allowed: {subnode.id}")
        
        # Compile and evaluate the AST
        code = compile(node, "<string>", 'eval')
        return eval(code, {"__builtins__": {}}, {"math": math})
    
    async def execute(self, args: CalculatorArgs) -> CalculatorResponse:
        """
        Evaluate a mathematical expression.
        
        Args:
            args: CalculatorArgs containing the expression to evaluate
            
        Returns:
            CalculatorResponse: The evaluation result or error
        """
        expression = args.expression
        
        try:
            # Check if expression is empty
            if not expression.strip():
                return CalculatorResponse(
                    result=None,
                    expression=expression,
                    error="Expression cannot be empty"
                )
            
            # Try to sanitize and evaluate
            sanitized_expr = self._sanitize_expression(expression)
            result = self._safe_eval(sanitized_expr)
            
            return CalculatorResponse(
                result=result,
                expression=expression,
                error=None  # All good in the math hood! ^_^
            )
        
        except ValueError as ve:
            # Validation error
            return CalculatorResponse(
                result=None,
                expression=expression,
                error=f"Validation error: {str(ve)}"
            )
        
        except SyntaxError as se:
            # Syntax error in the expression
            return CalculatorResponse(
                result=None,
                expression=expression,
                error=f"Syntax error: {str(se)}"
            )
        
        except ZeroDivisionError:
            # Division by zero
            return CalculatorResponse(
                result=None,
                expression=expression,
                error="Division by zero"
            )
        
        except Exception as e:
            # Other errors
            return CalculatorResponse(
                result=None,
                expression=expression,
                error=f"Error: {str(e)}"
            )
