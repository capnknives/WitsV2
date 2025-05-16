# tools/example_tool.py
from pydantic import BaseModel, Field
from typing import Type, Any, Dict, ClassVar
from .base_tool import BaseTool

# 1. Define the Pydantic model for the tool's arguments
class ExampleCalculatorInput(BaseModel):
    expression: str = Field(..., description="A simple arithmetic expression string (e.g., '2 + 2 * 3'). Supports +, -, *, /.")
    # precision: int = Field(default=2, description="Number of decimal places for the result.")

class ExampleCalculatorOutput(BaseModel): # Optional, for structured output
    expression: str
    result: float
    status: str

class ExampleCalculatorTool(BaseTool):
    # 2. Set class variables: name, description, and args_schema
    name: ClassVar[str] = "simple_calculator"
    description: ClassVar[str] = "Evaluates a simple arithmetic string expression. Supports addition, subtraction, multiplication, and division."
    args_schema: ClassVar[Type[BaseModel]] = ExampleCalculatorInput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # 3. Implement the asynchronous execute method
    async def execute(self, args: ExampleCalculatorInput) -> ExampleCalculatorOutput: # Use the specific Pydantic model for args
        print(f"[{self.name}] Executing with expression: '{args.expression}'")
        try:
            # WARNING: eval() is dangerous. For a real calculator, use a math expression parser.
            # This is a simplified example.
            if not all(c in "0123456789.+-*/ ()" for c in args.expression):
                raise ValueError("Expression contains invalid characters.")
            
            # Basic safety: no letters
            if any(char.isalpha() for char in args.expression):
                raise ValueError("Expression cannot contain letters.")

            result_val = eval(args.expression)
            # result_val = round(result_val, args.precision) # If precision was an arg
            
            return ExampleCalculatorOutput(
                expression=args.expression, 
                result=float(result_val), 
                status="success"
            )
        except ZeroDivisionError:
            return ExampleCalculatorOutput(expression=args.expression, result=0.0, status="error: division by zero")
        except Exception as e:
            print(f"[{self.name}_ERROR] Failed to evaluate expression '{args.expression}': {e}")
            return ExampleCalculatorOutput(expression=args.expression, result=0.0, status=f"error: {str(e)}")