# tools/datetime_tool.py
from typing import ClassVar, Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import pytz
from .base_tool import BaseTool

class DateTimeArgs(BaseModel):
    """Arguments for the DateTimeTool."""
    timezone: Optional[str] = Field(None, description="Timezone to use for the datetime (e.g., 'UTC', 'America/New_York', 'Europe/London'). Defaults to UTC if not specified.")
    format: Optional[str] = Field(None, description="Format string for the datetime. Follows Python's strftime format (e.g., '%Y-%m-%d %H:%M:%S'). Defaults to ISO format if not specified.")

class DateTimeResponse(BaseModel):
    """Response from the DateTimeTool."""
    datetime: str = Field(..., description="Current date and time in requested format.")
    iso_datetime: str = Field(..., description="Current date and time in ISO format.")
    timestamp: int = Field(..., description="Unix timestamp (seconds since epoch).")
    timezone: str = Field(..., description="Timezone used for the datetime.")
    error: Optional[str] = Field(None, description="Error message if any.")

class DateTimeTool(BaseTool):
    """
    Tool for getting the current date and time.
    
    This tool provides the current date and time in various formats
    and allows specifying a timezone.
    """
    
    name: ClassVar[str] = "datetime"
    description: ClassVar[str] = "Get the current date and time. Can provide time in different timezones and formats."
    args_schema: ClassVar[Type[BaseModel]] = DateTimeArgs
    
    async def execute(self, args: DateTimeArgs) -> DateTimeResponse:
        """
        Get the current date and time.
        
        Args:
            args: DateTimeArgs containing optional timezone and format
            
        Returns:
            DateTimeResponse: The current date and time information
        """
        # Get the timezone
        timezone_str = args.timezone or "UTC"
        
        try:
            # Get the timezone object
            if timezone_str.lower() == "utc":
                tz = timezone.utc
            else:
                tz = pytz.timezone(timezone_str)
            
            # Get current time in the specified timezone
            current_time = datetime.now(tz)
            
            # Format the datetime
            formatted_time = current_time.isoformat()
            if args.format:
                try:
                    formatted_time = current_time.strftime(args.format)
                except ValueError as e:
                    return DateTimeResponse(
                        datetime=current_time.isoformat(),
                        iso_datetime=current_time.isoformat(),
                        timestamp=int(current_time.timestamp()),
                        timezone=timezone_str,
                        error=f"Invalid format string: {str(e)}"
                    )
            
            # Construct the response
            return DateTimeResponse(
                datetime=formatted_time,
                iso_datetime=current_time.isoformat(),
                timestamp=int(current_time.timestamp()),
                timezone=timezone_str
            )
        
        except pytz.exceptions.UnknownTimeZoneError:
            # Use UTC as fallback for unknown timezone
            current_time = datetime.now(timezone.utc)
            return DateTimeResponse(
                datetime=current_time.isoformat(),
                iso_datetime=current_time.isoformat(),
                timestamp=int(current_time.timestamp()),
                timezone="UTC",
                error=f"Unknown timezone: {timezone_str}. Using UTC instead."
            )
        
        except Exception as e:
            # Get UTC time for error response
            current_time = datetime.now(timezone.utc)
            return DateTimeResponse(
                datetime=current_time.isoformat(),
                iso_datetime=current_time.isoformat(),
                timestamp=int(current_time.timestamp()),
                timezone="UTC",
                error=f"Error: {str(e)}"
            )
