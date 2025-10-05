import sys
import traceback
from typing import Optional,cast

class CustomException(Exception):
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
        _, _, tb = sys.exc_info()
        self.stack_trace = ''.join(traceback.format_tb(tb)) if tb else 'No stack trace available'

    def __str__(self):
        base_message = super().__str__()
        if self.original_exception:
            return f"{base_message} (Caused by {repr(self.original_exception)})\nStack trace:\n{self.stack_trace}"
        return base_message
    
def raise_custom_exception(message: str, original_exception: Optional[Exception] = None):
    raise CustomException(message, original_exception)                      



