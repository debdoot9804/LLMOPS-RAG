import os
import logging
from datetime import datetime

def get_logger(file_path=__file__):
    """
    Get a basic logger for the given file path.
    
    Args:
        file_path: The path of the file requesting the logger
        
    Returns:
        A configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    log_file = f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.log"
    log_path = os.path.join(logs_dir, log_file)
    
    # Get logger name from file name
    logger_name = os.path.basename(file_path)
    
    # Configure logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
