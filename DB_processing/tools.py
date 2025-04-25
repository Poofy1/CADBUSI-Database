
import easyocr
import threading
thread_local = threading.local()
import datetime
import os
from storage_adapter import *

def append_audit(database_path, text):
    """
    Append a message to the audit log file.
    
    Args:
        database_path (str): Path to the database directory
        text (str): Text message to append to the audit log
    """
    log_file = os.path.join(database_path, "audit_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {text}\n"
    
    # Create directory if it doesn't exist
    make_dirs(os.path.dirname(log_file))
    
    # Check if file exists and read existing content
    if file_exists(log_file):
        existing_content = read_txt(log_file)
        if existing_content is None:  # Handle case where file exists but is empty or can't be read
            existing_content = ""
        content = existing_content + log_entry
    else:
        content = log_entry
    
    # Save the updated content
    save_data(content, log_file)

def get_reader():
    # Check if this thread already has a reader
    if not hasattr(thread_local, "reader"):
        # If not, create a new reader and store it in the thread-local storage
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader


# configure easyocr reader
reader = easyocr.Reader(['en'])
