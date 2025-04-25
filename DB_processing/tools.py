
import easyocr
import threading
thread_local = threading.local()
import datetime
import os

def append_audit(database_path, text):
    """
    Append a message to the audit log file.
    
    Args:
        database_path (str): Path to the database directory
        text (str): Text message to append to the audit log
    """
    log_file = os.path.join(database_path, "audit_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Append to file (will create if doesn't exist)
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {text}\n")

def get_reader():
    # Check if this thread already has a reader
    if not hasattr(thread_local, "reader"):
        # If not, create a new reader and store it in the thread-local storage
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader


# configure easyocr reader
reader = easyocr.Reader(['en'])
