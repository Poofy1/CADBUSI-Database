import os
import datetime

def append_audit(output_path, text, new_file=False):
    """
    Append a message to the audit log file locally.
    
    Args:
        output_path (str): Path to the database directory
        text (str): Text message to append to the audit log
        new_file (bool): If True, create a new log file instead of appending
    """
    # Create the path for the log file
    log_file_path = os.path.join(output_path, "audit_log.txt")
    
    # Format timestamp and log entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {text}\n"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Determine write mode based on new_file parameter
    mode = "w" if new_file else "a"
    
    # Write to file (will create if doesn't exist)
    with open(log_file_path, mode) as f:
        f.write(log_entry)