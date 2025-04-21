
import easyocr
import threading
thread_local = threading.local()


def get_reader():
    # Check if this thread already has a reader
    if not hasattr(thread_local, "reader"):
        # If not, create a new reader and store it in the thread-local storage
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader


# configure easyocr reader
reader = easyocr.Reader(['en'])
