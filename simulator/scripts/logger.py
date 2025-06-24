



class Logger:
    def __init__(self):
        pass

    def error(self, message: str):
        print(f"ERROR: {message}")

    def info(self, message: str):
        print(f"INFO: {message}")

    def debug(self, message: str):
        print(f"DEBUG: {message}")

    def progress(self, message: str):
        print(f"PROGRESS: {message}")