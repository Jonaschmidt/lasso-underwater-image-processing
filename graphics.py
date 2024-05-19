# TODO: docstrings

from PIL import Image
from typing import Optional

from hypervectors import Hypervector

# filepath
# image
# bitstream
# label
# description (optional)

class Graphic:
    def __init__(self, file_path: str, label: Optional[str]=None, description: Optional[str]=None):
        self.file_path = file_path
        try:
            self.image = Image.open(file_path)  # load the image using PIL
        except (IOError, OSError) as e:
            raise FileNotFoundError(f"Error opening image at {file_path}: {e}")
        
        self.label = label
        self.description = description
