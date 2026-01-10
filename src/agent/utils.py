import io
import base64
from typing import Union
from PIL import Image
import aiohttp


class AsyncImageHandler:
    """
    Utility class for asynchronous image loading from various sources.
    Supports loading images from URLs, base64 strings, bytes, and file paths.
    """
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """
        Load image from base64 string, URL, or bytes
        
        Args:
            image_input: Image source - can be URL, base64 string, bytes, or file path
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif image_input.startswith("http"):
            return await self._load_from_url(image_input)
        elif image_input.startswith("data:image"):
            return self._load_from_base64(image_input)
        else:
            # Assume file path
            return Image.open(image_input).convert("RGB")
    
    async def _load_from_url(self, url: str) -> Image.Image:
        """Load image from HTTP/HTTPS URL"""
        async with self.session.get(url) as response:
            response.raise_for_status()
            image_bytes = await response.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    def _load_from_base64(self, base64_str: str) -> Image.Image:
        """Load image from base64 data URL"""
        # Extract base64 data from data URL
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")