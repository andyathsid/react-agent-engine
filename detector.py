import asyncio
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from typing import List, Tuple, Union, Optional
import io
import base64
import aiohttp
import requests

class AsyncImageHandler:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif image_input.startswith("http"):
            return await self._load_from_url(image_input)
        elif image_input.startswith("data:image"):
            return self._load_from_base64(image_input)
        else:
            return Image.open(image_input).convert("RGB")
    
    async def _load_from_url(self, url: str) -> Image.Image:
        async with self.session.get(url) as response:
            response.raise_for_status()
            image_bytes = await response.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    def _load_from_base64(self, base64_str: str) -> Image.Image:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

class ONNXOutput:
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes

class OWLv2Detector:
    def __init__(self, model_path: str, processor_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.processor_path = processor_path
        self.session = None
        self.processor = None
        self.device = device
        self._load_model()
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        from transformers import Owlv2Processor
        self.processor = Owlv2Processor.from_pretrained(self.processor_path, use_fast=False, local_files_only=True)
    
    async def predict(self, 
                     image_input: Union[str, bytes], 
                     labels: List[str], 
                     threshold: float = 0.4) -> List[dict]:
        image = await self._load_image(image_input)
        
        inputs = self._preprocess(image, labels)
        
        outputs = self._run_inference(inputs)
        
        results = self._post_process(outputs, image.size, labels, threshold)
        
        return results
    
    def _preprocess(self, image: Image.Image, labels: List[str]):
        texts = [labels]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].numpy(),
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
    
    def _run_inference(self, inputs: dict) -> ONNXOutput:
        outputs = self.session.run(None, inputs)
        
        return ONNXOutput(
            logits=torch.from_numpy(outputs[0]),
            pred_boxes=torch.from_numpy(outputs[1])
        )
    
    def _post_process(self, 
                     outputs: ONNXOutput, 
                     image_size: Tuple[int, int], 
                     labels: List[str],
                     threshold: float) -> List[dict]:
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=[image_size[::-1]],
            threshold=threshold,
        )
        
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        label_indices = results[0]["labels"]
        
        formatted_results = []
        for box, score, label_idx in zip(boxes, scores, label_indices):
            formatted_results.append({
                'box': box.tolist(),
                'score': float(score.item()),
                'label': labels[label_idx]
            })
        
        return formatted_results
    
    def visualize_detections(self, 
                           image_input: Union[str, bytes], 
                           detections: List[dict],
                           output_format: str = 'bytes') -> Union[bytes, str, Image.Image]:
        image = self._load_image_sync(image_input)
        
        draw = ImageDraw.Draw(image)
        for detection in detections:
            box = detection['box']
            score = detection['score']
            label = detection['label']
            
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            
            text = f"{label}: {score:.2f}"
            bbox = draw.textbbox((x1, y1 - 20), text)
            draw.rectangle(bbox, fill='green', outline='green')
            draw.text((x1, y1 - 20), text, fill='white')
        
        if output_format == 'bytes':
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
        elif output_format == 'base64':
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        else:
            return image
    
    def _load_image_sync(self, image_input: Union[str, bytes]) -> Image.Image:
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            if image_input.startswith("data:image"):
                if "," in image_input:
                    image_input = image_input.split(",")[1]
                image_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif image_input.startswith("http"):
                # For HTTP URLs, we need to download the image synchronously
                import requests
                response = requests.get(image_input)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image_input).convert("RGB")
        return None
    
    async def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        async with AsyncImageHandler() as handler:
            return await handler.load_image(image_input)

async def main():
    detector = OWLv2Detector(
        model_path="models/owlv2-onnx/model.onnx",
        processor_path="models/owlv2-onnx",
        device="cpu"
    )
    
    labels = ["a photo of a cat", "a remote control"]
    
    detections = await detector.predict(
        image_input="http://images.cocodataset.org/val2017/000000039769.jpg",
        labels=labels,
        threshold=0.4
    )
    
    for det in detections:
        print(f"{det['label']}: {det['score']:.3f} at {det['box']}")
    
    visualized = detector.visualize_detections(
        image_input="http://images.cocodataset.org/val2017/000000039769.jpg",
        detections=detections,
        output_format='bytes'
    )
    
    with open('output.png', 'wb') as f:
        f.write(visualized)

if __name__ == "__main__":
    asyncio.run(main())