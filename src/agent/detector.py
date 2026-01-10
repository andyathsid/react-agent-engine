import asyncio
import numpy as np
import onnxruntime as ort
import torch
import cv2
import os
from PIL import Image, ImageDraw
from typing import List, Tuple, Union
import io
import base64
import requests
from agent.utils import AsyncImageHandler


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
        
        # Convert to numpy for cv2 drawing (better than PIL)
        img_np = np.array(image)
        
        for detection in detections:
            box = detection['box']
            score = detection['score']
            label = detection['label']
            
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are in correct order (min, max)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Skip invalid boxes
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            
            # Use bright cyan color for better visibility
            color = (0, 255, 255)  # Cyan in BGR
            
            # Draw bounding box with thicker line
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text
            text = f"{label}: {score:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6,  # Larger font
                2
            )
            
            # Draw background rectangle for text
            text_y = max(y1 - 10, text_height + 5)
            cv2.rectangle(
                img_np,
                (x1, text_y - text_height - baseline - 2),
                (x1 + text_width, text_y + baseline),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                img_np,
                text,
                (x1, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text for better contrast
                2
            )
        
        # Convert back to PIL
        image = Image.fromarray(img_np)
        
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

class YOLOv11Detector:
    def __init__(self, model_dir: str = "models/yolov11", device: str = "cpu"):
        # Handle relative paths by making them absolute from current working directory
        if not os.path.isabs(model_dir):
            self.model_dir = os.path.abspath(model_dir)
        else:
            self.model_dir = model_dir
            
        self.device = device
        self.session = None
        print(f"YOLOv11Detector initialized, loading YOLOv11 Small model from: {self.model_dir}")
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv11 Small model"""
        model_file = 'yolo11s.sim.onnx'
        model_path = f"{self.model_dir}/{model_file}"
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        
        try:
            session = ort.InferenceSession(model_path, providers=providers)
            self.session = session
            print(f"✓ Loaded YOLOv11 Small model: {model_file}")
        except Exception as e:
            print(f"✗ Failed to load YOLOv11 Small model {model_file}: {str(e)}")
            raise
    
    def _letterbox(self, img: np.ndarray, new_shape: tuple = (416, 416),
                   color: tuple = (114, 114, 114), auto: bool = True,
                   scaleFill: bool = False, scaleup: bool = True, stride: int = 32) -> tuple:
        """
        Resize and pad image while preserving aspect ratio for YOLOv11 Small
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        return img, ratio, (dw, dh)
    
    def _preprocess(self, image: Image.Image, input_size: tuple = (416, 320)) -> tuple:
        """
        Preprocess image for YOLOv11 Small ONNX model inference
        
        Args:
            image: Input PIL Image
            input_size: Tuple of (height, width) for model input. Default is (416, 320)
        """
        # Convert PIL to numpy array and ensure RGB format
        img_np = np.array(image)
        if len(img_np.shape) == 2:  # Grayscale to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:  # RGBA to RGB
            img_np = img_np[:, :, :3]
        
        original_shape = img_np.shape
        
        # Apply letterbox resizing (auto=False to get exact target size)
        img_resized, ratio, pad = self._letterbox(img_np, new_shape=input_size, auto=False)
        
        # Normalize to 0-1 range
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, original_shape, ratio, pad
    
    def _postprocess(self, outputs: list, original_shape: tuple, ratio: tuple,
                     pad: tuple, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> list:
        """
        Post-process YOLOv11 Small ONNX model outputs to get final detections
        
        Handles both formats:
        - Format 1: [batch, num_detections, 5+num_classes] - from some ONNX exports
        - Format 2: [batch, 5+num_classes, num_detections] - transposed format (more common)
        """
        predictions = outputs[0]  # Get first output
        
        # Check output shape and transpose if needed
        if len(predictions.shape) == 3:
            batch, dim1, dim2 = predictions.shape
            
            # If dim1 is small (like 5 or 6), it's transposed - need to swap
            if dim1 < dim2:
                predictions = predictions.transpose(0, 2, 1)  # Swap to [batch, detections, features]
        
        # Remove batch dimension
        predictions = predictions[0]  # Now shape is [num_detections, features]
        
        # Handle different output formats
        if predictions.shape[1] == 5:
            # Format: [x, y, w, h, conf] - no class prediction (single class model)
            boxes_xywh = predictions[:, :4]
            confidences = predictions[:, 4]
            class_ids = np.zeros(len(predictions), dtype=int)  # All class 0
        elif predictions.shape[1] == 6:
            # Format: [x, y, w, h, conf, class] - simplified format
            boxes_xywh = predictions[:, :4]
            confidences = predictions[:, 4]
            class_ids = predictions[:, 5].astype(int)
        elif predictions.shape[1] > 6:
            # Format: [x, y, w, h, conf, class0_prob, class1_prob, ...] - full format
            boxes_xywh = predictions[:, :4]
            confidences = predictions[:, 4]
            class_probs = predictions[:, 5:]
            class_ids = np.argmax(class_probs, axis=1)
            # Multiply objectness confidence by class probability
            confidences = confidences * np.max(class_probs, axis=1)
        else:
            raise ValueError(f"Unexpected output format with {predictions.shape[1]} features")
        
        # Filter by confidence
        conf_mask = confidences > conf_threshold
        boxes_xywh = boxes_xywh[conf_mask]
        confidences = confidences[conf_mask]
        class_ids = class_ids[conf_mask]
        
        if len(boxes_xywh) == 0:
            return []
        
        # Convert from xywh to xyxy format
        boxes = boxes_xywh.copy()
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1 = x_center - w/2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1 = y_center - h/2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2 = x_center + w/2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2 = y_center + h/2
        
        # Remove letterbox padding (x coordinates)
        boxes[:, [0, 2]] -= pad[0]  # Remove x padding
        # Remove letterbox padding (y coordinates)
        boxes[:, [1, 3]] -= pad[1]  # Remove y padding
        
        # Scale back to original image size
        boxes[:, [0, 2]] /= ratio[0]  # Scale x coordinates
        boxes[:, [1, 3]] /= ratio[1]  # Scale y coordinates
        
        # Clip boxes to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])  # Width
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])  # Height
        
        # Apply NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )
        
        final_detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                final_detections.append({
                    'box': boxes[i].tolist(),
                    'score': float(confidences[i]),
                    'label': str(class_ids[i])  # Return class ID as string label
                })
        
        return final_detections
    
    async def predict(self,
                     image_input: Union[str, bytes],
                     conf_threshold: float = 0.3) -> List[dict]:
        """
        Run YOLOv11 Small inference on an image
        
        Args:
            image_input: Image path, URL, or bytes
            conf_threshold: Confidence threshold for detections
        
        Returns:
            List of detection dictionaries with box, score, and label
        """
        # Load image
        image = await self._load_image(image_input)
        
        # Preprocess image
        input_tensor, original_shape, ratio, pad = self._preprocess(image)
        
        # Get model session
        session = self.session
        
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # Post-process results
        detections = self._postprocess(outputs, original_shape, ratio, pad, conf_threshold)
        
        return detections
    
    def visualize_detections(self, 
                           image_input: Union[str, bytes], 
                           detections: List[dict],
                           class_names: dict = None,
                           output_format: str = 'bytes') -> Union[bytes, str, Image.Image]:
        """
        Visualize YOLOv11 detections on an image
        
        Args:
            image_input: Image path, URL, or bytes
            detections: List of detection dictionaries from predict()
            class_names: Optional dictionary mapping class IDs to names (e.g., {0: 'person', 1: 'car'})
            output_format: 'bytes', 'base64', or 'pil' for return format
        
        Returns:
            Annotated image in the specified format
        """
        # Load image
        image = self._load_image_sync(image_input)
        
        # Convert PIL to numpy array (RGB)
        img_np = np.array(image)
        
        # Define colors for different classes (BGR format for cv2)
        colors = [
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 165, 255),    # Orange
            (255, 255, 0),    # Yellow
            (147, 20, 255),   # Deep Pink
            (0, 255, 0),      # Green
            (255, 0, 0),      # Red
            (255, 144, 30),   # Dodger Blue
        ]
        
        # Draw each detection
        for detection in detections:
            box = detection['box']
            score = detection['score']
            label = detection['label']
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are valid
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Skip invalid boxes
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            
            # Get color based on class ID
            class_id = int(label) if label.isdigit() else 0
            color = colors[class_id % len(colors)]
            
            # Draw bounding box with thicker line
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text
            if class_names and class_id < len(class_names):
                label_text = f"{class_names[class_id]}: {score:.2f}"
            elif class_id == 0:
                label_text = f"leaf: {score:.2f}"
            else:
                label_text = f"Class {label}: {score:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6,  # Larger font
                2
            )
            
            # Draw background rectangle for text
            text_y = max(y1 - 10, text_height + 5)
            cv2.rectangle(
                img_np,
                (x1, text_y - text_height - baseline - 2),
                (x1 + text_width, text_y + baseline),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                img_np,
                label_text,
                (x1, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text for better contrast
                2
            )
        
        # Convert numpy array back to PIL Image
        result_image = Image.fromarray(img_np)
        
        # Return in requested format
        if output_format == 'bytes':
            buffer = io.BytesIO()
            result_image.save(buffer, format='PNG')
            return buffer.getvalue()
        elif output_format == 'base64':
            buffer = io.BytesIO()
            result_image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        else:  # 'pil' or any other value
            return result_image

    def _load_image_sync(self, image_input: Union[str, bytes]) -> Image.Image:
        """Synchronous image loading (same as OWLv2Detector)"""
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            if image_input.startswith("data:image"):
                if "," in image_input:
                    image_input = image_input.split(",")[1]
                image_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif image_input.startswith("http"):
                response = requests.get(image_input)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image_input).convert("RGB")
        return None
    
    async def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """Asynchronous image loading (same as OWLv2Detector)"""
        async with AsyncImageHandler() as handler:
            return await handler.load_image(image_input)


