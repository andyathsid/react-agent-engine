# %%
import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

# Set style
sns.set_style('darkgrid')
random.seed(420)

# Define paths
MODEL_DIR = r"D:\Workspace\Repository\thesis\research\object-detection-engine\models\yolov12"
IMAGE_DIR = r"D:\Workspace\Repository\thesis\research\object-detection-engine\data\plantdoc\txt\test\images"
DEFAULT_INPUT_SIZE = 416

print(f"Model directory: {MODEL_DIR}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Default input size: {DEFAULT_INPUT_SIZE}")
print(f"ONNX Runtime version: {ort.__version__}")

# %%
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while preserving aspect ratio
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
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

def preprocess_image(img_path, input_size=416):
    """
    Preprocess image for ONNX model inference
    """
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape
    
    # Apply letterbox resizing
    img_resized, ratio, pad = letterbox(img_rgb, new_shape=(input_size, input_size))
    
    # Normalize to 0-1 range
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch, original_shape, ratio, pad

def postprocess_detections(outputs, original_shape, ratio, pad, conf_threshold=0.25, iou_threshold=0.45):
    """
    Post-process ONNX model outputs to get final detections
    """
    # Extract predictions (assuming output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class])
    predictions = outputs[0][0]  # Remove batch dimension
    
    # Filter by confidence
    confident_detections = predictions[predictions[:, 4] > conf_threshold]
    
    if len(confident_detections) == 0:
        return []
    
    # Scale boxes back to original image size
    boxes = confident_detections[:, :4]
    confidences = confident_detections[:, 4]
    class_ids = confident_detections[:, 5].astype(int)
    
    # Adjust for letterbox padding
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    
    # Scale back to original size
    boxes[:, [0, 2]] /= ratio[0]
    boxes[:, [1, 3]] /= ratio[1]
    
    # Clip boxes to image boundaries
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])
    
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
                'bbox': boxes[i],
                'confidence': confidences[i],
                'class_id': class_ids[i]
            })
    
    return final_detections

print("Preprocessing and postprocessing functions defined successfully!")

# %%
# Load ONNX models
def load_onnx_models():
    """Load all available ONNX models"""
    models = {}
    model_files = ['yolov12n.onnx', 'yolov12s.onnx', 'yolov12m.onnx']
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            try:
                # Create ONNX Runtime session
                session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                models[model_file] = session
                print(f"✓ Loaded {model_file}")
                
                # Print model info
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                output_name = session.get_outputs()[0].name
                output_shape = session.get_outputs()[0].shape
                print(f"  Input: {input_name} {input_shape}")
                print(f"  Output: {output_name} {output_shape}")
                
            except Exception as e:
                print(f"✗ Failed to load {model_file}: {str(e)}")
        else:
            print(f"✗ Model file not found: {model_path}")
    
    return models

# Load models
onnx_models = load_onnx_models()
print(f"\nLoaded {len(onnx_models)} ONNX models successfully!")

# %%
# Load test images
def load_test_images(num_images=16):
    """Load sample test images"""
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    selected_images = image_files[:num_images]
    
    image_paths = []
    for img_file in selected_images:
        img_path = os.path.join(IMAGE_DIR, img_file)
        image_paths.append(img_path)
    
    print(f"Selected {len(image_paths)} test images")
    return image_paths

# Load sample images
test_image_paths = load_test_images(16)
print("Sample images:", [os.path.basename(p) for p in test_image_paths[:5]])

# %%
def run_onnx_inference(session, img_path, conf_threshold=0.55):
    """Run inference on a single image using ONNX model"""
    
    # Preprocess image
    input_tensor, original_shape, ratio, pad = preprocess_image(img_path, DEFAULT_INPUT_SIZE)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Run inference (timing removed)
    outputs = session.run(output_names, {input_name: input_tensor})
    inference_time = 0.0
    
    # Postprocess results
    detections = postprocess_detections(outputs, original_shape, ratio, pad, conf_threshold)
    
    return detections, inference_time

def draw_detections(img_path, detections, class_names=None):
    """Draw bounding boxes on image"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define colors for different classes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        x1, y1, x2, y2 = bbox.astype(int)
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Class {class_id}: {confidence:.2f}"
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
            
        cv2.putText(img_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_rgb

# %% [markdown]
# # Single Model Testing
# 
# Test inference on a single image to verify everything works correctly.

# %%
# Test single image with first available model
if onnx_models and test_image_paths:
    model_name = list(onnx_models.keys())[0]
    session = onnx_models[model_name]
    test_image = test_image_paths[0]
    
    print(f"Testing {model_name} on {os.path.basename(test_image)}")
    
    # Run inference
    detections, inference_time = run_onnx_inference(session, test_image)
    
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Number of detections: {len(detections)}")
    
    # Display results
    if detections:
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            cls = det['class_id']
            print(f"Detection {i+1}: Class {cls}, Confidence {conf:.3f}, BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    
    # Visualize
    result_img = draw_detections(test_image, detections)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.title(f"{model_name} Detection Results\nInference Time: {inference_time:.3f}s, Detections: {len(detections)}")
    plt.axis('off')
    plt.show()
    
else:
    print("No models or test images available!")

# %%
def show_model_comparisons(models, image_paths, num_images=4):
    """Show detection results from different models side by side"""
    
    if not models or not image_paths:
        print("No models or images available for comparison")
        return
    
    model_names = list(models.keys())
    selected_images = image_paths[:num_images]
    
    # Create subplot grid
    rows = len(selected_images)
    cols = len(model_names)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for row, img_path in enumerate(selected_images):
        img_name = os.path.basename(img_path)
        
        for col, model_name in enumerate(model_names):
            session = models[model_name]
            
            try:
                # Run inference
                detections, inference_time = run_onnx_inference(session, img_path)
                
                # Draw detections
                result_img = draw_detections(img_path, detections)
                
                # Display
                axes[row, col].imshow(result_img)
                axes[row, col].set_title(f'{model_name}\n{len(detections)} dets, {inference_time:.3f}s')
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error: {str(e)}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{model_name} - Error')
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show comparisons
if onnx_models and test_image_paths:
    print("Comparing detection results across models...")
    show_model_comparisons(onnx_models, test_image_paths, num_images=3)
else:
    print("No models or images available for comparison")


