import onnxruntime as ort
import numpy as np
from transformers import RobertaTokenizer
from PIL import Image

class SCOLDONNX:
    def __init__(self, model_path="scold.onnx"):
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
    
    def preprocess_image(self, image_path):
        """Preprocess image without torchvision - lighter dependencies"""
        # Open and convert image
        image = Image.open(image_path).convert("RGB")
        
        # Resize image using PIL
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Convert HWC to CHW format (PyTorch style)
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # Add batch dimension and convert to numpy
        image_tensor = np.expand_dims(image_array, axis=0)
        
        return image_tensor
    
    def predict(self, image_path, text):
        # Preprocess image without torchvision
        image_input = self.preprocess_image(image_path)
        
        # Tokenize text
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        
        # Prepare inputs
        inputs = {
            'image_input': image_input,
            'input_ids': tokens['input_ids'].numpy(),
            'attention_mask': tokens['attention_mask'].numpy()
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Calculate similarity
        image_embeddings = outputs[0]
        text_embeddings = outputs[1]
        similarity = np.matmul(image_embeddings, text_embeddings.T).squeeze()
        
        return similarity.item()

# Usage example
if __name__ == "__main__":
    model = SCOLDONNX()
    similarity = model.predict("applescab.jpg", "An apple leaf with dark brown spot")
    print(f"Similarity score: {similarity}")
