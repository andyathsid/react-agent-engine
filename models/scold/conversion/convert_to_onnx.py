import torch
import torch.onnx
import numpy as np
from transformers import RobertaTokenizer
from model import LVL
from PIL import Image
import os

def convert_scold_to_onnx(model_path="scold.pt", output_path="scold.onnx", text_sample="A maize leaf with bacterial spot"):
    """
    Convert SCOLD model to ONNX format
    
    Args:
        model_path: Path to the PyTorch model (.pt file)
        output_path: Path to save the ONNX model
        text_sample: Sample text for dynamic axis configuration
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Load model
    print("Loading model...")
    model = LVL()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create dummy inputs for tracing
    batch_size = 1
    seq_length = 77  # Standard RoBERTa sequence length
    
    # Dummy image input (batch_size, 3, 224, 224)
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Dummy text input
    dummy_text = "A sample text for conversion"
    tokens = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
    dummy_input_ids = tokens["input_ids"].to(device)
    dummy_attention_mask = tokens["attention_mask"].to(device)
    
    # Create sample inputs for dynamic axis configuration
    sample_image = torch.randn(batch_size, 3, 224, 224).to(device)
    sample_tokens = tokenizer(text_sample, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
    sample_input_ids = sample_tokens["input_ids"].to(device)
    sample_attention_mask = sample_tokens["attention_mask"].to(device)
    
    # Define dynamic axes for variable batch sizes and sequence lengths
    dynamic_axes = {
        'image_input': {0: 'batch_size'},
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'image_embeddings': {0: 'batch_size'},
        'text_embeddings': {0: 'batch_size'},
    }
    
    # Add sequence length dimension for text inputs
    if sample_input_ids.shape[1] > 1:
        dynamic_axes['input_ids'][1] = 'sequence_length'
        dynamic_axes['attention_mask'][1] = 'sequence_length'
    
    print("Converting model to ONNX...")
    
    # Export the model
    torch.onnx.export(
        model,
        (sample_image, sample_input_ids, sample_attention_mask),
        output_path,
        opset_version=14,  # Use ONNX opset version 14 for better compatibility
        do_constant_folding=True,
        input_names=[
            'image_input',
            'input_ids', 
            'attention_mask'
        ],
        output_names=[
            'image_embeddings',
            'text_embeddings'
        ],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model successfully converted to ONNX format: {output_path}")
    
    # Verify the ONNX model
    print("Verifying ONNX model...")
    try:
        import onnxruntime as ort
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(output_path)
        
        # Test with dummy inputs
        ort_inputs = {
            'image_input': dummy_image.cpu().numpy(),
            'input_ids': dummy_input_ids.cpu().numpy(),
            'attention_mask': dummy_attention_mask.cpu().numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print("ONNX model verification successful!")
        print(f"Image embeddings shape: {ort_outputs[0].shape}")
        print(f"Text embeddings shape: {ort_outputs[1].shape}")
        
        # Compare with PyTorch model
        with torch.no_grad():
            torch_outputs = model(dummy_image, dummy_input_ids, dummy_attention_mask)
            torch_img_emb = torch_outputs[0].cpu().numpy()
            torch_txt_emb = torch_outputs[1].cpu().numpy()
        
        # Check if outputs are close
        img_diff = np.max(np.abs(ort_outputs[0] - torch_img_emb))
        txt_diff = np.max(np.abs(ort_outputs[1] - torch_txt_emb))
        
        print(f"Max difference in image embeddings: {img_diff}")
        print(f"Max difference in text embeddings: {txt_diff}")
        
        if img_diff < 1e-5 and txt_diff < 1e-5:
            print("✓ ONNX model matches PyTorch model!")
        else:
            print("⚠ ONNX model has significant differences from PyTorch model")
            
    except ImportError:
        print("Warning: onnxruntime not installed. Skipping verification.")
    except Exception as e:
        print(f"Error during verification: {e}")
    
    return output_path

if __name__ == "__main__":
    # Convert the model
    onnx_path = convert_scold_to_onnx()
    
    print("\\nConversion complete!")
    print(f"ONNX model: {onnx_path}")