import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
from typing import List, Dict, Optional, Union, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import RobertaTokenizer
import aiohttp
import pandas as pd
from tqdm.auto import tqdm

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
        """Load image from base64 string, URL, or bytes"""
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
        async with self.session.get(url) as response:
            response.raise_for_status()
            image_bytes = await response.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    def _load_from_base64(self, base64_str: str) -> Image.Image:
        # Extract base64 data from data URL
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

class SCOLDClassifier:
    def __init__(self, 
                 model_path: str,
                 tokenizer_name: str = "roberta-base",
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "leaf_disease_collection"):
        
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Load ONNX model
        self.session = None
        self.tokenizer = None
        self.qdrant_client = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and tokenizer"""
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant at {self.qdrant_url}: {e}")
            self.qdrant_client = None
    
    def setup_collection(self):
        """Setup Qdrant collection with text and image vector configs"""
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized. Check connection.")
        
        # Delete existing collection if it exists
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        
        # Create collection with both text and image vectors
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
            }
        )
        print(f"Created collection: {self.collection_name}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using ONNX model"""
        # Preprocess text
        text_inputs = self._preprocess_text(text)
        
        # Run inference with dummy image input
        outputs = self.session.run(None, {
            'image_input': np.zeros((1, 3, 224, 224), dtype=np.float32),  # Dummy image
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask']
        })
        
        text_emb = outputs[1]  # Text embeddings are the second output
        
        # Normalize
        text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
        return text_emb[0]
    
    def encode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Encode image from bytes using ONNX model"""
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self._preprocess_image(image)
        
        # Run inference with dummy text input
        outputs = self.session.run(None, {
            'image_input': image_tensor,
            'input_ids': np.zeros((1, 77), dtype=np.int64),  # Dummy text
            'attention_mask': np.zeros((1, 77), dtype=np.int64)  # Dummy attention
        })
        
        image_emb = outputs[0]  # Image embeddings are the first output
        
        # Normalize
        image_emb = image_emb / np.linalg.norm(image_emb, axis=-1, keepdims=True)
        return image_emb[0]
    
    def ingest_gallery(self,
                      data: pd.DataFrame,
                      batch_size: int = 10) -> Dict[str, Any]:
        """
        Ingest gallery data into Qdrant collection
        
        Args:
            data: DataFrame with mandatory 'label' column and at least one of 'image' or 'caption' columns
            batch_size: Batch size for insertion
        
        Returns:
            Dictionary with ingestion statistics
        
        Raises:
            ValueError: If 'label' column is missing or if neither 'image' nor 'caption' columns are present
        """
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized. Check connection.")
        
        df = data.copy()
        
        # Validate mandatory 'label' column
        if 'label' not in df.columns:
            raise ValueError("Mandatory 'label' column is missing from the input data")
        
        # Validate that at least one of 'image' or 'caption' columns exists
        optional_columns = ['image', 'caption']
        available_optional = [col for col in optional_columns if col in df.columns]
        if not available_optional:
            raise ValueError(f"At least one of {optional_columns} must be present in the input data")
        
        # Setup collection
        self.setup_collection()
        
        # Ingest data in batches
        total_points = 0
        successful_batches = 0
        failed_batches = 0
        
        for i in tqdm(range(0, len(df), batch_size), desc="Ingesting batches"):
            batch_df = df.iloc[i:i+batch_size]
            batch_points = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Validate row structure
                    if 'label' not in row:
                        print(f"Skipping row {idx}: missing 'label' key")
                        continue
                        
                    if 'image' not in row and 'caption' not in row:
                        print(f"Skipping row {idx}: missing both 'image' and 'caption' keys")
                        continue
                    
                    # Process valid row
                    caption_text = str(row['caption']) if 'caption' in row else ""
                    
                    # Encode text if caption exists
                    text_vec = self.encode_text(caption_text) if 'caption' in row else None
                    
                    # Encode image if image exists
                    img_vec = self.encode_image_from_bytes(row['image']['bytes']) if 'image' in row else None
                    
                    # Create point with available vectors
                    vector_dict = {}
                    payload_dict = {
                        "label": row['label'],
                        "source_id": idx,
                    }
                    
                    if text_vec is not None:
                        vector_dict["text"] = text_vec.tolist()
                        payload_dict["caption"] = caption_text
                        
                    if img_vec is not None:
                        vector_dict["image"] = img_vec.tolist()
                    
                    # Add other columns to payload
                    for k, v in row.items():
                        if k not in ['image', 'caption', 'label']:
                            payload_dict[k] = v
                    
                    batch_points.append(models.PointStruct(
                        id=idx,
                        vector=vector_dict,
                        payload=payload_dict
                    ))
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
            
            # Insert batch into Qdrant
            if batch_points:
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                    successful_batches += 1
                    total_points += len(batch_points)
                    
                except Exception as e:
                    failed_batches += 1
                    print(f"Batch {i//batch_size + 1} error: {e}")
                    continue
        
        # Return statistics
        stats = {
            "total_points": total_points,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "total_batches": successful_batches + failed_batches,
            "collection_name": self.collection_name
        }
        
        print(f"Ingestion completed: {total_points} points inserted")
        return stats
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        
        image_array = np.transpose(image_array, (2, 0, 1))
        
        image_tensor = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        return image_tensor
    
    def _preprocess_text(self, text: str) -> Dict[str, np.ndarray]:
        """Preprocess text for ONNX model"""
        # Tokenize text
        tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        )
        
        return {
            'input_ids': tokens['input_ids'].numpy(),
            'attention_mask': tokens['attention_mask'].numpy()
        }
    
    async def predict(self,
                    image_input: Union[str, bytes],
                    candidate_boxes: Optional[List[dict]] = None,
                    query_text: Optional[str] = None,
                    top_k: int = 5,
                    method: str = "text-to-image") -> Dict:
        
        valid_methods = ["text-to-image", "image-to-image", "image-to-text"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        
        if method == "text-to-image" and query_text is None:
            raise ValueError("query_text is required for text-to-image classification")
        elif method in ["image-to-image", "image-to-text"] and query_text is not None:
            raise ValueError(f"query_text should not be provided for {method} classification")
        
        # Load image
        image = await self._load_image(image_input)
        
        # Process based on candidate_boxes
        if candidate_boxes:
            result = await self._process_candidate_boxes(
                image, candidate_boxes, query_text, top_k, method
            )
        else:
            result = await self._process_full_image(
                image, query_text, top_k, method
            )
        
        return result
    
    async def _process_full_image(self, image: Image.Image, 
                                 query_text: Optional[str],
                                 top_k: int,
                                 method: str) -> Dict:
        if method == "text-to-image":
            if query_text is None:
                raise ValueError("query_text is required for text-to-image classification")
            
            text_embedding = await self._encode_text(query_text)
            search_results = await self._search_adaptive(text_embedding, "text", top_k)
        elif method == "image-to-text":
            image_embedding = await self._encode_image(image)
            search_results = await self._search_adaptive(image_embedding, "image_against_text", top_k)
        else:
            image_embedding = await self._encode_image(image)
            search_results = await self._search_adaptive(image_embedding, "image", top_k)
        
        return self._format_results(search_results, method)
    
    async def _process_candidate_boxes(self, image: Image.Image,
                                     candidate_boxes: List[dict],
                                     query_text: Optional[str],
                                     top_k: int,
                                     method: str) -> Dict:
        """Process candidate boxes for classification"""
        results = []
        
        for box in candidate_boxes:
            # Crop image to box
            cropped_image = self._crop_image(image, box['box'])
            
            if method == "text-to-image":
                if query_text is None:
                    raise ValueError("query_text is required for text-to-image classification")
                
                # Get text embedding and search
                text_embedding = await self._encode_text(query_text)
                search_results = await self._search_adaptive(text_embedding, "text", top_k)
            else:
                # Get image embedding and search
                image_embedding = await self._encode_image(cropped_image)
                search_results = await self._search_adaptive(image_embedding, "image", top_k)
            
            box_result = {
                'box': box['box'],
                'score': box.get('score', 1.0),
                'classification': self._format_results(search_results, method)
            }
            results.append(box_result)
        
        return {'boxes': results}
    
    async def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image using ONNX model"""
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {
            'image_input': image_tensor,
            'input_ids': np.zeros((1, 77), dtype=np.int64),  # Dummy input for text
            'attention_mask': np.zeros((1, 77), dtype=np.int64)  # Dummy input for attention
        })
        
        image_embedding = outputs[0]
        
        # Normalize
        image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=-1, keepdims=True)
        return image_embedding[0]
    
    async def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using ONNX model"""
        # Preprocess text
        text_inputs = self._preprocess_text(text)
        
        # Run inference
        outputs = self.session.run(None, {
            'image_input': np.zeros((1, 3, 224, 224), dtype=np.float32),  # Dummy input for image
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask']
        })
        
        text_embedding = outputs[1]
        
        # Normalize
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=-1, keepdims=True)
        return text_embedding[0]
    
    async def _search_adaptive(self,
                              embedding: np.ndarray,
                              search_type: str,
                              top_k: int) -> List[dict]:
        """
        Adaptive search function that can handle different search types
        
        Args:
            embedding: The embedding vector to search with
            search_type: Type of search - "text", "image", or "image_against_text"
            top_k: Number of results to return
        
        Returns:
            List of search results
        """
        # Map search types to vector index names
        vector_map = {
            "text": "text",
            "image": "image",
            "image_against_text": "text"
        }
        
        vector_name = vector_map.get(search_type, "text")
        
        if not self.qdrant_client:
            return self._get_fallback_results(top_k)
        
        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=embedding.tolist(),
                using=vector_name,
                limit=top_k,
                with_payload=True
            )
            return [self._format_point(point) for point in search_result.points]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return self._get_fallback_results(top_k)
    
    
    def _format_point(self, point) -> dict:
        """Format Qdrant point result"""
        return {
            'id': point.id,
            'score': point.score,
            'payload': point.payload
        }
    
    def _format_results(self, search_results: List[dict], method: str) -> dict:
        if not search_results:
            return {'label': 'unknown', 'confidence': 0.0, 'top_k': []}
        
        labels = [result['payload']['label'] for result in search_results]
        scores = [result['score'] for result in search_results]
        
        label_votes = {}
        label_score_lists = {}
        
        for label, score in zip(labels, scores):
            if label not in label_votes:
                label_votes[label] = 0
                label_score_lists[label] = []
            label_votes[label] += score
            label_score_lists[label].append(score)
        
        best_label = max(label_votes.items(), key=lambda x: x[1])[0]
        confidence = float(np.mean(label_score_lists[best_label]))
        
        label_avg_scores = {
            label: float(np.mean(scores_list)) 
            for label, scores_list in label_score_lists.items()
        }
        
        return {
            'label': best_label,
            'confidence': confidence,
            'label_scores': label_avg_scores,
            'top_k': list(zip(labels, scores))
        }
    
    def _crop_image(self, image: Image.Image, box: List[float]) -> Image.Image:
        """Crop image to bounding box"""
        x1, y1, x2, y2 = map(int, box)
        return image.crop((x1, y1, x2, y2))
    
    async def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """Async image loading"""
        async with AsyncImageHandler() as handler:
            return await handler.load_image(image_input)
    
    def _get_fallback_results(self, top_k: int) -> List[dict]:
        raise RuntimeError("Qdrant client not available and no fallback data provided. Please ensure Qdrant is running and the collection is populated.")


