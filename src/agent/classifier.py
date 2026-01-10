import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
import requests
from typing import List, Dict, Optional, Union, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import RobertaTokenizer
import pandas as pd
from tqdm.auto import tqdm
from agent.utils import AsyncImageHandler


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
        """Process candidate boxes for classification
        
        Returns results with full metadata including image URLs for each detected region.
        """
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
            return {'label': 'unknown', 'confidence': 0.0, 'top_k': [], 'top_k_details': []}
        
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
        
        # Include full metadata for each top-k result
        top_k_details = [
            {
                'label': result['payload']['label'],
                'score': result['score'],
                'metadata': {k: v for k, v in result['payload'].items() if k != 'label'}
            }
            for result in search_results
        ]
        
        return {
            'label': best_label,
            'confidence': confidence,
            'label_scores': label_avg_scores,
            'top_k': list(zip(labels, scores)),
            'top_k_details': top_k_details
        }
    
    def _crop_image(self, image: Image.Image, box: List[float]) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = box
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1 = int(max(0, min(x1, width - 1)))
        x2 = int(max(0, min(x2, width)))
        y1 = int(max(0, min(y1, height - 1)))
        y2 = int(max(0, min(y2, height)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return image.crop((x1, y1, x2, y2))
    
    async def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """Async image loading"""
        async with AsyncImageHandler() as handler:
            return await handler.load_image(image_input)
    
    def _get_fallback_results(self, top_k: int) -> List[dict]:
        raise RuntimeError("Qdrant client not available and no fallback data provided. Please ensure Qdrant is running and the collection is populated.")
    
    def _rerank_with_jina(self, 
                         query: str, 
                         results, 
                         top_n: int = 5,
                         jina_api_key: Optional[str] = None) -> dict:
        """
        Rerank search results using Jina's multimodal reranker API.
        
        Args:
            query: Text query
            results: Qdrant search results with image URLs in payload
            top_n: Number of top results to return
            jina_api_key: Jina API key (optional, will use env var if not provided)
            
        Returns:
            Reranked results with relevance scores
        """
        import os
        api_key = jina_api_key or os.getenv("JINA_API_KEY")
        
        if not api_key:
            print("Warning: JINA_API_KEY not found. Skipping reranking.")
            return {"results": [{"index": i, "relevance_score": 0} for i in range(len(results.points))]}
        
        # Prepare documents for reranking
        documents = []
        for point in results.points:
            doc = {
                "image": point.payload.get("image_url", ""),
                "text": point.payload.get("caption", "")
            }
            documents.append(doc)
        
        # Prepare request payload
        payload = {
            "model": "jina-reranker-m0",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False
        }
        
        # Make API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank", 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Jina Reranker API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return {"results": [{"index": i, "relevance_score": 0} for i in range(len(results.points))]}
    
    async def predict_with_reranking(self,
                                    image_input: Union[str, bytes],
                                    candidate_boxes: Optional[List[dict]] = None,
                                    query_text: Optional[str] = None,
                                    top_k: int = 5,
                                    fetch_k: int = 20,
                                    method: str = "text-to-image",
                                    label_filter: Optional[str] = None,
                                    use_reranker: bool = True,
                                    jina_api_key: Optional[str] = None) -> Dict:
        """
        Enhanced prediction with filtering and reranking support.
        
        Args:
            image_input: Image URL or bytes
            candidate_boxes: Optional list of detection boxes for region-based analysis
            query_text: Text query (required for text-based methods)
            top_k: Number of final results after reranking
            fetch_k: Number of candidates to fetch before reranking
            method: Search modality - "text-to-text", "text-to-image", "image-to-text", "image-to-image"
            label_filter: Optional case-insensitive label filter (class name)
            use_reranker: Whether to apply Jina reranker
            jina_api_key: Optional Jina API key
            
        Returns:
            Enhanced classification results with full metadata
        """
        valid_methods = ["text-to-text", "text-to-image", "image-to-image", "image-to-text"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        
        if method in ["text-to-text", "text-to-image"] and query_text is None:
            raise ValueError(f"query_text is required for {method} classification")
        elif method in ["image-to-image", "image-to-text"] and query_text is not None:
            raise ValueError(f"query_text should not be provided for {method} classification")
        
        # Load image
        image = await self._load_image(image_input)
        
        # Process based on candidate_boxes
        if candidate_boxes:
            result = await self._process_candidate_boxes_with_reranking(
                image, candidate_boxes, query_text, top_k, fetch_k, 
                method, label_filter, use_reranker, jina_api_key
            )
        else:
            result = await self._process_full_image_with_reranking(
                image, query_text, top_k, fetch_k, 
                method, label_filter, use_reranker, jina_api_key
            )
        
        return result
    
    async def _process_full_image_with_reranking(self,
                                                 image: Image.Image,
                                                 query_text: Optional[str],
                                                 top_k: int,
                                                 fetch_k: int,
                                                 method: str,
                                                 label_filter: Optional[str],
                                                 use_reranker: bool,
                                                 jina_api_key: Optional[str]) -> Dict:
        """Process full image with filtering and reranking."""
        
        # Build filter if label_filter is provided
        query_filter = None
        if label_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchText(text=label_filter)  # Case-insensitive
                    )
                ]
            )
        
        # Get initial results (more candidates if reranking)
        limit = fetch_k if use_reranker else top_k
        
        if method == "text-to-text":
            # Text query searching caption text vectors
            text_embedding = await self._encode_text(query_text)
            search_results = await self._search_adaptive_filtered(
                text_embedding, "text", limit, query_filter
            )
            rerank_query = query_text
            
        elif method == "text-to-image":
            # Text query searching image vectors (cross-modal)
            text_embedding = await self._encode_text(query_text)
            search_results = await self._search_adaptive_filtered(
                text_embedding, "text_to_image", limit, query_filter
            )
            rerank_query = query_text
            
        elif method == "image-to-text":
            # Image query searching caption text vectors (cross-modal)
            image_embedding = await self._encode_image(image)
            search_results = await self._search_adaptive_filtered(
                image_embedding, "image_against_text", limit, query_filter
            )
            # For image queries, use caption as rerank query
            rerank_query = f"plant disease visual symptoms"
            
        else:  # image-to-image
            # Image query searching image vectors
            image_embedding = await self._encode_image(image)
            search_results = await self._search_adaptive_filtered(
                image_embedding, "image", limit, query_filter
            )
            rerank_query = f"plant disease visual symptoms"
        
        # Apply reranking if enabled
        if use_reranker and len(search_results) > 0 and rerank_query:
            # Convert search results to Qdrant-like structure
            class SearchResult:
                def __init__(self, points):
                    self.points = points
            
            qdrant_result = SearchResult([
                type('Point', (), {
                    'id': r['id'],
                    'score': r['score'],
                    'payload': r['payload']
                })() for r in search_results
            ])
            
            rerank_response = self._rerank_with_jina(
                rerank_query, 
                qdrant_result, 
                top_n=top_k,
                jina_api_key=jina_api_key
            )
            
            # Map reranked results back
            reranked_results = []
            for rerank_item in rerank_response.get("results", []):
                idx = rerank_item["index"]
                if idx < len(search_results):
                    result = search_results[idx].copy()
                    result['rerank_score'] = rerank_item["relevance_score"]
                    reranked_results.append(result)
            
            search_results = reranked_results[:top_k]
        
        return self._format_results_enhanced(search_results, method)
    
    async def _process_candidate_boxes_with_reranking(self,
                                                      image: Image.Image,
                                                      candidate_boxes: List[dict],
                                                      query_text: Optional[str],
                                                      top_k: int,
                                                      fetch_k: int,
                                                      method: str,
                                                      label_filter: Optional[str],
                                                      use_reranker: bool,
                                                      jina_api_key: Optional[str]) -> Dict:
        """Process candidate boxes with filtering and reranking."""
        results = []
        
        for box in candidate_boxes:
            # Crop image to box
            cropped_image = self._crop_image(image, box['box'])
            
            # Build filter if needed
            query_filter = None
            if label_filter:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="label",
                            match=models.MatchText(text=label_filter)
                        )
                    ]
                )
            
            # Get embeddings and search based on method
            limit = fetch_k if use_reranker else top_k
            
            if method in ["text-to-text", "text-to-image"]:
                text_embedding = await self._encode_text(query_text)
                search_type = "text" if method == "text-to-text" else "text_to_image"
                search_results = await self._search_adaptive_filtered(
                    text_embedding, search_type, limit, query_filter
                )
                rerank_query = query_text
            else:
                image_embedding = await self._encode_image(cropped_image)
                search_type = "image_against_text" if method == "image-to-text" else "image"
                search_results = await self._search_adaptive_filtered(
                    image_embedding, search_type, limit, query_filter
                )
                rerank_query = f"plant disease visual symptoms"
            
            # Apply reranking if enabled
            if use_reranker and len(search_results) > 0:
                class SearchResult:
                    def __init__(self, points):
                        self.points = points
                
                qdrant_result = SearchResult([
                    type('Point', (), {
                        'id': r['id'],
                        'score': r['score'],
                        'payload': r['payload']
                    })() for r in search_results
                ])
                
                rerank_response = self._rerank_with_jina(
                    rerank_query,
                    qdrant_result,
                    top_n=top_k,
                    jina_api_key=jina_api_key
                )
                
                reranked_results = []
                for rerank_item in rerank_response.get("results", []):
                    idx = rerank_item["index"]
                    if idx < len(search_results):
                        result = search_results[idx].copy()
                        result['rerank_score'] = rerank_item["relevance_score"]
                        reranked_results.append(result)
                
                search_results = reranked_results[:top_k]
            
            box_result = {
                'box': box['box'],
                'score': box.get('score', 1.0),
                'classification': self._format_results_enhanced(search_results, method)
            }
            results.append(box_result)
        
        return {'boxes': results}
    
    async def _search_adaptive_filtered(self,
                                       embedding: np.ndarray,
                                       search_type: str,
                                       limit: int,
                                       query_filter: Optional[models.Filter] = None) -> List[dict]:
        """
        Adaptive search with filtering support.
        
        Args:
            embedding: The embedding vector to search with
            search_type: Type of search - "text", "image", "text_to_image", or "image_against_text"
            limit: Number of results to return
            query_filter: Optional Qdrant filter
        
        Returns:
            List of search results
        """
        # Map search types to vector index names
        vector_map = {
            "text": "text",
            "text_to_image": "image",  # Text embedding searching image vectors
            "image": "image",
            "image_against_text": "text"  # Image embedding searching text vectors
        }
        
        vector_name = vector_map.get(search_type, "text")
        
        if not self.qdrant_client:
            return self._get_fallback_results(limit)
        
        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=embedding.tolist(),
                using=vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            return [self._format_point(point) for point in search_result.points]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return self._get_fallback_results(limit)
    
    def _format_results_enhanced(self, search_results: List[dict], method: str) -> dict:
        """Enhanced result formatting with full metadata."""
        if not search_results:
            return {
                'label': 'unknown', 
                'confidence': 0.0, 
                'label_scores': {},
                'top_k': [], 
                'top_k_details': []
            }
        
        labels = [result['payload']['label'] for result in search_results]
        scores = [result.get('rerank_score', result['score']) for result in search_results]
        
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
        
        # Include full metadata for each top-k result
        top_k_details = [
            {
                'label': result['payload']['label'],
                'score': result.get('rerank_score', result['score']),
                'original_score': result['score'],
                'metadata': {k: v for k, v in result['payload'].items() if k != 'label'}
            }
            for result in search_results
        ]
        
        return {
            'label': best_label,
            'confidence': confidence,
            'label_scores': label_avg_scores,
            'top_k': list(zip(labels, scores)),
            'top_k_details': top_k_details
        }
