"""
Plant Disease Identification Agent

This module provides an AI agent for plant disease identification using object detection
and multimodal classification. It supports both full-image and region-based analysis.
"""

from collections import Counter
from dataclasses import field
from typing import Dict, List, Optional
from typing_extensions import Annotated
import operator
import os
import io
import uuid
import requests

import boto3
from botocore.client import Config as BotoConfig
import httpx
from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch
from langsmith import traceable, trace
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from langgraph.types import Command
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware, ModelRetryMiddleware
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware

from agent.classifier import SCOLDClassifier
from agent.detector import OWLv2Detector, YOLOv11Detector
from agent.prompts import get_system_prompt, get_system_prompt_no_tools, get_system_prompt_no_detection, get_system_prompt_no_retrieval

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "knowledgebase_collection")
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    
    # Cloudflare R2 configuration
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET = os.getenv("R2_BUCKET", "thesis-bucket")
    R2_PATH_PREFIX = "detection-results"
    R2_PUBLIC_DOMAIN = "https://thesis-assets.andyathsid.com"
    
    # Jina Reranker configuration
    JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
    JINA_RERANK_MODEL = "jina-reranker-m0"

# Initialize Qdrant client and vector store
qdrant_client = QdrantClient(url=Config.QDRANT_URL)

embeddings = GoogleGenerativeAIEmbeddings(
    api_key=Config.GOOGLE_API_KEY,
    model=Config.EMBEDDING_MODEL
)

# Initialize SPLADE sparse embeddings for hybrid search
sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=Config.QDRANT_COLLECTION_NAME,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

# Initialize R2 client
s3_client = boto3.client(
    's3',
    endpoint_url=f'https://{Config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=Config.R2_ACCESS_KEY_ID,
    aws_secret_access_key=Config.R2_SECRET_ACCESS_KEY,
    config=BotoConfig(signature_version='s3v4'),
    region_name='auto'
)

# Type definitions
class State(AgentState):
    # Store multiple images with their metadata for handling multiple uploads
    images: Annotated[List[Dict], operator.add] = field(default_factory=list)
    # Store detections - can be updated by multiple detection tools
    detections: Annotated[List[Dict], operator.add] = field(default_factory=list)
    # Store classification results - can be updated by multiple classification runs
    plant_disease_classifications: Annotated[List[Dict], operator.add] = field(default_factory=list)
    # Store visualization URLs - can be updated by multiple tools
    visualization_urls: Annotated[List[str], operator.add] = field(default_factory=list)
    # Current active image URL for backward compatibility (points to last uploaded image)
    current_image_url: Optional[str] = field(default=None)

# Initialize models and tools
model = init_chat_model("gemini-3-flash-preview", model_provider="google_genai", temperature=0)

# Global detector instances 
_owlv2_detector = None
_yolov11_detector = None
_scold_classifier = None

def get_owlv2_detector():
    global _owlv2_detector
    if _owlv2_detector is None:
        _owlv2_detector = OWLv2Detector(
            model_path="models/owlv2/owlv2.onnx",
            processor_path="models/owlv2",
            device="cpu"
        )
    return _owlv2_detector

def get_yolov11_detector():
    global _yolov11_detector
    if _yolov11_detector is None:
        _yolov11_detector = YOLOv11Detector(
            model_dir="models/yolov11",
            device="cpu"
        )
    return _yolov11_detector

def get_scold_classifier():
    global _scold_classifier
    if _scold_classifier is None:
        _scold_classifier = SCOLDClassifier(
            model_path="models/scold/scold.onnx",
            collection_name="plantwild_collection"
        )
    return _scold_classifier

# R2 upload utility
def upload_detection_image_to_r2(image_bytes: bytes) -> str:
    """
    Upload a detection visualization image to Cloudflare R2 and return its public URL.
    
    Args:
        image_bytes: Image data as bytes (PNG format)
        
    Returns:
        Public URL of the uploaded image
    """
    # Generate unique filename
    filename = f"detection_{uuid.uuid4()}.png"
    storage_path = f"{Config.R2_PATH_PREFIX}/{filename}"
    
    # Upload to R2
    try:
        s3_client.put_object(
            Bucket=Config.R2_BUCKET,
            Key=storage_path,
            Body=image_bytes,
            ContentType="image/png"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to upload image to R2: {e}")
    
    # Construct public URL
    public_url = f"{Config.R2_PUBLIC_DOMAIN}/{storage_path}"
    
    return public_url

# Error handling middleware
class ErrorHandlingMiddleware(AgentMiddleware):
    """Handle tool execution errors by returning error messages to the agent."""
    
    def wrap_tool_call(self, request, handler):
        """Synchronous error handler."""
        try:
            return handler(request)
        except Exception as e:
            return self._create_error_message(e, request)
    
    async def awrap_tool_call(self, request, handler):
        """Asynchronous error handler."""
        try:
            return await handler(request)
        except Exception as e:
            return self._create_error_message(e, request)
    
    def _create_error_message(self, e: Exception, request):
        """Create appropriate error message based on exception type."""
        error_msg = f"Tool execution error: {str(e)}\n\nPlease check your input and try again."
        if isinstance(e, ValueError):
            error_msg = f"Invalid parameter usage: {str(e)}\n\nPlease adjust your parameters and retry."
        elif isinstance(e, FileNotFoundError):
            error_msg = f"File not found: {str(e)}\n\nPlease check the file path."
        
        return ToolMessage(
            content=error_msg,
            tool_call_id=request.tool_call["id"],
            status="error"
        )

handle_tool_errors = ErrorHandlingMiddleware()

# Image tool middleware
class ImageToolMiddleware(AgentMiddleware):
    """Middleware to convert image tool responses to HumanMessage format for Gemini."""
    
    async def awrap_tool_call(self, request, handler):
        """Intercept tool calls and transform image responses."""
        # Execute the tool
        result = await handler(request)
        
        # Check if this is a Command with an image_url in the update
        if isinstance(result, Command) and result.update:
            image_url = result.update.get("visualization_url")
            if image_url:
                # Extract the text content from the ToolMessage
                text_content = ""
                if "messages" in result.update and result.update["messages"]:
                    tool_msg = result.update["messages"][0]
                    if isinstance(tool_msg, ToolMessage):
                        text_content = tool_msg.content
                
                # Create HumanMessage with both text and image
                image_message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Detection visualization:\n{text_content}"
                        },
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        }
                    ]
                )
                
                # Return Command with HumanMessage instead of ToolMessage
                # Keep visualization_url in state for reference
                return Command(
                    update={
                        **{k: v for k, v in result.update.items() if k != "messages"},
                        "messages": [image_message]
                    }
                )
        
        # For non-image tools, return result as-is
        return result

image_tool_middleware = ImageToolMiddleware()

# Model retry middleware
model_retry_middleware = ModelRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0
)

# Tool retry middleware
tool_retry_middleware = ToolRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0
)

model_fallback_middleware = ModelFallbackMiddleware(
       init_chat_model("gemini-3-pro-preview", model_provider="google_genai", temperature=0, thinking_budget=1024),
)

# Tool definitions
@tool
def web_search(query: str) -> List[Dict]:
    """Search the web for additional plant disease information.
    
    Args:
        query: Search query string
        
    Returns:
        List of search results as retriever documents
    """
    with trace(name="web_search", run_type="retriever", inputs={"query": query}) as rt:
        try:
            search = TavilySearch(
                max_results=3
            )
            web_docs = search.invoke(query)
            
            # Convert to retriever format
            result = [
                {
                    "page_content": d["content"],
                    "type": "Document",
                    "metadata": {
                        "url": d.get("url", ""),
                        "title": d.get("title", ""),
                        "source": "web_search"
                    }
                }
                for d in web_docs['results']
            ]
            rt.end(outputs={"results": result})
            return result
        except Exception as e:
            error_result = [{"page_content": f"Web search error: {str(e)}", "type": "Document", "metadata": {}}]
            rt.end(outputs={"results": error_result})
            return error_result
    
@tool
def knowledgebase_search(
    query: str, 
    doc_type: str = "plant_info",
    plant_name: Optional[str] = None,
    disease_name: Optional[str] = None,
    product_group: Optional[str] = None,
    k: int = 5,
    fetch_k: int = 20
) -> List[Dict]:
    """Search knowledge base for plant disease information or products using hybrid search with Voyage AI reranking.
    
    Args:
        query: Search query describing symptoms, treatment, or product needs
        doc_type: "plant_info" for diseases or "product" for recommendations (default: "plant_info")
        plant_name: Optional case-insensitive plant filter (e.g., "tomato", "grape")
        disease_name: Optional case-insensitive disease filter (e.g., "blight", "mildew", "black rot")
        product_group: Optional product category filter - "fungicide", "herbicide", or "insecticide"
        k: Number of final results to return after reranking (default: 5)
        fetch_k: Number of candidates to fetch before reranking (default: 20)
    
    Returns:
        List of search results as retriever documents with content and metadata
    """
    with trace(
        name="knowledgebase_search", 
        run_type="retriever",
        inputs={
            "query": query,
            "doc_type": doc_type,
            "plant_name": plant_name,
            "disease_name": disease_name,
            "product_group": product_group,
            "k": k,
            "fetch_k": fetch_k
        }
    ) as rt:
        try:
            # Build filter conditions
            must_conditions = [
                models.FieldCondition(
                    key="metadata.doc_type",
                    match=models.MatchValue(value=doc_type),
                )
            ]
            
            # Add plant_name filter for plant_info (token-based matching)
            if doc_type == "plant_info" and plant_name:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.plant_name",
                        match=models.MatchText(text=plant_name),
                    )
                )
            
            # Add disease_name filter for plant_info (token-based matching)
            if doc_type == "plant_info" and disease_name:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.section",
                        match=models.MatchText(text=disease_name),
                    )
                )
            
            # Add product_group filter for product (exact matching)
            if doc_type == "product" and product_group:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.group",
                        match=models.MatchValue(value=product_group),
                    )
                )
            
            # Create filtered retriever for hybrid search
            qdrant_filter = models.Filter(must=must_conditions)
            base_retriever = vector_store.as_retriever(
                search_kwargs={"k": fetch_k, "filter": qdrant_filter}
            )
            
            # Try reranking with Voyage AI, fallback to unranked results if it fails
            try:
                # Setup Voyage AI reranker
                compressor = VoyageAIRerank(
                    model="rerank-2.5",
                    voyageai_api_key=Config.VOYAGE_API_KEY,
                    top_k=k
                )
                
                # Create compression retriever with reranking
                rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                
                # Perform hybrid search with reranking
                results = rerank_retriever.invoke(query)
            except Exception as rerank_error:
                # Fallback: return unranked results if reranker fails (e.g., rate limit)
                print(f"Warning: Reranking failed ({rerank_error}). Returning unranked results.")
                unranked_results = base_retriever.invoke(query)
                results = unranked_results[:k]  # Limit to top k
            
            result = [
                {
                    "page_content": doc.page_content,
                    "type": "Document",
                    "metadata": doc.metadata
                }
                for doc in results
            ]
            rt.end(outputs={"results": result})
            return result
        except Exception as e:
            error_result = [{"error": f"Knowledge base search failed: {str(e)}"}]
            rt.end(outputs={"results": error_result})
            return error_result
    
    
@tool
async def closed_set_leaf_detection(
    confidence_threshold: float = 0.3,
    runtime: ToolRuntime[State] = None
) -> Command:
    """Detect leaves using YOLOv11. Stores results in state.detections and generates visualization.
    
    Args:
        confidence_threshold: Confidence threshold 0.0-1.0 (default: 0.3)
    
    Returns:
        Detection summary with counts, bounding boxes, and visualization URL
    """
    image_url = runtime.state.get("current_image_url")
    if not image_url:
        raise ValueError("No plant image provided")
    
    # Download image
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        response.raise_for_status()
        image_binary = response.content
    
    detector = get_yolov11_detector()
    results = await detector.predict(
        image_input=image_binary,
        conf_threshold=confidence_threshold,
    )
    
    # Generate visualization
    viz_bytes = detector.visualize_detections(
        image_input=image_binary,
        detections=results,
        output_format='bytes'
    )
    
    # Upload visualization to R2
    viz_url = upload_detection_image_to_r2(viz_bytes)
    
    label_counts = Counter(det["label"] for det in results)
    
    summary = "Detection Summary:\n"
    for label, count in label_counts.items():
        summary += f"{label}: {count} detection(s)\n"
        
    summary += "\nDetailed Detections:\n"
    for det in results:
        box = det["box"]
        summary += (
            f"{det['label']}: {det['score']:.3f} at "
            f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n"
        )
    
    return Command(
        update={
            "detections": results,
            "visualization_url": viz_url,
            "messages": [
                ToolMessage(
                    summary,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
    
@tool
async def open_set_object_detection(
    labels: List[str],
    threshold: float = 0.3,
    runtime: ToolRuntime[State] = None
) -> Command:
    """Detect objects using text prompts (OWLv2). Stores results in state.detections and generates visualization.
    
    Args:
        labels: Text descriptions (e.g., ["diseased leaf", "healthy leaf"])
        threshold: Confidence threshold 0.0-1.0 (default: 0.3)
    
    Returns:
        Detection summary with counts, bounding boxes, and visualization URL
    """
    image_url = runtime.state.get("current_image_url")
    if not image_url:
        raise ValueError("No plant image provided")

    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        response.raise_for_status()
        image_binary = response.content

    detector = get_owlv2_detector()
    detections = await detector.predict(
        image_input=image_binary,
        labels=labels,
        threshold=threshold,
    )

    # Generate visualization
    viz_bytes = detector.visualize_detections(
        image_input=image_binary,
        detections=detections,
        output_format='bytes'
    )
    
    # Upload visualization to R2
    viz_url = upload_detection_image_to_r2(viz_bytes)

    label_counts = Counter(det["label"] for det in detections)

    summary = "Detection Summary:\n"
    for label, count in label_counts.items():
        summary += f"{label}: {count} detection(s)\n"

    summary += "\nDetailed Detections:\n"
    for det in detections:
        box = det["box"]
        summary += (
            f"{det['label']}: {det['score']:.3f} at "
            f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n"
        )

    return Command(
        update={
            "detections": detections,
            "visualization_url": viz_url,
            "messages": [
                ToolMessage(
                    summary,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
    
@tool
async def plant_disease_identification(
    query_text: Optional[str] = None,
    top_k: int = 5,
    fetch_k: int = 20,
    method: str = "text-to-image",
    use_detections: bool = True,
    label_filter: Optional[str] = None,
    use_reranker: bool = True,
    runtime: ToolRuntime[State] = None
) -> Command:
    """Identify plant diseases using multimodal retrieval with SCOLD embeddings, filtering, and reranking.
    
    Supports 4 search modalities:
    - "text-to-text": Text query → Caption vectors (semantic text matching)
    - "text-to-image": Text query → Image vectors (cross-modal visual search)
    - "image-to-image": Image → Image vectors (visual similarity)
    - "image-to-text": Image → Caption vectors (cross-modal caption search)
    
    Args:
        query_text: Symptom description (required for text-based methods)
        top_k: Number of final disease candidates to return (default: 5)
        fetch_k: Number of candidates to retrieve before reranking (default: 20)
        method: Search modality - "text-to-text", "text-to-image", "image-to-text", or "image-to-image"
        use_detections: Use state.detections for region-based analysis (default: True)
        label_filter: Optional case-insensitive plant or disease filter (e.g., "apple", "blight", "bell pepper", "black rot")
        use_reranker: Apply Jina multimodal reranker to improve results (default: True)
    
    Returns:
        Classification results with labels, confidence scores, image URLs, and top-k diseases
    """
    image_url = runtime.state.get("current_image_url")
    if not image_url:
        raise ValueError("No plant image provided")

    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        response.raise_for_status()
        image_binary = response.content

    candidate_boxes = None
    if use_detections and runtime.state.get("detections"):
        candidate_boxes = runtime.state["detections"]

    classifier = get_scold_classifier()
    
    # Enhanced prediction with filtering and reranking
    result = await classifier.predict_with_reranking(
        image_input=image_binary,
        candidate_boxes=candidate_boxes,
        query_text=query_text,
        top_k=top_k,
        fetch_k=fetch_k,
        method=method,
        label_filter=label_filter,
        use_reranker=use_reranker,
    )
    
    if candidate_boxes:
        summary = f"Region-Based Classification ({method})\n"
        summary += "=" * 60 + "\n"
        summary += f"Analyzed {len(result['boxes'])} detected regions\n"
        if label_filter:
            summary += f"Filter: {label_filter}\n"
        if use_reranker:
            summary += f"Reranking: Enabled (Jina {Config.JINA_RERANK_MODEL})\n"
        summary += "\n"
        
        for idx, box_result in enumerate(result['boxes'], 1):
            box = box_result['box']
            cls = box_result['classification']
            summary += f"Region {idx} [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]:\n"
            summary += f"  Label: {cls['label']} (confidence: {cls['confidence']:.4f})\n"
            
            # Show top predictions with image URLs if available
            summary += f"  Top predictions:\n"
            for i, detail in enumerate(cls.get('top_k_details', [])[:3], 1):
                label = detail['label']
                score = detail['score']
                img_url = detail['metadata'].get('image_url', 'N/A')
                summary += f"    {i}. {label} ({score:.3f}) - {img_url}\n"
            summary += "\n"
    else:
        summary = f"Full-Image Classification ({method})\n"
        summary += "=" * 60 + "\n"
        
        if method in ["text-to-text", "text-to-image"]:
            summary += f"Query: '{query_text}'\n"
        elif method == "image-to-text":
            summary += "Query: Image features against disease text vectors\n"
        else:
            summary += "Query: Image similarity search\n"
        
        if label_filter:
            summary += f"Filter: {label_filter}\n"
        if use_reranker:
            summary += f"Reranking: Enabled (Jina {Config.JINA_RERANK_MODEL})\n"
        
        summary += f"\nPredicted Label: {result['label']}\n"
        summary += f"Confidence: {result['confidence']:.4f}\n"
        
        summary += "\nLabel Scores:\n"
        for label, score in result['label_scores'].items():
            summary += f"  {label}: {score:.4f}\n"
        
        # Enhanced top-k with full metadata
        summary += f"\nTop-{len(result.get('top_k_details', []))} Results:\n"
        for i, detail in enumerate(result.get('top_k_details', []), 1):
            label = detail['label']
            score = detail['score']
            metadata = detail['metadata']
            plant_name = metadata.get('plant_name', 'unknown')
            img_url = metadata.get('image_url', 'N/A')
            caption = metadata.get('caption', 'No caption')
            
            summary += f"  {i}. {label} ({score:.4f})\n"
            summary += f"     Plant: {plant_name}\n"
            summary += f"     Caption: {caption[:80]}...\n"
            summary += f"     Image: {img_url}\n"
    
    return Command(
        update={
            "plant_disease_classifications": [result],
            "messages": [
                ToolMessage(
                    summary,
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


graph = create_agent(
    model=model,
    tools=[web_search, knowledgebase_search,  closed_set_leaf_detection, open_set_object_detection, plant_disease_identification],
    middleware=[get_system_prompt, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
    name="thesis-agent",
    state_schema=State
)

full_agent = create_agent(
        model=model,
        tools=[web_search, knowledgebase_search, closed_set_leaf_detection, open_set_object_detection, plant_disease_identification],
        middleware=[get_system_prompt, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
        name="thesis-agent-full",
        state_schema=State
    )

no_detection_agent = create_agent(
        model=model,
        tools=[web_search, knowledgebase_search],  # No detection tools
        middleware=[get_system_prompt_no_detection, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
        name="thesis-agent-no-detection",
        state_schema=State
    )

no_retrieval_agent = create_agent(
        model=model,
        tools=[closed_set_leaf_detection, open_set_object_detection, plant_disease_identification],  # No retrieval tools
        middleware=[get_system_prompt_no_retrieval, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
        name="thesis-agent-no-retrieval",
        state_schema=State
    )

no_tools_agent = create_agent(
        model=model,
        middleware=[get_system_prompt_no_tools, model_retry_middleware,model_fallback_middleware],
        name="thesis-agent-no-tools",
        state_schema=State
    )
