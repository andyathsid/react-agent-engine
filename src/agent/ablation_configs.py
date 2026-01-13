"""
Ablation experiment configurations for plant disease identification agent.

This module provides different agent configurations for ablation studies:
1. Full agent: All tools available
2. No detection: No object detection tools (closed_set_leaf_detection, open_set_object_detection)
3. No retrieval: No knowledge retrieval tools (knowledgebase_search, web_search)
4. No tools: Only LLM with no external tools
"""

import os
import io
import uuid
import boto3
from typing import List, Optional
from dataclasses import field
from typing import Dict, List, Optional

from botocore.client import Config as BotoConfig
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware, ModelRetryMiddleware
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from agent.classifier import SCOLDClassifier
from agent.detector import OWLv2Detector, YOLOv11Detector
from agent.prompts import get_system_prompt, get_system_prompt_no_tools, get_system_prompt_no_detection, get_system_prompt_no_retrieval

# Load environment variables
load_dotenv()

# Configuration (same as graph.py)
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

# Initialize Qdrant client and vector store (only needed for retrieval agents)
qdrant_client = QdrantClient(url=Config.QDRANT_URL)
embeddings = GoogleGenerativeAIEmbeddings(
    api_key=Config.GOOGLE_API_KEY,
    model=Config.EMBEDDING_MODEL
)
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

# Type definitions
class State(AgentState):
    image_url: Optional[str] = field(default=None)
    detections: List[Dict] = field(default_factory=list)
    plant_disease_classifications: List[str] = field(default_factory=list)
    visualization_url: Optional[str] = field(default=None)

# Initialize model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

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
    # Initialize R2 client
    s3_client = boto3.client(
        's3',
        endpoint_url=f'https://{Config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=Config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=Config.R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version='s3v4'),
        region_name='auto'
    )
    
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

# Error handling middleware (same as graph.py)
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

# Image tool middleware (same as graph.py)
class ImageToolMiddleware(AgentMiddleware):
    """Middleware to convert image tool responses to HumanMessage format for Gemini."""
    
    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        
        # If result is a Command with image data, convert to HumanMessage
        if hasattr(result, 'update') and 'messages' in result.update:
            messages = result.update['messages']
            new_messages = []
            
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    # Check if this is an image tool response
                    if "detection visualization" in msg.content.lower() or "image uploaded" in msg.content.lower():
                        # Convert to HumanMessage for Gemini
                        new_messages.append(HumanMessage(content=msg.content))
                    else:
                        new_messages.append(msg)
                else:
                    new_messages.append(msg)
            
            result.update['messages'] = new_messages
        
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

# Model fallback middleware
model_fallback_middleware = ModelFallbackMiddleware(
    init_chat_model("gemini-2.5-pro", model_provider="google_genai", temperature=0, thinking_budget=1024),
)

# Tool definitions (subset based on configuration)
# Note: We'll import the full tool definitions from graph.py and selectively use them

# Define tools locally to avoid circular imports
# These are simplified versions that will work with the existing graph structure

@tool
def web_search(query: str) -> str:
    """Search the web for plant disease information."""
    from langchain_tavily import TavilySearch
    search = TavilySearch(max_results=3)
    return search.run(query)

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
    """Search the plant disease knowledge base."""
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_voyageai import VoyageAIRerank
    
    # Build search parameters
    search_kwargs = {
        "k": fetch_k,
        "filter": {
            "doc_type": doc_type,
        }
    }
    
    if plant_name:
        search_kwargs["filter"]["plant_name"] = plant_name
    if disease_name:
        search_kwargs["filter"]["disease_name"] = disease_name
    if product_group:
        search_kwargs["filter"]["product_group"] = product_group
    
    # Perform hybrid search
    results = vector_store.similarity_search_with_score(query, **search_kwargs)
    
    # Apply reranking if enabled and API key is available
    if Config.JINA_API_KEY:
        try:
            reranker = VoyageAIRerank(
                voyageai_api_key=Config.VOYAGE_API_KEY,
                model="rerank-2.5",
                top_n=k
            )
            # Convert results to format expected by reranker
            documents = [doc for doc, _ in results]
            reranked = reranker.compress_documents(documents, query)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in reranked]
        except Exception:
            # Fallback to non-reranked results
            pass
    
    # Return top-k results without reranking
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc, _ in results[:k]]

@tool
async def closed_set_leaf_detection(
    confidence_threshold: float = 0.3,
    runtime: ToolRuntime[State] = None
) -> Command:
    """Detect leaves using YOLOv11 Small model."""
    import io
    from PIL import Image
    import base64
    
    if not runtime or not runtime.state.get("image_url"):
        return Command(update={
            "messages": [ToolMessage("No image available for detection", tool_call_id=runtime.tool_call_id if runtime else "unknown")]
        })
    
    detector = get_yolov11_detector()
    
    # Download image
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(runtime.state["image_url"])
        image_bytes = response.content
    
    # Perform detection
    image = Image.open(io.BytesIO(image_bytes))
    detections = detector.detect(image, confidence_threshold=confidence_threshold)
    
    # Create visualization
    visualization_bytes = detector.visualize(image, detections)
    visualization_url = upload_detection_image_to_r2(visualization_bytes)
    
    return Command(update={
        "detections": detections,
        "visualization_url": visualization_url,
        "messages": [
            ToolMessage(
                f"Detection complete. Found {len(detections)} objects.\nVisualization: {visualization_url}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
async def open_set_object_detection(
    labels: List[str],
    threshold: float = 0.3,
    runtime: ToolRuntime[State] = None
) -> Command:
    """Detect objects using OWLv2 vision-language model."""
    import io
    from PIL import Image
    import base64
    
    if not runtime or not runtime.state.get("image_url"):
        return Command(update={
            "messages": [ToolMessage("No image available for detection", tool_call_id=runtime.tool_call_id if runtime else "unknown")]
        })
    
    detector = get_owlv2_detector()
    
    # Download image
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(runtime.state["image_url"])
        image_bytes = response.content
    
    # Perform detection
    image = Image.open(io.BytesIO(image_bytes))
    detections = detector.detect(image, labels, threshold=threshold)
    
    # Create visualization
    visualization_bytes = detector.visualize(image, detections)
    visualization_url = upload_detection_image_to_r2(visualization_bytes)
    
    return Command(update={
        "detections": detections,
        "visualization_url": visualization_url,
        "messages": [
            ToolMessage(
                f"Open-set detection complete. Found {len(detections)} objects.\nVisualization: {visualization_url}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

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
    """Perform multimodal plant disease identification."""
    import io
    from PIL import Image
    import httpx
    
    if not runtime or not runtime.state.get("image_url"):
        return Command(update={
            "messages": [ToolMessage("No image available for classification", tool_call_id=runtime.tool_call_id if runtime else "unknown")]
        })
    
    classifier = get_scold_classifier()
    
    # Download image
    async with httpx.AsyncClient() as client:
        response = await client.get(runtime.state["image_url"])
        image_bytes = response.content
    
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform classification
    if method == "text-to-image" and query_text:
        result = classifier.classify_text_to_image(image, query_text, top_k=top_k, fetch_k=fetch_k, use_reranker=use_reranker)
    elif method == "image-to-image":
        result = classifier.classify_image_to_image(image, top_k=top_k, fetch_k=fetch_k, use_reranker=use_reranker)
    else:
        return Command(update={
            "messages": [ToolMessage("Invalid method or missing query text", tool_call_id=runtime.tool_call_id)]
        })
    
    # Format summary
    summary = "Plant Disease Identification Results:\n\n"
    
    if method == "text-to-image":
        summary += f"Query: {query_text}\n"
    else:
        summary += "Query: Image similarity search\n"
    
    if label_filter:
        summary += f"Filter: {label_filter}\n"
    if use_reranker:
        summary += f"Reranking: Enabled\n"
    
    summary += f"\nPredicted Label: {result['label']}\n"
    summary += f"Confidence: {result['confidence']:.4f}\n"
    
    summary += "\nLabel Scores:\n"
    for label, score in result['label_scores'].items():
        summary += f"  {label}: {score:.4f}\n"
    
    summary += f"\nTop-{len(result.get('top_k_details', []))} Results:\n"
    for i, detail in enumerate(result.get('top_k_details', []), 1):
        label = detail['label']
        score = detail['score']
        summary += f"  {i}. {label} ({score:.4f})\n"
    
    return Command(update={
        "plant_disease_classifications": result,
        "messages": [
            ToolMessage(
                summary,
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# Define agent configurations
def create_full_agent():
    """Create agent with all tools available."""
    return create_agent(
        model=model,
        tools=[web_search, knowledgebase_search, closed_set_leaf_detection, open_set_object_detection, plant_disease_identification],
        middleware=[get_system_prompt, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
        name="thesis-agent-full",
        state_schema=State
    )

def create_no_detection_agent():
    """Create agent without detection tools (no object detection)."""
    return create_agent(
        model=model,
        tools=[web_search, knowledgebase_search],  # No detection tools
        middleware=[get_system_prompt_no_detection, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
        name="thesis-agent-no-detection",
        state_schema=State
    )

def create_no_retrieval_agent():
    """Create agent without retrieval tools (no knowledge base or web search)."""
    return create_agent(
        model=model,
        tools=[closed_set_leaf_detection, open_set_object_detection, plant_disease_identification],  # No retrieval tools
        middleware=[get_system_prompt_no_retrieval, handle_tool_errors, image_tool_middleware, model_retry_middleware, tool_retry_middleware, model_fallback_middleware],
        name="thesis-agent-no-retrieval",
        state_schema=State
    )

def create_no_tools_agent():
    """Create agent with no tools (just LLM)."""
    return create_agent(
        model=model,
        middleware=[get_system_prompt_no_tools, model_retry_middleware,model_fallback_middleware],
        name="thesis-agent-no-tools",
        state_schema=State
    )

# Export all agent creation functions
__all__ = [
    'create_full_agent',
    'create_no_detection_agent', 
    'create_no_retrieval_agent',
    'create_no_tools_agent',
    'State'
]