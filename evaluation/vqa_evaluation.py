#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, List
import json
import ast
import asyncio
import hashlib
import io
import requests
from PIL import Image
from dotenv import load_dotenv
from contextlib import contextmanager

# Setup paths
load_dotenv()

# Use dedicated evaluation API key if available to avoid rate limits
if os.environ.get("EVALUATION_GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["EVALUATION_GOOGLE_API_KEY"]
 
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))  # Add project root for data module
sys.path.insert(0, str(project_root / "src"))  # Add src for agent module

# LangChain/LangSmith
from langsmith import Client, aevaluate, tracing_context
from langsmith.evaluation import aevaluate_existing
import langchain_core.messages as lc
from langchain_core.tracers.context import tracing_v2_callback_var
from agentevals.trajectory.llm import create_trajectory_llm_as_judge

# RAGAS
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import ContextRelevance
import ragas.messages as r

# DeepEval
from deepeval.models import GeminiModel, GPTModel, OpenRouterModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall, MLLMImage
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics import (
    GEval,
    DAGMetric,
)

# Agent
from agent.graph import agent as agent_graph

# Dataset examples - load from JSON to preserve metadata
import json

# ============================================================================
# Input Validation Helper
# ============================================================================

def validate_and_normalize_inputs(inputs: dict) -> dict:
    """
    Validate and normalize inputs for evaluation.
    
    Handles cases where:
    - image_url is missing or None
    - user_text is missing
    - inputs is None or not a dict
    
    Returns:
        dict with guaranteed keys: 'user_text', 'image_url' (or None)
    """
    if inputs is None:
        return {"user_text": "", "image_url": None}
    
    if not isinstance(inputs, dict):
        return {"user_text": str(inputs), "image_url": None}
    
    # Get user_text with fallback
    user_text = inputs.get("user_text", inputs.get("input", inputs.get("query", "")))
    if not user_text:
        user_text = ""
    
    # Get image_url with fallback
    image_url = inputs.get("image_url", inputs.get("image", None))
    if image_url == "":
        image_url = None
    
    return {
        "user_text": user_text,
        "image_url": image_url
    }

def process_and_resize_image(image_url: str, max_size: int = 1024) -> str:
    """
    Download and resize image if it's too large, returning a local path.
    Caches the result to avoid repeated downloads and processing.
    
    This prevents timeouts in DeepEval/MLLM when handling very large dataset images.
    """
    if not image_url:
        return None
        
    # Create cache directory
    cache_dir = project_root / "data" / "evaluation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename based on URL/path
    url_hash = hashlib.md5(image_url.encode()).hexdigest()
    
    # Try to preserve extension if possible
    file_ext = ".jpg"
    if "." in image_url.split("/")[-1]:
        potential_ext = Path(image_url).suffix.lower()
        if potential_ext in ['.jpg', '.jpeg', '.png', '.webp']:
            file_ext = potential_ext
            
    cache_path = cache_dir / f"{url_hash}{file_ext}"
    
    # Return cached path if it exists
    if cache_path.exists():
        return str(cache_path)
    
    try:
        # Load image
        if image_url.startswith(('http://', 'https://')):
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            # Handle local file
            img = Image.open(image_url)
            
        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
            
        # Resize if any dimension exceeds max_size
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        # Save to cache
        img.save(cache_path, "JPEG", quality=85, optimize=True)
        return str(cache_path)
        
    except Exception as e:
        print(f"Warning: Failed to process image {image_url}: {e}")
        # Return original URL as fallback if it's a remote URL
        return image_url

def iter_dataset(json_path):
    """Stream JSON array items without loading everything into memory."""
    decoder = json.JSONDecoder()
    with open(json_path, "r") as f:
        buffer = ""
        in_array = False
        for chunk in iter(lambda: f.read(65536), ""):
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not in_array:
                    if not buffer:
                        break
                    if buffer[0] == "[":
                        in_array = True
                        buffer = buffer[1:]
                        continue
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield obj
                buffer = buffer[idx:]
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]


def load_dataset_from_json(filename="vqa_dataset.json"):
    """Load dataset examples from JSON file with metadata using streaming."""
    json_path = project_root / "data" / "langsmith" / filename
    
    if not json_path.exists():
        print(f"Warning: Dataset file {json_path} not found.")
        return [], [], []
    
    # Filter by prompt type using streaming
    type1 = []
    type2 = []
    type3 = []
    
    if filename == "ood_dataset.json":
        for ex in iter_dataset(json_path):
            prompt_type = ex.get("metadata", {}).get("prompt_type", "")
            if prompt_type in ["scenario_1_diseased", "healthy_scenario_1",]:
                type1.append(ex)
            elif prompt_type in ["scenario_2_diseased", "healthy_scenario_2",]:
                type2.append(ex)
            elif prompt_type in ["scenario_3_diseased", "healthy_scenario_3"]:
                type3.append(ex)
    else:
        for ex in iter_dataset(json_path):
            prompt_type = ex.get("metadata", {}).get("prompt_type", "")
            if prompt_type in ["vague_symptoms", "healthy_scenario_1",]:
                type1.append(ex)
            elif prompt_type in ["direct_inquiry", "healthy_scenario_2",]:
                type2.append(ex)
            elif prompt_type in ["general_inquiry", "healthy_scenario_3"]:
                type3.append(ex)
        
    return type1, type2, type3

# Load ID dataset by default
TYPE1_DETAILED_EXAMPLES, TYPE2_SPECIES_EXAMPLES, TYPE3_MINIMAL_EXAMPLES = load_dataset_from_json("vqa_dataset.json")
# Load OOD dataset
OOD_TYPE1, OOD_TYPE2, OOD_TYPE3 = load_dataset_from_json("ood_dataset.json")

# ============================================================================
# Configuration
# ============================================================================

# Langsmith Client
ls_client = Client()

# AsyncOpenAI for RAGAS
oai_gemini = AsyncOpenAI(
    api_key=os.environ["GOOGLE_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=3,
)

# RAGAS LLMs
ragas_llm_pro = llm_factory(
    "gemini-2.5-pro",
    provider="openai",
    client=oai_gemini,
    max_tokens=8192,
)

ragas_llm_flash = llm_factory(
    "gemini-2.5-flash",
    provider="openai",
    client=oai_gemini,
    max_tokens=8192,
)

ragas_embeddings = OpenAIEmbeddings(
    client=oai_gemini,
    model="gemini-embedding-001",
)

# DeepEval Models
deepeval_model_pro = GeminiModel(
    model="gemini-2.5-pro",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)

deepeval_model_flash = GeminiModel(
    model="gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)

deepeval_model_openrouter = OpenRouterModel(
    model="openrouter:gemini-2.f-pro",
    temperature=0
)

deepeval_model_gpt = GPTModel(
    model="gpt-4.1",
    temperature=0
)

deepeval_model_gpt4o = GPTModel(
    model="gpt-4.1-mini",
    temperature=0
)


# ============================================================================
# Setup Datasets
# ============================================================================

def get_or_create_dataset(client: Client, name: str, description: str):
    """Get existing dataset or create a new one."""
    for ds in client.list_datasets(dataset_name=name):
        return ds
    return client.create_dataset(dataset_name=name, description=description)


def load_examples_to_dataset(client: Client, dataset_name: str, examples: list, description: str):
    """Load examples into a LangSmith dataset."""
    dataset = get_or_create_dataset(client, dataset_name, description)
    
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    
    if existing_examples:
        print(f"  Dataset '{dataset.name}' already has {len(existing_examples)} examples. Skipping upload.")
    else:
        if not examples:
            print(f"  Dataset '{dataset.name}' is empty and no examples were provided to add. Skipping.")
            return dataset
            
        print(f"  Dataset '{dataset.name}' is empty. Adding {len(examples)} examples...")
        inputs = []
        outputs = []
        metadatas = []
        for ex in examples:
            inputs.append({
                "user_text": ex["inputs"]["user_text"],
                "image_url": ex["inputs"].get("image_url"),
            })
            # Store metadata within outputs for evaluator access
            output_data = ex["outputs"].copy()
            output_data["metadata"] = ex.get("metadata", {})
            outputs.append(output_data)
            # Store metadata in example metadata for LangSmith organization
            metadata = ex.get("metadata", {})
            metadatas.append({
                "example_id": ex.get("id"),
                "class": metadata.get("class", ""),
                "plant": metadata.get("plant", ""),
                "pathogen_type": metadata.get("pathogen_type", ""),
                "prompt_type": metadata.get("prompt_type", ""),
                "filename": metadata.get("filename", "")
            })

        client.create_examples(dataset_id=dataset.id, inputs=inputs, outputs=outputs, metadata=metadatas)
        print(f"  Successfully added {len(inputs)} examples to dataset.")
    
    return dataset


def setup_datasets(dataset_type="vqa"):
    """Setup all evaluation datasets. dataset_type can be 'vqa' or 'ood'."""
    print("=" * 80)
    print(f"Setting up LangSmith datasets for {dataset_type.upper()}...")
    print("=" * 80)
    
    if dataset_type == "ood":
        t1, t2, t3 = OOD_TYPE1, OOD_TYPE2, OOD_TYPE3
        dataset_prefix = "thesis_vqa_ood"
    else:
        t1, t2, t3 = TYPE1_DETAILED_EXAMPLES, TYPE2_SPECIES_EXAMPLES, TYPE3_MINIMAL_EXAMPLES
        dataset_prefix = "thesis_vqa_id"

    datasets = {}
    
    datasets['type1'] = load_examples_to_dataset(
        ls_client,
        f"{dataset_prefix}_scenario1",
        t1,
        f"Scenario 1 ({dataset_type.upper()}): Spesies dan Gejala"
    )
    
    datasets['type2'] = load_examples_to_dataset(
        ls_client,
        f"{dataset_prefix}_scenario2",
        t2,
        f"Scenario 2 ({dataset_type.upper()}): Hanya Spesies"
    )
    
    datasets['type3'] = load_examples_to_dataset(
        ls_client,
        f"{dataset_prefix}_scenario3",
        t3,
        f"Scenario 3 ({dataset_type.upper()}): Minimal"
    )
    
    print(f"\n{dataset_type.upper()} Dataset Summary:")
    print(f"  - Scenario 1 (Detailed): {len(t1)} examples")
    print(f"  - Scenario 2 (Species): {len(t2)} examples")
    print(f"  - Scenario 3 (Minimal): {len(t3)} examples")
    print("=" * 80 + "\n")
    
    return datasets

# ============================================================================
# Individual Evaluator Functions
# Each metric is now separated into its own evaluator function
# ============================================================================



# 1. RAGAS Context Relevance Evaluator
async def context_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate context relevance using RAGAS with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        user_text = inputs.get("user_text", "")
        image_url = inputs.get("image_url")
        
        # Create multimodal input for RAGAS evaluation
        if image_url:
            user_question = f"[Gambar disediakan - agen akan melakukan observasi visual] {user_text}"
        else:
            user_question = user_text
        
        trace_messages = outputs.get("trace_messages", {})
        retrieved_context = _extract_retrieval_context_from_trace(trace_messages)
        
        if not retrieved_context or len(retrieved_context) == 0:
            return []
        
        try:
            result = await ragas_context_relevance.ascore(
                user_input=user_question,
                retrieved_contexts=retrieved_context
            )
            return [{"key": "context_relevance", "score": result.value}]
        except Exception as e:
            return [{"key": "context_relevance", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 2. Plant Agent Faithfulness Evaluator (DeepEval GEval)
async def faithfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate faithfulness using custom DeepEval GEval metric with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        # Create test case with multimodal support
        test_case = create_deepeval_test_case(inputs, outputs, reference_outputs)
        
        # Extract retrieval context
        trace_messages = outputs.get("trace_messages", {})
        retrieved_context = _extract_retrieval_context_from_trace(trace_messages)
        
        if not retrieved_context or len(retrieved_context) == 0:
            return [{"key": "faithfulness", "score": 0.0, "comment": f"No retrieval context available: {trace_messages}"}]
        
        final_answer = outputs.get("final_answer", "")
        
        # Handle case where no image is provided
        image_url = inputs.get("image_url")
        if not image_url:
            # Use text-only test case
            faithfulness_test_case = LLMTestCase(
                input=test_case.input,
                actual_output=final_answer,
                retrieval_context=retrieved_context,
                tools_called=test_case.tools_called
            )
        else:
            # Use multimodal test case (already created in test_case.input)
            faithfulness_test_case = test_case
            # Override actual_output and retrieval_context to ensure they're correct
            faithfulness_test_case.actual_output = final_answer
            faithfulness_test_case.retrieval_context = retrieved_context
        
        try:
            score = measure_with_fallback(deepeval_plant_faithfulness, faithfulness_test_case)
            return [{
                "key": "faithfulness",
                "score": score,
                "comment": deepeval_plant_faithfulness.reason
            }]
        except Exception as e:
            return [{"key": "faithfulness", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 3. Trajectory Accuracy - With Reference Evaluator
async def trajectory_with_ref_evaluator_func(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate trajectory accuracy with reference using LangChain AgentEval with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        user_text = inputs.get("user_text", "")
        image_url = inputs.get("image_url")
        
        # Create multimodal input for trajectory evaluation
        if image_url:
            user_question = f"[Gambar disediakan - agen akan melakukan observasi visual] {user_text}"
        else:
            user_question = user_text
        
        trace_messages = outputs.get("trace_messages", {})
        reference_tool_calls = reference_outputs.get("reference_tool_calls", [])
        
        if not reference_tool_calls:
            return [{"key": "trajectory_accuracy", "score": 0.0, "comment": "No reference trajectory provided"}]
        
        try:
            # Create the trajectory evaluator with custom prompt
            trajectory_evaluator = create_trajectory_llm_as_judge(
                prompt=TRAJECTORY_WITH_REF_PROMPT,
                model="google_genai:gemini-2.5-pro",
                continuous=True,
            )
            
            traj_result = trajectory_evaluator(
                reference_tool_calls=reference_tool_calls,
                input=user_question,
                outputs=trace_messages,
            )
            return [traj_result]
        except Exception as e:
            return [{"key": "trajectory_accuracy", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 4. Trajectory Accuracy - Without Reference Evaluator
async def trajectory_without_ref_evaluator_func(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate trajectory accuracy without reference using LangChain AgentEval with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        user_text = inputs.get("user_text", "")
        image_url = inputs.get("image_url")
        
        # Create multimodal input for trajectory evaluation
        if image_url:
            user_question = f"[Gambar disediakan - agen akan melakukan observasi visual] {user_text}"
        else:
            user_question = user_text
        
        trace_messages = outputs.get("trace_messages", {})
        
        try:
            # Create the trajectory evaluator with custom prompt
            trajectory_evaluator = create_trajectory_llm_as_judge(
                prompt=TRAJECTORY_WITHOUT_REF_PROMPT,
                model="google_genai:gemini-2.5-pro",
                continuous=True,
            )
            
            traj_result = trajectory_evaluator(
                input=user_question,
                outputs=trace_messages
            )
            return [{"key": "trajectory_accuracy", "score": traj_result.get("score", 0.0)}]
        except Exception as e:
            return [{"key": "trajectory_accuracy", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 5. Disease Accuracy Evaluator (DeepEval DAG)
async def disease_accuracy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate disease accuracy using DeepEval DAG metric with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        # Create test case with multimodal support
        test_case = create_deepeval_test_case(inputs, outputs, reference_outputs)
        
        final_answer = outputs.get("final_answer", "")
        
        # Get metadata from reference_outputs
        if isinstance(reference_outputs, str):
            try:
                reference_outputs = json.loads(reference_outputs)
            except Exception:
                try:
                    reference_outputs = ast.literal_eval(reference_outputs)
                except Exception:
                    reference_outputs = {}
        if not isinstance(reference_outputs, dict):
            reference_outputs = {}
        
        metadata = reference_outputs.get("metadata", {})
        
        # Extract expected disease and pathogen type
        expected_pathogen = metadata.get("pathogen_type", "")
        expected_disease = metadata.get("class", "")
        reference_answer = reference_outputs.get("reference_answer", "")
        
        # Handle healthy plant cases
        is_healthy_case = (
            expected_disease.endswith("leaf") or 
            expected_disease.endswith("healthy") or
            expected_pathogen == "healthy"
        )
        
        try:
            # Handle case where no image is provided
            image_url = inputs.get("image_url")
            if not image_url:
                # Use text-only test case
                if is_healthy_case:
                    disease_test_case = LLMTestCase(
                        input=test_case.input,
                        actual_output=final_answer,
                        expected_output=f"Jenis Patogen: Tidak ada (tanaman sehat)\nPenyakit: TANAMAN SEHAT ({expected_disease})\nReferensi: {reference_answer}"
                    )
                else:
                    disease_test_case = LLMTestCase(
                        input=test_case.input,
                        actual_output=final_answer,
                        expected_output=f"Jenis Patogen: {expected_pathogen}\nPenyakit: {expected_disease}\nReferensi: {reference_answer}"
                    )
            else:
                # Use multimodal test case (already created in test_case.input)
                # Override expected_output and actual_output to ensure they're correct
                test_case.actual_output = final_answer
                if is_healthy_case:
                    test_case.expected_output = f"Jenis Patogen: Tidak ada (tanaman sehat)\nPenyakit: TANAMAN SEHAT ({expected_disease})\nReferensi: {reference_answer}"
                else:
                    test_case.expected_output = f"Jenis Patogen: {expected_pathogen}\nPenyakit: {expected_disease}\nReferensi: {reference_answer}"
                disease_test_case = test_case
            
            # Create DAG metric
            # Score mapping (0-10 ints; metric.score is 0-1 normalized)
            SCORE_1_0  = 10
            SCORE_0_75 = 8
            SCORE_0_5  = 5
            SCORE_0_25 = 3

            # Build DAG with healthy plant check as root node
            # Primary disease check comes after healthy plant check
            # Then pathogen type check, disease partial match, and abstention check
            
            abstention_appropriate = BinaryJudgementNode(
                criteria="""
Berdasarkan output agen dan output yang diharapkan:
Apakah agen sudah menunda memberikan jawaban (menyatakan ketidakpastian/meminta info lebih lanjut) dengan tepat ketika bukti memang ambigu atau tidak cukup?
Jawab Ya (0.25) atau Tidak (0.0).
""",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                children=[
                    VerdictNode(verdict=False, score=0),
                    VerdictNode(verdict=True, score=SCORE_0_25),
                ],
            )

            pathogen_type_check = BinaryJudgementNode(
                criteria=f"""
Jenis patogen yang diharapkan: {expected_pathogen}

Berdasarkan output aktual agen:
Apakah jenis patogen yang dinyatakan atau tersirat oleh agen sudah benar (seharusnya: {expected_pathogen})?
Jika agen tidak memberikan klaim spesifik tentang jenis patogen, jawab Tidak.
Jawab Ya atau Tidak.

CATATAN: Terjemahan Bahasa Indonesia diperbolehkan (misal: "jamur/fungi" = "fungal", "bakteri" = "bacterial", "virus" = "viral", "serangga/hama" = "pest/insect").
""",
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                children=[
                    VerdictNode(verdict=False, child=abstention_appropriate),
                    VerdictNode(verdict=True, score=SCORE_0_5),
                ],
            )

            disease_partial_match = BinaryJudgementNode(
                criteria=f"""
Penyakit yang diharapkan: {expected_disease}
Jenis patogen yang diharapkan: {expected_pathogen}

Berdasarkan output aktual agen:
Apakah penyakit "{expected_disease}" disebutkan dalam diagnosis diferensial (top 2-3 kandidat) ATAU apakah kategori/pola penyakitnya sangat konsisten?
Jika ya DAN jenis patogen ({expected_pathogen}) benar, ini menghasilkan skor 0.75.
Jawab Ya atau Tidak.

CATATAN: Penamaan bisa dalam Bahasa Indonesia atau Bahasa Inggris. Terjemahan yang akurat secara semantik (misal: "Hawar Daun" = "Leaf Blight") harus dianggap benar.
""",
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                children=[
                    VerdictNode(verdict=False, child=pathogen_type_check),
                    VerdictNode(verdict=True, score=SCORE_0_75),
                ],
            )

            primary_disease_correct = BinaryJudgementNode(
                criteria=f"""
Penyakit yang diharapkan: {expected_disease}

Berdasarkan output aktual agen:
Apakah identifikasi penyakit UTAMA agen tepat "{expected_disease}" (memperbolehkan sedikit variasi penamaan)?

Jawab Ya atau Tidak.

CATATAN:
1. Penamaan bisa dalam Bahasa Indonesia atau Bahasa Inggris (misal: "Hawar Daun" = "Leaf Blight", "Busuk Hitam" = "Black Rot").
2. Identifikasi yang lebih spesifik menggunakan nama ilmiah patogen (misal: menyebutkan "Stemphylium" untuk "Leaf Blight") harus dianggap BENAR jika patogen tersebut memang penyebab penyakit tersebut.
3. Terjemahan yang akurat secara semantik harus dianggap BENAR.
""",
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                children=[
                    VerdictNode(verdict=False, child=disease_partial_match),
                    VerdictNode(verdict=True, score=SCORE_1_0),
                ],
            )

            healthy_plant_correct = BinaryJudgementNode(
                criteria=f"""
Ini adalah kasus {"TANAMAN SEHAT" if is_healthy_case else "PENYAKIT"} (is_healthy: {is_healthy_case}).

"Kasus TANAMAN SEHAT - Agen harus mengidentifikasi tanaman sebagai SEHAT/tidak ada penyakit/tidak ada gejala sebagai kesimpulan UTAMA. 

PERTIMBANGAN KHUSUS:
1. Agen diperbolehkan memberikan saran perawatan pencegahan atau tips 'agar tetap sehat'.
2. Agen diperbolehkan menyebutkan bercak kecil, sedikit perubahan warna, atau 'tanda awal' yang perlu diwaspadai, SELAMA kesimpulan keseluruhannya tetap menganggap tanaman tersebut sehat/produktif/baik.
3. Jawab Ya jika agen menyatakan tanaman 'terlihat sehat secara keseluruhan' atau semacamnya, meskipun ia juga memberikan diagnosis diferensial sebagai kemungkinan kecil atau saran kewaspadaan.
4. Jawab Tidak HANYA jika agen secara definitif dan tegas menyatakan bahwa tanaman sedang menderita penyakit tertentu sebagai diagnosis utama (misal: 'Tanaman ini menderita Hawar Daun').

Jawab Ya jika agen benar mengidentifikasi sebagai tanaman sehat (sesuai kriteria di atas). Jawab Tidak jika agen salah mengidentifikasi sebagai sakit (menyatakan penyakit secara tegas)." if is_healthy_case else "Kasus PENYAKIT - Ini BUKAN tanaman sehat. Jawab Tidak untuk lanjut ke pengecekan identifikasi penyakit secara detail."
""",
                evaluation_params=[
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                children=[
                    VerdictNode(verdict=False, child=primary_disease_correct),
                    VerdictNode(verdict=True, score=SCORE_1_0),
                ],
            )

            deepeval_disease_accuracy_dag = DeepAcyclicGraph(
                root_nodes=[healthy_plant_correct]
            )

            deepeval_disease_accuracy = DAGMetric(
                name="Disease Accuracy",
                dag=deepeval_disease_accuracy_dag,
                model=deepeval_model_flash,
                threshold=0.5,
                include_reason=True,
                verbose_mode=False,
            )
            
            # DAG metric only supports GPT models, use GPT-only fallback
            score = measure_with_fallback(deepeval_disease_accuracy, disease_test_case, fallback_models=[deepeval_model_pro])
            return [{"key": "disease_accuracy", "score": score, "comment": deepeval_disease_accuracy.reason}]
        except Exception as e:
            return [{"key": "disease_accuracy", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 6. Goal Achievement - With Reference Evaluator
async def goal_achievement_with_ref_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate goal achievement with reference using DeepEval GEval with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        # Create test case with multimodal support
        test_case = create_deepeval_test_case(inputs, outputs, reference_outputs)
        
        final_answer = outputs.get("final_answer", "")
        reference_goal = reference_outputs.get("reference_goal", "")
        
        if not reference_goal:
            return [{"key": "agent_goal_accuracy", "score": 0.0, "comment": "No reference goal provided"}]
        
        # Create the GEval metric
        deepeval_goal_achievement_with_ref = GEval(
            name="Goal Achievement With Reference",
            model=deepeval_model_pro,
            criteria="""
    Evaluasi apakah agen berhasil mencapai tujuan referensi (reference goal) berdasarkan output aktual dan penggunaan alat (tool usage).
    
    Agen dirancang untuk identifikasi penyakit tanaman dengan kemampuan berikut:
    1. Identifikasi penyakit dari gambar dan deskripsi pengguna
    2. Panduan manajemen/pengobatan ketika diminta secara eksplisit
    3. Analisis gejala dan diagnosis diferensial
    4. Ekspresi ketidakpastian yang tepat ketika bukti tidak mencukupi
    5. Penolakan ketika permintaan di luar jangkauan (gambar non-tanaman, saran medis manusia, dll.)
    
    KRITERIA PENCAPAIAN TUJUAN:
    - Pencapaian Penuh (0.9-1.0): Agen sepenuhnya memenuhi tujuan referensi dengan kualitas tinggi, menggunakan alat yang tepat dan mendasarkan respons pada output alat
    - Pencapaian Parsial (0.5-0.8): Agen menangani tujuan utama tetapi melewatkan komponen kunci, kurang mendalam, atau gagal menggunakan output alat secara efektif
    - Pencapaian Minimal (0.2-0.4): Agen mencoba mencapai tujuan tetapi dengan celah signifikan, kesalahan, atau penggunaan alat yang buruk
    - Tidak Ada Pencapaian (0.0-0.1): Agen gagal menangani tujuan atau melenceng sepenuhnya
    
    PERTIMBANGAN PENTING:
    - Periksa output akhir DAN alat yang dipanggil untuk memahami perilaku agen sepenuhnya
    - Jika tujuan adalah "identifikasi penyakit": periksa apakah identifikasi dicoba dengan kualitas wajar dan apakah alat deteksi/klasifikasi digunakan dengan tepat
    - Jika tujuan mencakup "berikan saran manajemen": periksa apakah saran diberikan dan didasarkan pada output alat pencarian (knowledgebase_search, web_search)
    - Jika tujuan adalah "tolak permintaan yang tidak pantas": periksa apakah agen menolak dengan benar beserta penjelasan, menghindari panggilan alat yang tidak perlu
    - Output alat memberikan konteks krusial - verifikasi bahwa respons agen selaras dengan dan menggunakan informasi dari hasil alat secara tepat
    - Ketidakpastian yang tepat dapat diterima dan tidak boleh dihukum berat
    - Mengajukan pertanyaan klarifikasi untuk mencapai tujuan dapat diterima
    - PERTIMBANGAN MULTIMODAL: Jika gambar disediakan, verifikasi agen melakukan observasi visual yang tepat dan menggunakan informasi gambar dengan benar
    """,
            evaluation_steps=[
                "Ekstrak tujuan referensi dari bidang expected_output (diformat sebagai 'REFERENCE_GOAL: ...')",
                "Identifikasi tujuan utama yang dinyatakan dalam tujuan referensi (misalnya, identifikasi penyakit, berikan pengobatan, tolak permintaan)",
                "Periksa bidang tools_called untuk memahami alat apa yang dipanggil agen dan apa output yang dikembalikan",
                "Periksa output aktual untuk menentukan apa yang sebenarnya dikomunikasikan agen kepada pengguna",
                "Periksa apakah agen menggunakan alat yang tepat untuk mencapai tujuan referensi (misalnya, alat deteksi untuk identifikasi, knowledgebase_search untuk panduan)",
                "Verifikasi bahwa output aktual agen menggabungkan dan mencerminkan informasi dari output alat dengan benar",
                "Jika gambar disediakan dalam input, verifikasi agen melakukan observasi visual yang tepat dan menggunakan informasi gambar",
                "Periksa apakah setiap tujuan utama dari tujuan referensi ditangani dalam output aktual",
                "Evaluasi kualitas dan kelengkapan bagaimana setiap tujuan dipenuhi",
                "Untuk tujuan identifikasi: verifikasi apakah penyakit/masalah diidentifikasi dengan bukti pendukung dari output alat DAN analisis visual",
                "Untuk tujuan panduan: verifikasi apakah saran manajemen diberikan dan didasarkan pada output alat pencarian",
                "Untuk tujuan penolakan: verifikasi apakah agen menolak dengan benar dengan penjelasan yang tepat tanpa penggunaan alat yang tidak perlu",
                "Pertimbangkan apakah ketidakpastian yang tepat atau pertanyaan klarifikasi membantu mencapai tujuan",
                "Tentukan tingkat pencapaian keseluruhan: penuh, parsial, minimal, atau tidak ada"
            ],
            rubric=[
                Rubric(
                    score_range=(0, 1),
                    expected_outcome="Tidak ada pencapaian - agen gagal total menangani tujuan referensi, melenceng sepenuhnya, atau menggunakan alat secara tidak tepat tanpa menggabungkan outputnya."
                ),
                Rubric(
                    score_range=(2, 4),
                    expected_outcome="Pencapaian minimal - agen mencoba mencapai tujuan tetapi dengan celah signifikan, kesalahan, atau kehilangan komponen utama, atau pemanfaatan output alat yang buruk."
                ),
                Rubric(
                    score_range=(5, 8),
                    expected_outcome="Pencapaian parsial - agen menangani tujuan utama dan menggunakan alat dengan tepat, tetapi melewatkan komponen kunci, kurang mendalam, memiliki kesalahan kecil, atau tidak sepenuhnya memanfaatkan output alat."
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome="Pencapaian penuh - agen sepenuhnya memenuhi tujuan referensi dengan kualitas tinggi, menggunakan alat yang tepat dan secara efektif mendasarkan respons pada output alat."
                ),
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.TOOLS_CALLED],
            threshold=0.5,
            verbose_mode=False,
        )
        
        try:
            # Handle case where no image is provided
            image_url = inputs.get("image_url")
            if not image_url:
                # Use text-only test case
                goal_test_case = LLMTestCase(
                    input=test_case.input,
                    actual_output=final_answer,
                    expected_output=f"REFERENCE_GOAL: {reference_goal}",
                    tools_called=test_case.tools_called
                )
            else:
                # Use multimodal test case (already created in test_case.input)
                goal_test_case = test_case
                # Override to ensure correct values
                goal_test_case.actual_output = final_answer
                goal_test_case.expected_output = f"REFERENCE_GOAL: {reference_goal}"
            
            score = measure_with_fallback(deepeval_goal_achievement_with_ref, goal_test_case)
            return [{"key": "agent_goal_accuracy", "score": score, "comment": deepeval_goal_achievement_with_ref.reason}]
        except Exception as e:
            return [{"key": "agent_goal_accuracy", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 7. Goal Achievement - Without Reference Evaluator
async def goal_achievement_without_ref_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate goal achievement without reference using DeepEval GEval with multimodal support."""
    with tracing_context(enabled=False):
        # Validate inputs
        inputs = validate_and_normalize_inputs(inputs)
        
        # Create test case with multimodal support
        test_case = create_deepeval_test_case(inputs, outputs, reference_outputs)
        
        final_answer = outputs.get("final_answer", "")
        
        # Create the GEval metric
        deepeval_goal_achievement_without_ref = GEval(
            name="Goal Achievement Without Reference",
            model=deepeval_model_pro,
            criteria="""
    Evaluasi apakah agen menanggapi permintaan pengguna dengan tepat berdasarkan input dan output aktual.
    
    Agen dirancang untuk identifikasi penyakit tanaman dengan kemampuan berikut:
    1. Identifikasi penyakit dari gambar dan deskripsi pengguna
    2. Panduan manajemen/pengobatan ketika diminta secara eksplisit
    3. Analisis gejala dan diagnosis diferensial
    4. Ekspresi ketidakpastian yang tepat ketika bukti tidak mencukupi
    5. Penolakan ketika permintaan di luar jangkauan (gambar non-tanaman, saran medis manusia, dll.)
    
    KRITERIA PENCAPAIAN TUJUAN:
    - Pencapaian Penuh (0.9-1.0): Agen sepenuhnya memenuhi apa yang diminta pengguna dengan kualitas tinggi
    - Pencapaian Parsial (0.5-0.8): Agen menangani permintaan utama tetapi melewatkan komponen atau kurang mendalam
    - Pencapaian Minimal (0.2-0.4): Agen mencoba membantu tetapi dengan celah atau kesalahan signifikan
    - Tidak Ada Pencapaian (0.0-0.1): Agen gagal menangani permintaan atau melenceng sepenuhnya
    
    PERTIMBANGAN PENTING:
    - Simpulkan niat pengguna dari input (hanya identifikasi vs permintaan panduan)
    - Jika pengguna bertanya "apa ini?": tujuannya adalah identifikasi penyakit
    - Jika pengguna bertanya "bagaimana cara mengobati/menangani/memperbaiki": tujuannya mencakup identifikasi + saran manajemen
    - Jika input adalah non-tanaman atau di luar topik: tujuannya adalah penolakan yang tepat
    - Ketidakpastian yang tepat dapat diterima dan tidak boleh dihukum berat
    - Mengajukan pertanyaan klarifikasi untuk melayani pengguna dengan lebih baik dapat diterima
    - PERTIMBANGAN MULTIMODAL: Jika gambar disediakan, verifikasi agen melakukan observasi visual yang tepat dan menggunakan informasi gambar dengan benar
    """,
            evaluation_steps=[
                "Analisis input pengguna untuk menentukan apa yang diminta pengguna (identifikasi, panduan, keduanya, atau lainnya)",
                "Identifikasi apakah permintaan tersebut tepat (terkait tanaman) atau harus ditolak",
                "Periksa output aktual untuk menentukan apa yang sebenarnya dilakukan agen",
                "Periksa apakah agen menangani permintaan utama pengguna",
                "Evaluasi kualitas dan kelengkapan respons",
                "Jika gambar disediakan dalam input, verifikasi agen melakukan observasi visual yang tepat dan menggunakan informasi gambar",
                "Untuk permintaan identifikasi: verifikasi apakah penyakit/masalah diidentifikasi dengan bukti pendukung DAN analisis visual",
                "Untuk permintaan panduan: verifikasi apakah saran manajemen diberikan dan didasarkan pada bukti",
                "Untuk permintaan yang tidak pantas: verifikasi apakah agen menolak dengan benar",
                "Pertimbangkan apakah ketidakpastian yang tepat atau pertanyaan klarifikasi melayani pengguna dengan baik",
                "Tentukan tingkat pencapaian keseluruhan: penuh, parsial, minimal, atau tidak ada"
            ],
            rubric=[
                Rubric(
                    score_range=(0, 1),
                    expected_outcome="Tidak ada pencapaian - agen gagal total menangani permintaan pengguna atau melenceng sepenuhnya."
                ),
                Rubric(
                    score_range=(2, 4),
                    expected_outcome="Pencapaian minimal - agen mencoba membantu tetapi dengan celah signifikan, kesalahan, atau kehilangan komponen utama."
                ),
                Rubric(
                    score_range=(5, 8),
                    expected_outcome="Pencapaian parsial - agen menangani permintaan utama tetapi melewatkan komponen, kurang mendalam, atau memiliki masalah kecil."
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome="Pencapaian penuh - agen sepenuhnya memenuhi permintaan pengguna dengan kualitas tinggi dan kelengkapan."
                ),
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            verbose_mode=False,
        )
        
        try:
            # Handle case where no image is provided
            image_url = inputs.get("image_url")
            if not image_url:
                # Use text-only test case
                goal_test_case = LLMTestCase(
                    input=test_case.input,
                    actual_output=final_answer
                )
            else:
                # Use multimodal test case (already created in test_case.input)
                goal_test_case = test_case
                # Override to ensure correct values
                goal_test_case.actual_output = final_answer
            
            score = measure_with_fallback(deepeval_goal_achievement_without_ref, goal_test_case)
            return [{"key": "goal_achievement_without_ref", "score": score, "comment": deepeval_goal_achievement_without_ref.reason}]
        except Exception as e:
            return [{"key": "goal_achievement_without_ref", "score": 0.0, "comment": f"Error: {str(e)}"}]


# RAG Metrics (RAGAS)
# ragas_faithfulness = Faithfulness(llm=ragas_llm_flash)
ragas_context_relevance = ContextRelevance(llm=ragas_llm_pro)

# Custom Faithfulness Metric for Plant Health Agent (DeepEval GEval)
deepeval_plant_faithfulness = GEval(
    name="Plant Agent Faithfulness",
    model=deepeval_model_pro,
    criteria="""
    Evaluasi apakah output agen setia (faithful) terhadap bukti yang ada, dengan mempertimbangkan kemampuan spesifik dan domain agen.
    
    KEMAMPUAN AGEN & PERILAKU YANG DIHARAPKAN:
    1. Observasi Visual: Agen HARUS melakukan observasi dari gambar (gejala, pola, warna, lokasi)
    2. Saran Perawatan Tanaman Umum: Agen DAPAT memberikan praktik terbaik umum (isolasi, sanitasi, aliran udara, pemangkasan, konsultasi layanan ekstensi) TANPA pencarian
    3. Klaim Spesifik Penyakit: Agen HARUS mendasarkan identifikasi penyakit dan manajemen spesifik penyakit pada konteks yang dicari atau informasi yang diberikan pengguna
    4. Ekspresi Ketidakpastian: Agen HARUS menyatakan ketidakpastian yang tepat ketika bukti terbatas
    
    KRITERIA EVALUASI KESETIAAN (FAITHFULNESS):
    
    DIPERBOLEHKAN (BUKAN halusinasi):
    - Observasi visual dari gambar: "Daun menunjukkan bintik melingkar," "halo kuning terlihat," "lesi di sisi bawah," "tampilan basah"
    - Rekomendasi perawatan tanaman umum: "tingkatkan aliran udara," "buang bagian yang terinfeksi," "isolasi tanaman," "sanitasi alat," "konsultasi dengan ahli"
    - Integrasi konteks pengguna: Merujuk pada spesies tanaman, garis waktu, atau kondisi yang disebutkan pengguna
    - Ketidakpastian yang tepat: "kemungkinan," "tampaknya," "konsisten dengan," "bisa mengindikasikan"
    - Pengetahuan domain standar: Biologi tanaman dasar, pola gejala umum, praktik perawatan umum
    
    MEMERLUKAN DUKUNGAN PENCARIAN (dianggap halusinasi jika tidak didukung):
    - Identifikasi penyakit: Menyebutkan nama penyakit spesifik harus didasarkan pada gejala yang diamati + pengetahuan yang dicari
    - Pengobatan spesifik penyakit: Rekomendasi kimia, fungisida/pestisida spesifik, waktu aplikasi
    - Informasi siklus hidup spesifik: Tahapan hibernasi, garis waktu infeksi, pemicu lingkungan
    - Spesifik wilayah/kultivar: Ketahanan varietas, prevalensi wilayah, rekomendasi lokal
    - Nama produk atau formulasi spesifik: Konsentrasi tembaga sulfat, nama merek, bahan aktif
    
    RUBRIK PENILAIAN:
    - 0.9-1.0 (Sangat Setia): Semua klaim didasarkan pada bukti dengan tepat; observasi visual jelas; saran umum masuk akal; klaim spesifik penyakit didukung
    - 0.7-0.8 (Sebagian Besar Setia): Detail kecil tidak didukung tetapi klaim inti didasarkan pada bukti; tidak ada misrepresentasi signifikan
    - 0.5-0.6 (Setia Sebagian): Beberapa klaim spesifik penyakit kurang dukungan ATAU mengandung detail spekulatif yang disajikan sebagai fakta
    - 0.3-0.4 (Sebagian Besar Tidak Setia): Banyak klaim spesifik penyakit tidak didukung ATAU fabrikasi detail yang signifikan
    - 0.0-0.2 (Tidak Setia): Informasi didominasi oleh fabrikasi; klaim penyakit tidak didasarkan pada bukti; rekomendasi spesifik yang dikarang
    
    CATATAN PENTING:
    - Agen bekerja dengan GAMBAR - jika gambar disediakan, observasi visual DIHARAPKAN dan VALID
    - Saran perawatan umum adalah bagian dari keahlian domain agen - TIDAK memerlukan pencarian
    - Hanya tandai klaim spesifik penyakit (nama penyakit, pengobatan spesifik, biologi spesifik) jika tidak didukung
    - Merekomendasikan "konsultasi layanan ahli/ekstensi" adalah praktik terbaik standar - BUKAN halusinasi
    - Fokus pada halusinasi SUBSTANTIF, bukan pilihan gaya bahasa atau inferensi yang masuk akal
    """,
    evaluation_steps=[
        "Identifikasi semua observasi visual dalam output aktual (deskripsi tentang apa yang terlihat di gambar)",
        "Identifikasi semua rekomendasi perawatan tanaman umum (sanitasi, isolasi, aliran udara, konsultasi ahli, dll.)",
        "Identifikasi semua klaim spesifik penyakit (nama penyakit, pengobatan spesifik, biologi patogen spesifik)",
        "Periksa apakah observasi visual masuk akal untuk agen penganalisis gambar (ini VALID)",
        "Periksa apakah rekomendasi perawatan umum adalah praktik terbaik yang sesuai domain (ini VALID)",
        "Untuk setiap klaim spesifik penyakit, periksa apakah itu didukung oleh retrieval_context atau input pengguna",
        "Identifikasi detail spesifik yang difabrikasi (bahan kimia karangan, garis waktu palsu, detail spesifik yang salah)",
        "Tentukan apakah output secara tepat menyatakan ketidakpastian di mana bukti terbatas",
        "Hitung jumlah dan tingkat keparahan klaim spesifik penyakit yang tidak didukung",
        "Berikan skor kesetiaan berdasarkan keseimbangan antara klaim spesifik yang didasarkan pada bukti vs yang tidak didukung"
    ],
    rubric=[
        Rubric(
            score_range=(0, 2),
            expected_outcome="Tidak setia - informasi spesifik penyakit didominasi fabrikasi; banyak klaim penyakit tidak didukung; rekomendasi spesifik dikarang tanpa dukungan pencarian."
        ),
        Rubric(
            score_range=(3, 4),
            expected_outcome="Sebagian besar tidak setia - beberapa klaim spesifik penyakit yang signifikan kurang dukungan pencarian; detail pengobatan atau biologi difabrikasi atau tidak terverifikasi."
        ),
        Rubric(
            score_range=(5, 6),
            expected_outcome="Setia sebagian - beberapa klaim spesifik penyakit kurang dukungan ATAU detail spekulatif disajikan sebagai fakta; identifikasi inti mungkin didasarkan pada bukti tetapi detail pendukungnya tidak."
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Sebagian besar setia - identifikasi penyakit dan klaim utama didasarkan pada pencarian atau konteks pengguna; hanya detail kecil yang tidak didukung; penggunaan observasi visual dan saran umum yang tepat."
        ),
        Rubric(
            score_range=(9, 10),
            expected_outcome="Sangat setia - semua klaim spesifik penyakit didasarkan pada pencarian atau input pengguna dengan tepat; observasi visual jelas dan masuk akal; saran perawatan umum sesuai standar; ketidakpastian dinyatakan dengan tepat."
        ),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.TOOLS_CALLED],
    threshold=0.5,
    verbose_mode=False,
)

# Trajectory Accuracy - With Reference (LangChain AgentEval)
TRAJECTORY_WITH_REF_PROMPT = """Anda adalah evaluator ahli untuk agen identifikasi penyakit tanaman.

Tugas Anda adalah mengevaluasi apakah trajektori penggunaan alat (tool usage trajectory) agen merupakan SUPERSET YANG VALID dari trajektori referensi.

<Prinsip Desain Agen>
Agen dirancang untuk identifikasi penyakit tanaman dengan prinsip alur kerja berikut:
1. Identifikasi visual terlebih dahulu (plant_disease_identification) ketika gambar tanaman yang jelas disediakan
2. Pra-pemrosesan deteksi opsional (closed_set_leaf_detection, open_vocabulary_plant_detection, atau open_set_object_detection) SELALU dapat diterima
3. Pencarian pengetahuan (knowledgebase_search atau web_search) diperlukan ketika pengguna meminta panduan/pengobatan
4. Penggunaan alat cadangan (fallback) ke knowledgebase_search/web_search jika alat visual memberikan kepercayaan rendah atau hasil yang tidak konsisten
</Prinsip Desain Agen>

<Semantik Superset>
- Agen HARUS memanggil semua alat WAJIB (non-deteksi) dalam trajektori referensi (atau alternatif yang setara secara semantik).
- Alat deteksi (closed_set_leaf_detection, open_vocabulary_plant_detection) bersifat opsional. Agen tidak dihukum jika menghilangkannya, tetapi mendapat nilai lebih jika memanggilnya sesuai dengan referensi.
- Agen DAPAT memanggil alat TAMBAHAN jika sesuai:
  * Alat deteksi yang tidak ada di referensi selalu merupakan tambahan yang dapat diterima.
  * web_search dapat diterima jika knowledgebase_search diharapkan (atau sebaliknya) - keduanya adalah pencarian pengetahuan.
  * Alat cadangan (fallback) dapat diterima ketika alat utama menunjukkan ketidakpastian.
- Urutan alat dapat bervariasi jika logis (misalnya, deteksi sebelum klasifikasi).
</Semantik Superset>

<Analisis Pencarian Pengetahuan>
- Jika referensi memiliki knowledgebase_search ATAU web_search: agen harus memanggil setidaknya SATU alat pencarian
- Kedua alat tersebut melayani tujuan yang sama (pencarian pengetahuan) dan dapat saling menggantikan dalam banyak kasus
- Memanggil KEDUANYA dapat diterima untuk validasi silang atau ketika basis pengetahuan (KB) tidak mencukupi
</Analisis Pencarian Pengetahuan>

<JANGAN Menghukum>
- Menambahkan alat deteksi sebelum klasifikasi.
- MENGHILANGKAN alat deteksi, bahkan jika ada di referensi (meskipun memanggilnya lebih baik).
- Menggunakan web_search alih-alih knowledgebase_search (atau sebaliknya).
- Memanggil alat validasi tambahan ketika ketidakpastian tinggi.
- Nilai parameter yang berbeda untuk alat yang sama.
</JANGAN>

<Langkah Evaluasi>
1. Ekstrak nama alat dari trajektori referensi (reference_tool_calls).
2. Ekstrak nama alat dari trajektori aktual.
3. Periksa apakah SEMUA alat WAJIB (non-deteksi) dari referensi ada dalam trajektori aktual (memperbolehkan substitusi knowledgebase_search ↔ web_search).
4. Identifikasi alat TAMBAHAN dalam trajektori aktual yang tidak ada dalam referensi.
5. Evaluasi apakah alat tambahan tersebut dibenarkan: alat deteksi selalu oke, alat pencarian oke untuk validasi/cadangan.
6. PERTIMBANGKAN PEMANGGILAN ALAT DETEKSI: Jika alat deteksi ada di referensi dan agen memanggilnya, ini adalah sinyal positif untuk skor yang lebih tinggi. Jika agen tidak memanggilnya, jangan hukum secara signifikan.
7. Nilai alur logis: alat harus dalam urutan yang masuk akal berdasarkan permintaan pengguna dan kompleksitas gambar.
8. Tentukan apakah trajektori mencapai tujuan yang sama dengan referensi dengan tambahan yang dapat diterima atau menguntungkan.
</Langkah Evaluasi>

<Rubrik>
Skor 0.0:
- Kehilangan alat kritis - alat referensi WAJIB (non-deteksi) sama sekali tidak ada (misalnya, tidak ada alat identifikasi padahal referensi mengharapkannya).

Skor 0.2:
- Kehilangan banyak alat kunci - beberapa alat referensi wajib ada tetapi kehilangan langkah penting (misalnya, kehilangan pencarian padahal pengguna meminta pengobatan).

Skor 0.5:
- Memiliki semua alat referensi wajib tetapi dengan tambahan yang bermasalah - mencakup panggilan alat tambahan yang tidak perlu atau tidak logis yang tidak selaras dengan desain agen.

Skor 0.75:
- Superset valid dengan masalah minor - semua alat referensi wajib ada, alat tambahan sebagian besar dibenarkan, masalah alur logis minor. Mungkin menghilangkan alat deteksi opsional dari referensi.

Skor 1.0:
- Superset sempurna - semua alat referensi (termasuk deteksi opsional yang diharapkan) ada, setiap alat tambahan menguntungkan (deteksi, validasi, cadangan), alur logis.
</Rubrik>

<Trajectory Referensi>
{reference_tool_calls}
</Trajectory Referensi>

<Input Pengguna>
{input}
</Input Pengguna>

<Trajectory Aktual>
{outputs}
</Trajectory Aktual>

Berikan skor Anda (0.0, 0.2, 0.5, 0.75, atau 1.0) dan penalaran mendalam.
"""

# Trajectory Accuracy - Without Reference (LangChain AgentEval)
TRAJECTORY_WITHOUT_REF_PROMPT = """Anda adalah evaluator ahli untuk agen identifikasi penyakit tanaman.

Tugas Anda adalah mengevaluasi alur logis dan efisiensi trajektori penggunaan alat agen berdasarkan prinsip desain.

<Prinsip Desain Agen>
Agen dirancang untuk identifikasi penyakit tanaman dengan prinsip-prinsip berikut:
1. Visi-pertama: Mulai dengan triase visual sebelum panggilan alat
2. Identifikasi visual (plant_disease_identification) ketika gambar adalah jaringan tanaman yang jelas
3. Pra-pemrosesan deteksi opsional (closed_set_leaf_detection, open_vocabulary_plant_detection, open_set_object_detection) untuk pemandangan kompleks
4. Pencarian pengetahuan WAJIB ketika pengguna meminta panduan/pengobatan/manajemen
5. Cadangan (fallback) ke knowledgebase_search atau web_search ketika alat utama menunjukkan kepercayaan rendah (<0.5) atau kontradiksi
6. Validasi silang output ketika ketidakpastian tinggi
7. Lewati alat pada gambar non-tanaman atau foto yang tidak jelas
</Prinsip Desain Agen>

<Variasi yang Dapat Diterima>
- Alat deteksi sebelum klasifikasi (untuk pemandangan berantakan, banyak daun)
- web_search vs knowledgebase_search (keduanya adalah pencarian, dapat saling menggantikan)
- Beberapa panggilan pencarian untuk validasi silang
- Menolak output alat dengan kepercayaan rendah
</Variasi yang Dapat Diterima>

<Masalah yang Perlu Dihukum>
- Memanggil alat pada gambar non-tanaman
- Tidak ada pencarian ketika pengguna bertanya "bagaimana cara mengobati" atau "apa yang harus dilakukan"
- Tidak menggunakan cadangan (fallback) ketika alat utama gagal
- Panggilan berlebih yang redundan (alat yang sama, parameter yang sama, tidak ada informasi baru)
- Menerima output yang kontradiktif tanpa validasi
</Masalah yang Perlu Dihukum>

<Langkah Evaluasi>
1. Identifikasi jenis permintaan pengguna: hanya identifikasi vs permintaan panduan (pengobatan, manajemen, pencegahan)
2. Periksa apakah gambar ada dan berisi jaringan tanaman (berdasarkan observasi agen)
3. Evaluasi pemilihan alat: apakah plant_disease_identification dipanggil saat sesuai?
4. Periksa apakah alat deteksi digunakan dengan tepat (pemandangan kompleks) atau dilewati (gambar sederhana)
5. Jika pengguna meminta panduan: verifikasi knowledgebase_search ATAU web_search dipanggil
6. Nilai perilaku cadangan (fallback): apakah agen menggunakan pencarian saat alat visual menunjukkan kepercayaan rendah atau kesalahan?
7. Periksa redundansi: apakah ada panggilan alat duplikat yang tidak perlu?
8. Evaluasi alur logis: urutan alat yang masuk akal dan pengambilan keputusan berbasis bukti
</Langkah Evaluasi>

<Rubrik>
Skor 0.0:
- Kegagalan kritis - memanggil alat pada gambar non-tanaman, memberikan panduan tanpa pencarian saat diperlukan, atau trajektori yang sama sekali tidak logis

Skor 0.2:
- Trajektori buruk - kehilangan alat wajib (pencarian untuk permintaan panduan), tidak ada cadangan untuk alat yang gagal, atau banyak panggilan redundan

Skor 0.5:
- Dapat diterima - mencapai kesimpulan yang benar tetapi dengan inefisiensi (panggilan redundan, urutan suboptimal, masalah logis minor)

Skor 0.75:
- Trajektori baik - alur sebagian besar logis, pemilihan alat yang tepat, inefisiensi minor (misalnya, satu panggilan tidak perlu), penggunaan cadangan yang tepat

Skor 1.0:
- Trajektori luar biasa - kemajuan alat yang optimal, jalur yang efisien, cadangan/validasi yang tepat, mengikuti semua prinsip desain agen
</Rubrik>

<Input Pengguna>
{input}
</Input Pengguna>

<Trajectory Aktual>
{outputs}
</Trajectory Aktual>

Pertama, identifikasi tujuan pengguna dari input. Kemudian, evaluasi apakah trajektori mencapai tujuan tersebut secara efisien dan logis sesuai prinsip desain agen.

Berikan skor Anda (0.0, 0.2, 0.5, 0.75, atau 1.0) dan penalaran mendalam.
"""

# Disease Accuracy (DeepEval DAG) - Configuration preserved for reference
# Score mapping (0-10 ints; metric.score is 0-1 normalized)
SCORE_1_0  = 10
SCORE_0_75 = 8
SCORE_0_5  = 5
SCORE_0_25 = 3

# ============================================================================
# Helper Functions
# ============================================================================

def _get_final_answer(messages: List[Any]) -> str:
    """Extract the last AI message content as the final answer."""
    for msg in reversed(messages):
        if isinstance(msg, lc.AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif "text" in block:
                            text_parts.append(str(block["text"]))
                result = "\n".join(text_parts).strip()
                if result:
                    return result
    return ""


def _extract_retrieval_context_from_trace(trace_messages: dict) -> list[str]:
    """Extract all tool outputs from trace, including retrieval, detection, and classification results.
    
    This function extracts context from all tool calls to provide comprehensive information
    It handles:
    - Retrieval tools (knowledgebase_search, web_search) with structured document outputs
    - Detection tools (closed_set_leaf_detection, open_set_object_detection) with summary outputs
    - Classification tools (plant_disease_identification) with classification results
    
    Returns:
        List of context strings from all tool outputs
    """
    retrieval_context = []
    messages = trace_messages.get("messages", [])
    for msg in messages:
        # Handle both message objects and message dictionaries
        msg_name = None
        msg_content = None
        
        if isinstance(msg, dict):
            msg_name = msg.get('name')
            msg_content = msg.get('content')
        elif hasattr(msg, 'name'):
            msg_name = msg.name
            msg_content = msg.content
        
        if msg_name in ['knowledgebase_search', 'web_search']:
            try:
                import json
                docs = json.loads(msg_content)
                for doc in docs:
                    if isinstance(doc, dict) and "page_content" in doc:
                        # Extract page content
                        page_content = doc["page_content"]

                        # Extract metadata if available
                        metadata = doc.get("metadata", {})

                        # Build comprehensive context string
                        context_parts = []

                        # Add page content
                        if page_content:
                            context_parts.append(f"Content: {page_content}")

                        # Add metadata information
                        if metadata:
                            # Add plant name (new payload structure)
                            if "plant" in metadata:
                                context_parts.append(f"Plant: {metadata['plant']}")

                            # Add disease name (new payload structure)
                            if "disease" in metadata:
                                context_parts.append(f"Disease: {metadata['disease']}")

                            # Add document type (new payload structure)
                            if "type" in metadata:
                                context_parts.append(f"Type: {metadata['type']}")

                            # Add source (new payload structure)
                            if "source" in metadata:
                                context_parts.append(f"Source: {metadata['source']}")

                            # Add relevance score for context
                            if "relevance_score" in metadata:
                                context_parts.append(f"Relevance: {metadata['relevance_score']:.3f}")

                        # Join all parts into a comprehensive context string
                        if context_parts:
                            retrieval_context.append(" | ".join(context_parts))
                        else:
                            retrieval_context.append(page_content)

            except:
                if isinstance(msg_content, str):
                    retrieval_context.append(msg_content)
        # Also include outputs from detection and classification tools
        elif msg_name in ['closed_set_leaf_detection', 'open_set_object_detection', 'plant_disease_identification']:
            # These tools return structured summaries that should be included as context
            if isinstance(msg_content, str):
                retrieval_context.append(f"[{msg_name}]: {msg_content}")
            else:
                retrieval_context.append(f"[{msg_name}]: {str(msg_content)}")
    return retrieval_context


def _extract_text_content(content) -> str:
    """Extract plain text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block:
                    text_parts.append(str(block["text"]))
        return "\n".join(text_parts).strip()
    return str(content) if content else ""


def _ragas_user_input_from_trace(trace_messages):
    """Convert trace messages to RAGAS message format."""
    user_input = []
    
    if isinstance(trace_messages, dict):
        messages = trace_messages.get("messages", [])
    elif isinstance(trace_messages, list):
        messages = trace_messages
    else:
        return user_input
    
    for m in messages:
        if hasattr(m, '__class__'):
            msg_type = m.__class__.__name__
            
            if msg_type == "HumanMessage":
                content = _extract_text_content(m.content)
                user_input.append(r.HumanMessage(content=content))
                
            elif msg_type == "AIMessage":
                tcs = []
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    for tc in m.tool_calls:
                        tool_name = tc.get('name') if isinstance(tc, dict) else tc.name
                        tool_args = tc.get('args', {}) if isinstance(tc, dict) else (tc.args if hasattr(tc, 'args') else {})
                        tcs.append(r.ToolCall(name=tool_name, args=tool_args))
                        
                content = _extract_text_content(m.content)
                user_input.append(r.AIMessage(content=content, tool_calls=tcs))
                
            elif msg_type == "ToolMessage":
                content = _extract_text_content(m.content)
                user_input.append(r.ToolMessage(content=content))
                
    return user_input


def create_deepeval_test_case(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> LLMTestCase:
    """
    Create a DeepEval LLMTestCase from agent inputs/outputs with multimodal support.
    
    Args:
        inputs: Dict with 'user_text' and 'image_url' (optional)
        outputs: Dict with 'final_answer' and 'trace_messages'
        reference_outputs: Dict with reference data
        
    Returns:
        LLMTestCase configured for evaluation with multimodal support
    """
    # Validate and normalize inputs
    inputs = validate_and_normalize_inputs(inputs)
    
    # Extract data
    user_text = inputs.get("user_text", "")
    image_url = inputs.get("image_url")
    final_answer = outputs.get("final_answer", "")
    trace_messages = outputs.get("trace_messages", {})
    
    # Extract reference data
    reference_answer = reference_outputs.get("reference_answer")
    
    # Get metadata - this comes from LangSmith example metadata
    metadata = reference_outputs.get("metadata", {})
    
    # Create structured expected_output with disease and pathogen type from metadata
    expected_disease = metadata.get("class", "")
    expected_pathogen_type = metadata.get("pathogen_type", "")
    
    # Detect healthy plant cases
    is_healthy_case = False
    if expected_pathogen_type == "healthy":
        is_healthy_case = True 
        
    # Combine reference answer with structured metadata for DAG parsing
    if is_healthy_case:
        expected_output = f"""{reference_answer}

METADATA:
- expected_disease: TANAMAN SEHAT ({expected_disease})
- expected_pathogen_type: Tidak ada (tanaman sehat)
- is_healthy: True"""
    else:
        expected_output = f"""{reference_answer}

METADATA:
- expected_disease: {expected_disease}
- expected_pathogen_type: {expected_pathogen_type}
- is_healthy: False"""
    
    # Extract retrieval context from tool messages (list of strings for DeepEval)
    retrieval_context = _extract_retrieval_context_from_trace(trace_messages)
    
    # Extract tool calls from trace with their outputs
    tools_called = []
    messages = trace_messages.get("messages", [])
    
    # First, build a map of tool_call_id to tool output for quick lookup
    tool_outputs = {}
    for msg in messages:
        msg_type = msg.get('type') if isinstance(msg, dict) else getattr(msg, 'type', None)
        if msg_type == 'tool':
            tool_call_id = msg.get('tool_call_id') if isinstance(msg, dict) else getattr(msg, 'tool_call_id', None)
            tool_content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
            if tool_call_id:
                tool_outputs[tool_call_id] = tool_content
    
    # Now extract tool calls from AI messages and match with their outputs
    for msg in messages:
        msg_tool_calls = None
        if isinstance(msg, dict):
            msg_tool_calls = msg.get('tool_calls', [])
        elif hasattr(msg, 'tool_calls'):
            msg_tool_calls = msg.tool_calls
        
        if msg_tool_calls:
            for tc in msg_tool_calls:
                # Handle both dict and object tool calls
                if isinstance(tc, dict):
                    tool_name = tc.get('name')
                    tool_args = tc.get('args', {})
                    tool_id = tc.get('id')
                else:
                    tool_name = getattr(tc, 'name', None)
                    tool_args = getattr(tc, 'args', {})
                    tool_id = getattr(tc, 'id', None)
                
                # Get the output for this tool call
                tool_output = tool_outputs.get(tool_id)
                
                tools_called.append(
                    ToolCall(
                        name=tool_name,
                        input_parameters=tool_args,
                        output=tool_output
                    )
                )
    
    # Extract expected tools from reference
    expected_tools = []
    reference_tool_calls = reference_outputs.get("reference_tool_calls", [])
    for tc in reference_tool_calls:
        expected_tools.append(
            ToolCall(
                name=tc.get('name'),
                input_parameters=tc.get('args', {})
            )
        )
    
    # Create multimodal input with MLLMImage if image is provided
    # This allows the LLM judge to see the image during evaluation
    if image_url:
        try:
            # Process and resize image to avoid timeouts with large files
            processed_image_path = process_and_resize_image(image_url)
            
            # Create MLLMImage object
            # Note: We now use the local processed path which is guaranteed to be a local file
            image = MLLMImage(url=processed_image_path, local=True)
            
            # Create multimodal input - embed image reference in the input string
            # DeepEval will convert this to special format [DEEPEVAL:IMAGE:uuid]
            test_input = f"{user_text} {image}"
        except Exception as e:
            # Fallback to text-only if image creation fails
            print(f"Warning: Failed to create MLLMImage for {image_url}: {e}")
            test_input = user_text
    else:
        # Text-only case - still include a note that no image was provided
        # This helps the evaluator understand the context
        test_input = user_text
    
    # Create multimodal retrieval context if available
    # This allows the judge to see images that were in retrieved context
    multimodal_retrieval_context = []
    if retrieval_context:
        for context_item in retrieval_context:
            # Check if context contains image references
            # For now, we'll keep it as text - DeepEval handles this automatically
            multimodal_retrieval_context.append(context_item)
    
    # Create the test case
    test_case = LLMTestCase(
        input=test_input,
        actual_output=final_answer,
        expected_output=expected_output,
        retrieval_context=multimodal_retrieval_context,
        tools_called=tools_called,
        expected_tools=expected_tools,
        # Store additional metadata for custom metrics
        additional_metadata={
            "reference_goal": reference_outputs.get("reference_goal"),
            "reference_context": reference_outputs.get("reference_context"),
            "metadata": metadata,
            "image_url": image_url,  # Store for potential use in custom logic
        }
    )
    
    return test_case

# ============================================================================
# Agent Target Function
# ============================================================================

async def target(inputs: dict) -> dict:
    """Run the agent on input and return outputs for evaluation."""
    user_text = inputs["user_text"]
    image_url = inputs.get("image_url")
    
    human = lc.HumanMessage(
        content=[
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": image_url},
        ]
    )

    graph_inputs = {
        "messages": [human],
        "current_image_url": image_url,
    }

    user_preview = user_text[:50] + "..." if len(user_text) > 50 else user_text
    configured_graph = agent_graph.with_config({
        "run_name": f"Agent: {user_preview}",
        "tags": ["evaluation", "thesis-agent"],
    })
    
    result = await configured_graph.ainvoke(graph_inputs)
    
    messages = result.get("messages") if isinstance(result, dict) else None
    if messages is None:
        raise ValueError("Agent output did not include 'messages' list.")

    return {
        "final_answer": _get_final_answer(messages),
        "trace_messages": result,
    }

# ============================================================================
# Retry Helper with Model Fallback
# ============================================================================

def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is related to rate limiting or overload."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "503",
        "overloaded",
        "rate limit",
        "quota",
        "resource exhausted",
        "too many requests",
        "429"
    ])

def measure_with_fallback(metric, test_case, fallback_models=None):
    """
    Measure a metric with automatic fallback to alternative models on rate limiting.
    
    Args:
        metric: The DeepEval metric to measure
        test_case: The test case to evaluate
        fallback_models: List of fallback models to try (defaults to [deepeval_model_gpt, deepeval_model_flash])
    
    Returns:
        The metric score
    
    Raises:
        Exception: If all models fail
    """
    if fallback_models is None:
        fallback_models = [deepeval_model_pro, deepeval_model_flash]
    
    original_model = metric.model
    all_models = [original_model] + fallback_models
    
    last_error = None
    for i, model in enumerate(all_models):
        try:
            metric.model = model
            metric.measure(test_case)
            if i > 0:  # Only print if we used a fallback
                print(f"  ✓ Successfully retried with {model.model}")
            return metric.score
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                if i < len(all_models) - 1:  # Not the last model
                    print(f"  ⚠ Rate limited on {model.model}, trying fallback...")
                    continue
                else:
                    print(f"  ✗ All models exhausted for {metric.name}")
            else:
                # Non-rate-limit error, don't retry
                break
        finally:
            # Restore original model
            metric.model = original_model
    
    # All attempts failed
    raise last_error

# ============================================================================
# Evaluator Function - Now uses separate evaluators
# ============================================================================

async def evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Comprehensive evaluator that calls all individual metric evaluators."""
    results = []
    
    # Normalize reference_outputs if needed
    if isinstance(reference_outputs, str):
        try:
            reference_outputs = json.loads(reference_outputs)
        except Exception:
            try:
                reference_outputs = ast.literal_eval(reference_outputs)
            except Exception:
                reference_outputs = {}
    if not isinstance(reference_outputs, dict):
        reference_outputs = {}
    
    # Ensure metadata is available for create_deepeval_test_case
    metadata = reference_outputs.get("metadata", {})
    reference_outputs["metadata"] = metadata
    
    # Get reference data
    reference_goal = reference_outputs.get("reference_goal")
    reference_tool_calls = reference_outputs.get("reference_tool_calls", [])
    
    # Get trace messages and final answer
    trace_messages = outputs.get("trace_messages", {})
    final_answer = outputs.get("final_answer", "")
    
    # Extract retrieval context to determine which metrics to run
    retrieved_context = _extract_retrieval_context_from_trace(trace_messages)
    has_retrieval_context = retrieved_context and len(retrieved_context) > 0
    
    # Run all evaluators and collect results
    print("  Running individual evaluators...")
    
    # 1. Context Relevance (only if retrieval context exists)
    if has_retrieval_context:
        print("    - Context Relevance")
        context_results = await context_relevance_evaluator(inputs, outputs, reference_outputs)
        results.extend(context_results)
    
    # 2. Faithfulness (only if retrieval context exists)
    if has_retrieval_context:
        print("    - Faithfulness")
        faithfulness_results = await faithfulness_evaluator(inputs, outputs, reference_outputs)
        results.extend(faithfulness_results)
    
    # 3. Trajectory Accuracy - With Reference 
    if reference_tool_calls:
        print("    - Trajectory Accuracy (with reference)")
        trajectory_results = await trajectory_with_ref_evaluator_func(inputs, outputs, reference_outputs)
        results.extend(trajectory_results)
    
    # 4. Disease Accuracy
    print("    - Disease Accuracy")
    disease_results = await disease_accuracy_evaluator(inputs, outputs, reference_outputs)
    results.extend(disease_results)
    
    # 5. Goal Achievement - With Reference (if reference goal exists)
    if reference_goal:
        print("    - Goal Achievement (with reference)")
        goal_results = await goal_achievement_with_ref_evaluator(inputs, outputs, reference_outputs)
        results.extend(goal_results)
    else:
        # 8. Goal Achievement - Without Reference
        print("    - Goal Achievement (without reference)")
        goal_results = await goal_achievement_without_ref_evaluator(inputs, outputs, reference_outputs)
        results.extend(goal_results)
    
    return results

async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    description: str,
    max_concurrency: int = 2
):
    """Run evaluation on a specific dataset."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    experiment_prefix = f"{experiment_name}-{timestamp}"
    
    print("=" * 80)
    print(f"Running Evaluation: {experiment_name}")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {experiment_prefix}")
    print(f"Description: {description}")
    print(f"Max Concurrency: {max_concurrency}")
    print("=" * 80)
    
    with tracing_context(enabled=False):
        result = await aevaluate(
            target,
            data=dataset_name,
            evaluators=[disease_accuracy_evaluator],
            experiment_prefix=experiment_prefix,
            description=description,
            max_concurrency=max_concurrency,
            metadata={
                "dataset_type": dataset_name,
                "evaluation_timestamp": timestamp,
                "agent_version": "v1",
            },
        )
    
    return result


async def continue_evaluation(
    experiment_id: str,
    description: str = "Continuing incomplete experiment",
    max_concurrency: int = 8
):
    """
    Continue an incomplete experiment by running evaluators on missing examples.
    
    Args:
        experiment_id: The experiment name or UUID from LangSmith
        description: Human-readable description 
        max_concurrency: Number of concurrent evaluations
    
    Returns:
        Evaluation results object
    """
    print("=" * 80)
    print(f"Continuing Incomplete Experiment")
    print("=" * 80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Description: {description}")
    print(f"Max Concurrency: {max_concurrency}")
    print("=" * 80)
    print("\nNote: This will only run evaluators on examples that don't have results yet.")
    print("The agent won't be re-run, only the evaluation metrics will be computed.")
    print("=" * 80)
    
    # Wrap the entire batch in disable_tracing to avoid concurrency race conditions
    # with os.environ toggling inside individual evaluators
    with tracing_context(enabled=False):
        result = await aevaluate_existing(
            experiment_id,
            evaluators=[disease_accuracy_evaluator],
            max_concurrency=max_concurrency,
        )
    
    print(f"\nEvaluation complete for: {experiment_id}")
    print(f"View in LangSmith: https://smith.langchain.com")
    print("=" * 80 + "\n")
    
    return result


# ============================================================================
# Batch Evaluation Management Functions
# ============================================================================

def get_all_evaluators() -> dict:
    """
    Returns a dictionary of all available evaluators with metadata.
    
    Returns:
        dict: Mapping of evaluator names to evaluator functions and their properties
    """
    return {
        "context_relevance": {
            "func": context_relevance_evaluator,
            "requires_retrieval": True,
            "description": "RAGAS context relevance - needs retrieval context",
            "metric_type": "RAGAS"
        },
        "faithfulness": {
            "func": faithfulness_evaluator,
            "requires_retrieval": True,
            "description": "DeepEval GEval faithfulness - needs retrieval context",
            "metric_type": "DeepEval"
        },
        "trajectory_with_ref": {
            "func": trajectory_with_ref_evaluator_func,
            "requires_retrieval": False,
            "description": "Trajectory accuracy with reference",
            "metric_type": "LangChain"
        },
        "trajectory_without_ref": {
            "func": trajectory_without_ref_evaluator_func,
            "requires_retrieval": False,
            "description": "Trajectory accuracy without reference",
            "metric_type": "LangChain"
        },
        "disease_accuracy": {
            "func": disease_accuracy_evaluator,
            "requires_retrieval": False,
            "description": "Disease identification accuracy",
            "metric_type": "DeepEval"
        },
        "goal_with_ref": {
            "func": goal_achievement_with_ref_evaluator,
            "requires_retrieval": False,
            "description": "Goal achievement with reference",
            "metric_type": "DeepEval"
        },
        "goal_without_ref": {
            "func": goal_achievement_without_ref_evaluator,
            "requires_retrieval": False,
            "description": "Goal achievement without reference",
            "metric_type": "DeepEval"
        }
    }


async def run_metrics_sequentially(
    experiment_id: str,
    metrics: List[str] = None,
    max_concurrency: int = 2,
    delay_between_metrics: float = 1.0
):
    """
    Run multiple metrics sequentially on an existing experiment to avoid rate limits.
    
    This function addresses the rate limit issue by:
    1. Running one metric at a time on the existing experiment
    2. Adding delays between metrics to avoid overwhelming the API
    3. Providing progress feedback and error handling
    
    Args:
        experiment_id: The experiment name or UUID from LangSmith
        metrics: List of metric names to run. If None, runs all available metrics.
        max_concurrency: Number of concurrent evaluations per metric run
        delay_between_metrics: Seconds to wait between metric runs
    
    Returns:
        dict: Summary of results for each metric
    
    Example:
        # Run all metrics sequentially
        results = await run_metrics_sequentially(
            "eval-vqa-type1-detailed-20260109-0536-05bd7f69"
        )
        
        # Run specific metrics only
        results = await run_metrics_sequentially(
            "eval-vqa-type1-detailed-20260109-0536-05bd7f69",
            metrics=["faithfulness", "disease_accuracy"]
        )
    """
    import asyncio
    
    all_evaluators = get_all_evaluators()
    
    if metrics is None:
        metrics = list(all_evaluators.keys())
    
    # Validate metrics
    invalid_metrics = [m for m in metrics if m not in all_evaluators]
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}. Available: {list(all_evaluators.keys())}")
    
    print("=" * 80)
    print("Sequential Metric Evaluation")
    print("=" * 80)
    print(f"Experiment: {experiment_id}")
    print(f"Metrics to run: {len(metrics)}")
    print(f"Max concurrency per metric: {max_concurrency}")
    print(f"Delay between metrics: {delay_between_metrics}s")
    print("=" * 80)
    
    results = {}
    
    for i, metric_name in enumerate(metrics, 1):
        print(f"\n[{i}/{len(metrics)}] Running metric: {metric_name}")
        print("-" * 40)
        
        evaluator_func = all_evaluators[metric_name]["func"]
        
        try:
            print(f"  Starting evaluation...")
            
            # Wrap the entire metric run in disable_tracing to avoid concurrency race conditions
            # This ensures tracing is OFF for all concurrent evaluator tasks
            with tracing_context(enabled=False):
                result = await aevaluate_existing(
                    experiment_id,
                    evaluators=[evaluator_func],
                    max_concurrency=max_concurrency,
                )
            
            results[metric_name] = {
                "status": "success",
                "result": result,
                "description": all_evaluators[metric_name]["description"]
            }
            
            print(f"  ✓ {metric_name} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ {metric_name} failed: {error_msg}")
            results[metric_name] = {
                "status": "failed",
                "error": error_msg,
                "description": all_evaluators[metric_name]["description"]
            }
            
            # Continue with next metric instead of stopping
            continue
        
        # Add delay between metrics (except after the last one)
        if i < len(metrics):
            print(f"  Waiting {delay_between_metrics}s before next metric...")
            await asyncio.sleep(delay_between_metrics)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Sequential Evaluation Summary")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"Total metrics attempted: {len(metrics)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    for metric_name, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} {metric_name:25s} - {result['description']}")
        if result["status"] == "failed":
            print(f"    Error: {result['error'][:100]}...")
    
    print("=" * 80)
    print(f"\nView results in LangSmith: https://smith.langchain.com")
    print("Filter by tags: 'evaluation' and 'thesis-agent'")
    print("=" * 80 + "\n")
    
    return results


async def run_full_evaluation_workflow(
    dataset_name: str,
    experiment_name: str,
    description: str,
    max_concurrency: int = 2,
    run_all_metrics_at_once: bool = False
):
    """
    Complete workflow: Run agent + all metrics, with smart metric execution.
    
    This function provides two modes:
    1. Sequential metrics (default): Run agent once, then metrics one-by-one to avoid rate limits
    2. All-at-once: Run agent + all metrics together (may hit rate limits)
    
    Args:
        dataset_name: Name of the LangSmith dataset
        experiment_name: Base name for the experiment
        description: Description of the evaluation
        max_concurrency: Number of concurrent evaluations
        run_all_metrics_at_once: If True, runs all metrics together (may cause rate limits)
    
    Returns:
        dict: Summary of the evaluation workflow
    
    Example:
        # Safe mode (sequential metrics)
        results = await run_full_evaluation_workflow(
            dataset_name="thesis_vqa_type1_detailed",
            experiment_name="eval-vqa-type1-detailed",
            description="Complete evaluation with sequential metrics",
            max_concurrency=4
        )
        
        # Fast mode (all metrics at once - may hit rate limits)
        results = await run_full_evaluation_workflow(
            dataset_name="thesis_vqa_type1_detailed",
            experiment_name="eval-vqa-type1-detailed",
            description="Complete evaluation with all metrics at once",
            max_concurrency=4,
            run_all_metrics_at_once=True
        )
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    experiment_prefix = f"{experiment_name}-{timestamp}"
    
    print("\n" + "=" * 80)
    print("Full Evaluation Workflow")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {experiment_prefix}")
    print(f"Description: {description}")
    print(f"Max Concurrency: {max_concurrency}")
    print(f"Mode: {'All metrics at once' if run_all_metrics_at_once else 'Sequential metrics'}")
    print("=" * 80)
    
    if run_all_metrics_at_once:
        # Original behavior: run all metrics with the agent
        print("\nRunning agent + all metrics together (may hit rate limits)...")
        
        all_evaluators = get_all_evaluators()
        evaluator_funcs = [evaluator["func"] for evaluator in all_evaluators.values()]
        
        result = await aevaluate(
            target,
            data=dataset_name,
            evaluators=evaluator_funcs,
            experiment_prefix=experiment_prefix,
            description=description,
            max_concurrency=max_concurrency,
            metadata={
                "dataset_type": dataset_name,
                "evaluation_timestamp": timestamp,
                "agent_version": "v1",
                "mode": "all_metrics_at_once"
            },
        )
        
        print(f"\nEvaluation complete: {experiment_prefix}")
        print(f"View in LangSmith: https://smith.langchain.com")
        print("=" * 80 + "\n")
        
        return {
            "experiment_id": experiment_prefix,
            "status": "completed",
            "mode": "all_metrics_at_once"
        }
    
    else:
        # Sequential mode: First run agent + minimal metrics, then add others
        print("\nStep 1: Running agent with minimal metrics to establish baseline...")
        
        # First, run agent with just 1-2 metrics to get outputs
        baseline_metrics = ["disease_accuracy"]  # Always run this first
        
        result = await aevaluate(
            target,
            data=dataset_name,
            evaluators=[disease_accuracy_evaluator],  # Minimal set
            experiment_prefix=experiment_prefix,
            description=f"{description} (Baseline)",
            max_concurrency=max_concurrency,
            metadata={
                "dataset_type": dataset_name,
                "evaluation_timestamp": timestamp,
                "agent_version": "v1",
                "mode": "sequential_baseline"
            },
        )
        
        print(f"✓ Baseline complete: {experiment_prefix}")
        print(f"  Agent outputs and basic metrics saved to LangSmith")
        
        # Step 2: Add remaining metrics sequentially
        print("\nStep 2: Adding remaining metrics sequentially...")
        
        remaining_metrics = [
            "context_relevance",
            "faithfulness", 
            "trajectory_with_ref",
            "goal_with_ref"
        ]
        
        await asyncio.sleep(2)  # Brief pause before sequential runs
        
        sequential_results = await run_metrics_sequentially(
            experiment_id=experiment_prefix,
            metrics=remaining_metrics,
            max_concurrency=max_concurrency,
            delay_between_metrics=1.5
        )
        
        print("\n" + "=" * 80)
        print("Workflow Complete!")
        print("=" * 80)
        print(f"Experiment ID: {experiment_prefix}")
        print(f"Baseline metrics: {baseline_metrics}")
        print(f"Additional metrics: {len(remaining_metrics)}")
        print(f"Total metrics: {len(baseline_metrics) + len(remaining_metrics)}")
        print("=" * 80 + "\n")
        
        return {
            "experiment_id": experiment_prefix,
            "status": "completed",
            "mode": "sequential",
            "baseline_metrics": baseline_metrics,
            "additional_metrics": sequential_results
        }




# ============================================================================
# Helper Functions for Experiment Management
# ============================================================================

def list_experiment_metrics(experiment_id: str, client: Client = None):
    """
    List all metrics and their scores for a given experiment.
    
    Args:
        experiment_id: The experiment ID
        client: LangSmith client (uses default if None)
    
    Returns:
        dict: Metric names and their average scores
    """
    if client is None:
        client = Client()
    
    print(f"\nMetrics for experiment: {experiment_id}")
    print("-" * 40)
    
    try:
        # Get feedback from the experiment
        feedback = client.list_feedback(
            project_name=experiment_id,
        )
        
        metric_scores = {}
        for f in feedback:
            metric_name = f.key
            if metric_name not in metric_scores:
                metric_scores[metric_name] = []
            metric_scores[metric_name].append(f.score)
        
        # Calculate averages
        result = {}
        for metric_name, scores in metric_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            result[metric_name] = {
                "average": avg_score,
                "count": len(scores)
            }
            print(f"  {metric_name:30s}: {avg_score:.3f} ({len(scores)} samples)")
        
        return result
        
    except Exception as e:
        print(f"Error retrieving metrics: {e}")
        return {}

async def main():
    """Main execution function."""
    import sys
    from pprint import pprint
    
    print("\n" + "=" * 80)
    print("VQA Agent Evaluation Script")
    print("=" * 80 + "\n")
    
    
    # ========================================================================
    # SANITY CHECK: Test single example before running full evaluation
    # ========================================================================
    SANITY_CHECK = False  # Set to False to skip
    
    if SANITY_CHECK:
        print("\n" + "=" * 80)
        print("SANITY CHECK: Testing single example with separated evaluators")
        print("=" * 80 + "\n")
        
        # Pick a test example
        test_example = TYPE1_DETAILED_EXAMPLES[3]
        print("TEST EXAMPLE:")
        print("=" * 80)
        print(f"Input: {test_example['inputs']['user_text'][:100]}...")
        print(f"Reference Goal: {test_example['outputs'].get('reference_goal', 'N/A')[:100]}...")
        print()
        print(f"Reference Answer: {test_example['outputs'].get('reference_answer', 'N/A')[:100]}...")
        print()
        print(f"Reference Context: {test_example['outputs'].get('reference_context', 'N/A')[:100]}...")
        print()
        
        # Run the agent to get outputs
        print("=" * 80)
        print("RUNNING AGENT...")
        print("=" * 80)
        test_results = await target(test_example["inputs"])
        print(f"Final Answer: {test_results['final_answer'][:150]}...")
        print()
        
        # Test comprehensive evaluator
        print("\n" + "=" * 80)
        print("TESTING COMPREHENSIVE EVALUATOR")
        print("=" * 80)
        comprehensive_results = await evaluator(
            test_example["inputs"],
            test_results,
            test_example["outputs"]
        )
        
        print("\nComprehensive Evaluation Results:")
        print("=" * 80)
        for result in comprehensive_results:
            print(f"{result['key']:35s}: {result['score']:.3f}")
            if result.get("comment"):
                print(f"  Comment: {result['comment'][:100]}...")
        print("=" * 80)
        
        print("\n✓ Sanity check complete! All separated evaluators working.")
        print("=" * 80 + "\n")
        
        return  # Exit after sanity check
    
    print("\n" + "=" * 80)
    print("Running Full Evaluation Suite")
    print("=" * 80 + "\n")
    
    # Setup ID datasets
    setup_datasets("vqa")
    
    # Setup OOD datasets
    # setup_datasets("ood")
    
    # Run evaluations on all dataset types
    print("\nStarting evaluations...")
    
    results = await run_evaluation(
        dataset_name="thesis_vqa_ood_scenario2",
        experiment_name="experiment-vqa-ood-scenario2",
        description="Eksperimen VQA OOD Scenario 2: Hanya Spesies",
        max_concurrency=2
    )
    
    # # 
    try:
        experiment_id = results.experiment_name
    except (AttributeError, KeyError):
        # Fallback: ambil kata terakhir dari string representation (misal: <AsyncExperimentResults name>)
        experiment_id = str(results).strip("<>").split()[-1]
    # experiment_id = "experiment-vqa-scenario2-20260124-0929-8e322968"   
    print(f"\nExtracted Experiment ID: {experiment_id}")  
    seq_results = await run_metrics_sequentially(
        experiment_id=experiment_id,
        metrics=["faithfulness", "goal_with_ref", "context_relevance", "trajectory_with_ref"],
        max_concurrency=2,
        delay_between_metrics=1.5
    )
    
    # seq_results = await run_metrics_sequentially(
    #     experiment_id=experiment_id,
    #     metrics=[ "disease_accuracy"],
    #     max_concurrency=2,
    #     delay_between_metrics=1.5
    # )    
    print("\n" + "=" * 80)
    print("All Evaluations Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys

    asyncio.run(main())
