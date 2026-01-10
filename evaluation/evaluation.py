#!/usr/bin/env python3
"""
Plant Disease Identification Agent - Evaluation Suite

This module provides comprehensive evaluation metrics for the plant disease identification agent,
with full support for multimodal (text + image) evaluation and smart metric management to avoid rate limits.

MULTIMODAL EVALUATION FEATURES:
- All DeepEval metrics now support MLLMImage objects for direct image evaluation
- LLM judges can see and analyze the actual images when evaluating agent performance
- Robust handling of both image-present and text-only scenarios
- Automatic fallback to text-only evaluation if image loading fails

KEY METRICS:
1. Faithfulness (DeepEval GEval) - Evaluates if agent's claims are grounded in evidence
2. Disease Accuracy (DeepEval DAG) - Evaluates correctness of disease identification
3. Goal Achievement (With/Without Reference) - Evaluates if agent achieved user's goal
4. Trajectory Accuracy (With/Without Reference) - Evaluates tool usage efficiency
5. Context Relevance (RAGAS) - Evaluates quality of retrieved context

SMART METRIC MANAGEMENT (NEW):
To avoid rate limits when running multiple metrics, this module provides:

1. SEQUENTIAL METRIC EXECUTION:
   - Run metrics one-by-one on existing experiments
   - Automatic delays between metric runs
   - Error handling with continuation

2. BATCH PROCESSING:
   - Process multiple experiments with specified metrics
   - Configurable concurrency and delays

3. ANALYSIS TOOLS:
   - Analyze experiments to recommend appropriate metrics
   - Identify missing metrics
   - Check experiment health

USAGE EXAMPLES:

# Add metrics to existing experiment (avoids rate limits)
await run_metrics_sequentially(
    experiment_id="your-experiment-id",
    metrics=["faithfulness", "context_relevance"],
    max_concurrency=2,
    delay_between_metrics=1.5
)

# Complete workflow with sequential metrics
await run_full_evaluation_workflow(
    dataset_name="thesis_vqa_type1_detailed",
    experiment_name="my-eval",
    description="Safe evaluation",
    max_concurrency=4,
    run_all_metrics_at_once=False  # Sequential mode
)

# Command line usage:
python evaluation.py --analyze-exp "experiment-id"
python evaluation.py --add-metrics "experiment-id" --metrics faithfulness context_relevance
python evaluation.py --full-workflow "dataset-name" --name "experiment-name" --sequential

Each metric automatically handles:
- Missing image_url fields (falls back to text-only)
- Invalid or inaccessible images (logs warning and continues)
- Empty or malformed inputs (validation and normalization)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, List
import json
import json
import ast
import asyncio
from dotenv import load_dotenv

# Setup paths
load_dotenv()

project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))  # Add project root for data module
sys.path.insert(0, str(project_root / "src"))  # Add src for agent module

# LangChain/LangSmith
from langsmith import Client, aevaluate
from langsmith.evaluation import aevaluate_existing
import langchain_core.messages as lc
from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from openevals.llm import create_llm_as_judge
from openevals.prompts import HALLUCINATION_PROMPT

# RAGAS
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import Faithfulness, ContextRelevance
import ragas.messages as r

# DeepEval
from deepeval.models import GeminiModel, GPTModel
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
    FaithfulnessMetric,
)

# Agent
from agent.graph import graph as agent_graph

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


def load_dataset_from_json():
    """Load dataset examples from JSON file with metadata using streaming."""
    json_path = project_root / "data" / "langsmith" / "vqa_identification_examples.json"
    
    # Filter by prompt type using streaming
    type1 = []
    type2 = []
    type3 = []
    
    for ex in iter_dataset(json_path):
        prompt_type = ex.get("metadata", {}).get("prompt_type", "")
        if prompt_type in ["vague_symptoms", "condition_description"]:
            type1.append(ex)
        elif prompt_type == "species_only":
            type2.append(ex)
        elif prompt_type == "minimal":
            type3.append(ex)
    
    return type1, type2, type3

TYPE1_DETAILED_EXAMPLES, TYPE2_SPECIES_EXAMPLES, TYPE3_MINIMAL_EXAMPLES = load_dataset_from_json()

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

deepeval_model_gpt = GPTModel(
    model="gpt-4o",
    temperature=0
)

deepeval_model_gpt4o = GPTModel(
    model="gpt-4o-mini-2024-07-18",
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


def setup_datasets():
    """Setup all evaluation datasets."""
    print("=" * 80)
    print("Setting up LangSmith datasets...")
    print("=" * 80)
    
    datasets = {}
    
    datasets['type1'] = load_examples_to_dataset(
        ls_client,
        "thesis_vqa_type1_detailed",
        TYPE1_DETAILED_EXAMPLES,
        "Type 1: Detailed/descriptive queries (vague_symptoms + condition_description)"
    )
    
    datasets['type2'] = load_examples_to_dataset(
        ls_client,
        "thesis_vqa_type2_species",
        TYPE2_SPECIES_EXAMPLES,
        "Type 2: Species-only queries (both diseased & healthy)"
    )
    
    datasets['type3'] = load_examples_to_dataset(
        ls_client,
        "thesis_vqa_type3_minimal",
        TYPE3_MINIMAL_EXAMPLES,
        "Type 3: Minimal queries (both diseased & healthy)"
    )
    
    print("\nDataset Summary:")
    print(f"  - Type 1 (Detailed): {len(TYPE1_DETAILED_EXAMPLES)} examples")
    print(f"  - Type 2 (Species): {len(TYPE2_SPECIES_EXAMPLES)} examples")
    print(f"  - Type 3 (Minimal): {len(TYPE3_MINIMAL_EXAMPLES)} examples")
    print("=" * 80 + "\n")
    
    return datasets

# ============================================================================
# Individual Evaluator Functions
# Each metric is now separated into its own evaluator function
# ============================================================================

# 1. RAGAS Context Relevance Evaluator
async def context_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate context relevance using RAGAS with multimodal support."""
    # Validate inputs
    inputs = validate_and_normalize_inputs(inputs)
    
    user_text = inputs.get("user_text", "")
    image_url = inputs.get("image_url")
    
    # Create multimodal input for RAGAS evaluation
    if image_url:
        user_question = f"[Image provided - agent will perform visual observation] {user_text}"
    else:
        user_question = user_text
    
    trace_messages = outputs.get("trace_messages", {})
    retrieved_context = _extract_retrieval_context_from_trace(trace_messages)
    
    if not retrieved_context or len(retrieved_context) == 0:
        return [{"key": "context_relevance", "score": 0.0, "comment": "No retrieval context available"}]
    
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
    # Validate inputs
    inputs = validate_and_normalize_inputs(inputs)
    
    user_text = inputs.get("user_text", "")
    image_url = inputs.get("image_url")
    
    # Create multimodal input for trajectory evaluation
    if image_url:
        user_question = f"[Image provided - agent will perform visual observation] {user_text}"
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
    # Validate inputs
    inputs = validate_and_normalize_inputs(inputs)
    
    user_text = inputs.get("user_text", "")
    image_url = inputs.get("image_url")
    
    # Create multimodal input for trajectory evaluation
    if image_url:
        user_question = f"[Image provided - agent will perform visual observation] {user_text}"
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
        expected_disease.endswith("healthy")
    ) and not expected_pathogen
    
    try:
        # Handle case where no image is provided
        image_url = inputs.get("image_url")
        if not image_url:
            # Use text-only test case
            if is_healthy_case:
                disease_test_case = LLMTestCase(
                    input=test_case.input,
                    actual_output=final_answer,
                    expected_output=f"Pathogen Type: None (healthy plant)\nDisease: HEALTHY PLANT ({expected_disease})\nReference: {reference_answer}"
                )
            else:
                disease_test_case = LLMTestCase(
                    input=test_case.input,
                    actual_output=final_answer,
                    expected_output=f"Pathogen Type: {expected_pathogen}\nDisease: {expected_disease}\nReference: {reference_answer}"
                )
        else:
            # Use multimodal test case (already created in test_case.input)
            # Override expected_output and actual_output to ensure they're correct
            test_case.actual_output = final_answer
            if is_healthy_case:
                test_case.expected_output = f"Pathogen Type: None (healthy plant)\nDisease: HEALTHY PLANT ({expected_disease})\nReference: {reference_answer}"
            else:
                test_case.expected_output = f"Pathogen Type: {expected_pathogen}\nDisease: {expected_disease}\nReference: {reference_answer}"
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
Based on agent and expected outputs:
Did the agent appropriately abstain (express uncertainty/ask for info) when evidence is genuinely ambiguous/insufficient?
Answer Yes (0.25) or No (0.0).
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
            criteria="""
Based on agent and expected outputs:
Is the agent's stated or implied pathogen type (fungal/bacterial/viral/pest) correct?
If agent made no specific claim about pathogen type, answer No.
Answer Yes or No.
""",
            evaluation_params=[
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            children=[
                VerdictNode(verdict=False, child=abstention_appropriate),
                VerdictNode(verdict=True, score=SCORE_0_5),
            ],
        )

        disease_partial_match = BinaryJudgementNode(
            criteria="""
Based on agent and expected outputs:
Is the expected disease mentioned in agent's differential (top 2-3 candidates) OR is the disease category/pattern strongly consistent?
If yes AND pathogen type is correct, this leads to 0.75.
Answer Yes or No.
""",
            evaluation_params=[
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            children=[
                VerdictNode(verdict=False, child=pathogen_type_check),
                VerdictNode(verdict=True, score=SCORE_0_75),
            ],
        )

        primary_disease_correct = BinaryJudgementNode(
            criteria="""
Based on agent and expected outputs:
Is the agent's PRIMARY disease identification exactly correct (allowing minor naming variants)?
The expected disease should be extracted from the METADATA section.

Answer Yes or No.
""",
            evaluation_params=[
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            children=[
                VerdictNode(verdict=False, child=disease_partial_match),
                VerdictNode(verdict=True, score=SCORE_1_0),
            ],
        )

        healthy_plant_correct = BinaryJudgementNode(
        criteria="""
Look at the METADATA section in expected_output for the "is_healthy" field:

If is_healthy: True (HEALTHY PLANT case):
  - Agent should correctly identify the plant as HEALTHY/no disease/no symptoms present as PRIMARY conclusion
  - Agent may provide preventive care advice (this is appropriate)
  - Agent may mention tool results with low confidence or ask clarifying questions (this is appropriate)
  - Answer Yes if agent's primary conclusion is healthy, even when mentioning low-confidence possibilities
  - Answer No only if agent definitively states a specific disease as the primary diagnosis

If is_healthy: False (DISEASE case):
  - This is NOT a healthy case, answer No to proceed to disease identification check

Answer Yes (1.0 for correct healthy identification) or No (proceed to disease check).
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
            model=deepeval_model_gpt,
            threshold=0.5,
            include_reason=True,
            verbose_mode=False,
        )
        
        # DAG metric only supports GPT models, use GPT-only fallback
        score = measure_with_fallback(deepeval_disease_accuracy, disease_test_case, fallback_models=[deepeval_model_gpt4o])
        return [{"key": "disease_accuracy", "score": score, "comment": deepeval_disease_accuracy.reason}]
    except Exception as e:
        return [{"key": "disease_accuracy", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 6. Goal Achievement - With Reference Evaluator
async def goal_achievement_with_ref_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate goal achievement with reference using DeepEval GEval with multimodal support."""
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
    Evaluate whether the agent successfully achieved the reference goal based on the actual output and tool usage.
    
    The agent is designed for plant disease identification with these capabilities:
    1. Disease identification from images and user descriptions
    2. Management/treatment guidance when explicitly requested
    3. Symptom analysis and differential diagnosis
    4. Appropriate uncertainty expression when evidence is insufficient
    5. Refusal when requests are outside scope (non-plant images, medical advice, etc.)
    
    GOAL ACHIEVEMENT CRITERIA:
    - Full Achievement (0.9-1.0): Agent completely fulfilled the reference goal with high quality, using appropriate tools and grounding responses in tool outputs
    - Partial Achievement (0.5-0.8): Agent addressed the main objective but missed key components, lacked depth, or failed to effectively utilize tool outputs
    - Minimal Achievement (0.2-0.4): Agent attempted the goal but with significant gaps, errors, or poor tool usage
    - No Achievement (0.0-0.1): Agent failed to address the goal or went completely off-track
    
    IMPORTANT CONSIDERATIONS:
    - Examine both the final output AND the tools called to understand the full agent behavior
    - If goal was "identify disease": check if identification was attempted with reasonable quality and whether detection/classification tools were used appropriately
    - If goal included "provide management advice": check if advice was given and grounded in retrieval tool outputs (knowledgebase_search, web_search)
    - If goal was "refuse inappropriate request": check if agent properly refused with explanation, avoiding unnecessary tool calls
    - Tool outputs provide crucial context - verify the agent's response aligns with and appropriately uses information from tool results
    - Appropriate uncertainty is acceptable and should not be heavily penalized
    - Asking clarifying questions to achieve the goal is acceptable
    - MULTIMODAL CONSIDERATIONS: If image provided, verify agent made appropriate visual observations and used image information correctly
    """,
        evaluation_steps=[
            "Extract the reference goal from the expected_output field (formatted as 'REFERENCE_GOAL: ...')",
            "Identify the main objectives stated in the reference goal (e.g., identify disease, provide treatment, refuse request)",
            "Examine the tools_called field to understand what tools the agent invoked and what outputs they returned",
            "Examine the actual output to determine what the agent actually communicated to the user",
            "Check if the agent used appropriate tools to achieve the reference goal (e.g., detection tools for identification, knowledgebase_search for guidance)",
            "Verify that the agent's actual output properly incorporates and reflects information from tool outputs",
            "If image was provided in input, verify agent made appropriate visual observations and used image information",
            "Check if each main objective from the reference goal was addressed in the actual output",
            "Evaluate the quality and completeness of how each objective was fulfilled",
            "For identification goals: verify if disease/issue was identified with supporting evidence from tool outputs AND visual analysis",
            "For guidance goals: verify if management advice was provided and grounded in retrieval tool outputs",
            "For refusal goals: verify if agent properly declined with appropriate explanation without unnecessary tool usage",
            "Consider whether appropriate uncertainty or clarifying questions helped achieve the goal",
            "Determine the overall achievement level: full, partial, minimal, or none"
        ],
        rubric=[
            Rubric(
                score_range=(0, 1),
                expected_outcome="No achievement - agent completely failed to address the reference goal, went entirely off-track, or used tools inappropriately without incorporating their outputs."
            ),
            Rubric(
                score_range=(2, 4),
                expected_outcome="Minimal achievement - agent attempted the goal but with significant gaps, errors, missing major components, or poor utilization of tool outputs."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Partial achievement - agent addressed the main objective and used tools appropriately, but missed key components, lacked depth, had minor errors, or didn't fully leverage tool outputs."
            ),
            Rubric(
                score_range=(9, 10),
                expected_outcome="Full achievement - agent completely fulfilled the reference goal with high quality, using appropriate tools and effectively grounding the response in tool outputs."
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
        return [{"key": "agent_goal_accurac", "score": score, "comment": deepeval_goal_achievement_with_ref.reason}]
    except Exception as e:
        return [{"key": "agent_goal_accurac", "score": 0.0, "comment": f"Error: {str(e)}"}]


# 7. Goal Achievement - Without Reference Evaluator
async def goal_achievement_without_ref_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate goal achievement without reference using DeepEval GEval with multimodal support."""
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
    Evaluate whether the agent appropriately addressed the user's request based on the input and actual output.
    
    The agent is designed for plant disease identification with these capabilities:
    1. Disease identification from images and user descriptions
    2. Management/treatment guidance when explicitly requested
    3. Symptom analysis and differential diagnosis
    4. Appropriate uncertainty expression when evidence is insufficient
    5. Refusal when requests are outside scope (non-plant images, medical advice, etc.)
    
    GOAL ACHIEVEMENT CRITERIA:
    - Full Achievement (0.9-1.0): Agent completely fulfilled what the user requested with high quality
    - Partial Achievement (0.5-0.8): Agent addressed the main request but missed components or lacked depth
    - Minimal Achievement (0.2-0.4): Agent attempted to help but with significant gaps or errors
    - No Achievement (0.0-0.1): Agent failed to address the request or went completely off-track
    
    IMPORTANT CONSIDERATIONS:
    - Infer the user's intent from the input (identification-only vs guidance-requested)
    - If user asked "what is this?": goal is disease identification
    - If user asked "how to treat/manage/fix": goal includes identification + management advice
    - If input is non-plant or off-topic: goal is appropriate refusal
    - Appropriate uncertainty is acceptable and should not be heavily penalized
    - Asking clarifying questions to better serve the user is acceptable
    - MULTIMODAL CONSIDERATIONS: If image provided, verify agent made appropriate visual observations and used image information correctly
    """,
        evaluation_steps=[
            "Analyze the user input to determine what the user requested (identification, guidance, both, or other)",
            "Identify whether the request was appropriate (plant-related) or should have been refused",
            "Examine the actual output to determine what the agent actually did",
            "Check if the agent addressed the user's main request",
            "Evaluate the quality and completeness of the response",
            "If image was provided in input, verify agent made appropriate visual observations and used image information",
            "For identification requests: verify if disease/issue was identified with supporting evidence AND visual analysis",
            "For guidance requests: verify if management advice was provided and grounded",
            "For inappropriate requests: verify if agent properly declined",
            "Consider whether appropriate uncertainty or clarifying questions served the user well",
            "Determine the overall achievement level: full, partial, minimal, or none"
        ],
        rubric=[
            Rubric(
                score_range=(0, 1),
                expected_outcome="No achievement - agent completely failed to address the user's request or went entirely off-track."
            ),
            Rubric(
                score_range=(2, 4),
                expected_outcome="Minimal achievement - agent attempted to help but with significant gaps, errors, or missing major components."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Partial achievement - agent addressed the main request but missed components, lacked depth, or had minor issues."
            ),
            Rubric(
                score_range=(9, 10),
                expected_outcome="Full achievement - agent completely fulfilled the user's request with high quality and completeness."
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


# ============================================================================
# Metrics Configuration (Preserved for backward compatibility)
# ============================================================================

# RAG Metrics (RAGAS)
# ragas_faithfulness = Faithfulness(llm=ragas_llm_flash)
ragas_context_relevance = ContextRelevance(llm=ragas_llm_pro)

# Custom Faithfulness Metric for Plant Health Agent (DeepEval GEval)
deepeval_plant_faithfulness = GEval(
    name="Plant Agent Faithfulness",
    model=deepeval_model_pro,
    criteria="""
    Evaluate whether the agent's output is faithful to the available evidence, accounting for the agent's specific capabilities and domain.
    
    AGENT CAPABILITIES & EXPECTED BEHAVIORS:
    1. Visual Observations: Agent SHOULD make observations from images (symptoms, patterns, colors, locations)
    2. General Plant Care Advice: Agent CAN provide general best practices (isolation, sanitation, airflow, pruning, consulting extension services) WITHOUT retrieval
    3. Disease-Specific Claims: Agent MUST ground disease identification and disease-specific management in retrieved context or user-provided information
    4. Uncertainty Expression: Agent SHOULD express appropriate uncertainty when evidence is limited
    
    FAITHFULNESS EVALUATION CRITERIA:
    
    ALLOWED (NOT hallucinations):
    - Visual observations from images: "The leaves show circular spots," "yellow halos visible," "lesions on underside," "wet appearance"
    - General plant care recommendations: "improve airflow," "remove infected material," "isolate the plant," "sanitize tools," "consult local extension service"
    - User-provided context integration: Referencing plant species, timeline, or conditions the user mentioned
    - Appropriate uncertainty: "likely," "appears to be," "consistent with," "could indicate"
    - Standard domain knowledge: Basic plant biology, common symptom patterns, general care practices
    
    REQUIRES RETRIEVAL SUPPORT (would be hallucination if not supported):
    - Disease identification: Naming specific diseases must be based on symptoms observed + retrieved knowledge
    - Disease-specific treatment: Chemical recommendations, specific fungicides/pesticides, application timing
    - Specific lifecycle information: Overwintering stages, infection timelines, environmental triggers
    - Regional/cultivar specifics: Variety resistance, regional prevalence, local recommendations
    - Product names or specific formulations: Copper sulfate concentrations, brand names, active ingredients
    
    SCORING RUBRIC:
    - 0.9-1.0 (Highly Faithful): All claims appropriately grounded; visual observations clear; general advice reasonable; disease-specific claims supported
    - 0.7-0.8 (Mostly Faithful): Minor unsupported details but core claims grounded; no significant misrepresentations
    - 0.5-0.6 (Partially Faithful): Some disease-specific claims lack support OR contains speculative details presented as fact
    - 0.3-0.4 (Largely Unfaithful): Multiple unsupported disease-specific claims OR significant fabrication of details
    - 0.0-0.2 (Unfaithful): Predominantly fabricated information; disease claims not grounded; invented specific recommendations
    
    IMPORTANT NOTES:
    - The agent works with IMAGES - if image provided, visual observations are EXPECTED and VALID
    - General care advice is part of the agent's domain expertise - does NOT require retrieval
    - Only flag disease-specific claims (disease names, specific treatments, specific biology) if unsupported
    - Recommending "consult extension service" is standard best practice - NOT a hallucination
    - Focus on SUBSTANTIVE hallucinations, not stylistic choices or reasonable inferences
    """,
    evaluation_steps=[
        "Identify all visual observations in the actual output (descriptions of what's visible in images)",
        "Identify all general plant care recommendations (sanitation, isolation, airflow, consulting experts, etc.)",
        "Identify all disease-specific claims (disease names, specific treatments, specific pathogen biology)",
        "Check if visual observations are reasonable for an image-analyzing agent (these are VALID)",
        "Check if general care recommendations are domain-appropriate best practices (these are VALID)",
        "For each disease-specific claim, check if it's supported by retrieval_context or user input",
        "Identify any fabricated specific details (invented chemicals, made-up timelines, false specifics)",
        "Determine if the output appropriately expresses uncertainty where evidence is limited",
        "Count the number and severity of unsupported disease-specific claims",
        "Assign a faithfulness score based on the balance of grounded vs. ungrounded specific claims"
    ],
    rubric=[
        Rubric(
            score_range=(0, 2),
            expected_outcome="Unfaithful - predominantly fabricated disease-specific information; multiple unsupported disease claims; invented specific recommendations without retrieval support."
        ),
        Rubric(
            score_range=(3, 4),
            expected_outcome="Largely unfaithful - several significant disease-specific claims lack retrieval support; specific treatments or biology details are fabricated or unverified."
        ),
        Rubric(
            score_range=(5, 6),
            expected_outcome="Partially faithful - some disease-specific claims lack support OR speculative details presented as fact; core identification may be grounded but supporting details are not."
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Mostly faithful - disease identification and major claims are grounded in retrieval or user context; only minor unsupported details present; appropriate use of visual observations and general advice."
        ),
        Rubric(
            score_range=(9, 10),
            expected_outcome="Highly faithful - all disease-specific claims appropriately grounded in retrieval_context or user input; visual observations are clear and reasonable; general care advice is standard best practice; appropriate uncertainty expressed."
        ),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.TOOLS_CALLED],
    threshold=0.5,
    verbose_mode=False,
)

# Trajectory Accuracy - With Reference (LangChain AgentEval)
TRAJECTORY_WITH_REF_PROMPT = """You are an expert evaluator for a plant disease identification agent.

Your task is to evaluate whether the agent's tool usage trajectory is a VALID SUPERSET of the reference trajectory.

<Agent Design Principles>
The agent is designed for plant disease identification with these workflow principles:
1. Visual identification first (plant_disease_identification) when clear plant image provided
2. Optional detection preprocessing (closed_set_leaf_detection or open_set_object_detection) is ALWAYS acceptable
3. Knowledge retrieval (knowledgebase_search or web_search) required when user requests guidance/treatment
4. Fallback to knowledge_base/web_search if visual tools return low confidence or incorrect results
</Agent Design Principles>

<Superset Semantics>
- Agent MUST call all tools in the reference trajectory (or semantically equivalent alternatives)
- Agent MAY call ADDITIONAL tools if appropriate:
  * Detection tools (closed_set_leaf_detection, open_set_object_detection) are ALWAYS acceptable extras
  * web_search is acceptable if knowledgebase_search was expected (or vice versa) - both are retrieval
  * Fallback tools are acceptable when primary tools show uncertainty
- Tool order may vary if logical (e.g., detection before classification)
</Superset Semantics>

<Knowledge Retrieval Analysis>
- If reference has knowledgebase_search OR web_search: agent should call at least ONE retrieval tool
- Both tools serve the same purpose (knowledge retrieval) and are interchangeable in most cases
- Calling BOTH is acceptable for cross-validation or when KB is insufficient
</Knowledge Retrieval Analysis>

<DO NOT Penalize>
- Adding detection tools before classification
- Using web_search instead of knowledgebase_search (or vice versa)
- Calling extra validation tools when uncertainty is high
- Different parameter values for the same tool
</DO NOT Penalize>

<Evaluation Steps>
1. Extract the tool names from the reference trajectory (reference_tool_calls)
2. Extract the tool names from the actual trajectory
3. Check if ALL reference tools are present in actual trajectory (allowing knowledgebase_search ↔ web_search substitution)
4. Identify any EXTRA tools in actual trajectory not in reference
5. Evaluate if extra tools are justified: detection tools are always OK, retrieval tools are OK for validation/fallback
6. Assess logical flow: tools should be in sensible order based on user request and image complexity
7. Determine if the trajectory achieves the same goal as reference with acceptable or beneficial additions
</Evaluation Steps>

<Rubric>
Score 0.0:
- Missing critical tools - reference tools are completely absent (e.g., no identification tool when reference expects it)

Score 0.2:
- Missing multiple key tools - some reference tools present but missing essential steps (e.g., missing retrieval when user asked for treatment)

Score 0.5:
- Has all reference tools but with problematic additions - includes unnecessary or illogical extra tool calls that don't align with agent design

Score 0.75:
- Valid superset with minor issues - all reference tools present, extra tools are mostly justified, minor logical flow issues

Score 1.0:
- Perfect superset - all reference tools present (or acceptable substitutions), any extra tools are beneficial (detection, validation, fallback), logical flow
</Rubric>

<Reference Trajectory>
{reference_tool_calls}
</Reference Trajectory>

<User Input>
{input}
</User Input>

<Actual Trajectory>
{outputs}
</Actual Trajectory>

Provide your score (0.0, 0.2, 0.5, 0.75, or 1.0) and detailed reasoning.
"""

# Trajectory Accuracy - Without Reference (LangChain AgentEval)
TRAJECTORY_WITHOUT_REF_PROMPT = """You are an expert evaluator for a plant disease identification agent.

Your task is to evaluate the logical flow and efficiency of the agent's tool usage trajectory based on design principles.

<Agent Design Principles>
The agent is designed for plant disease identification with these principles:
1. Vision-first: Start with visual triage before tool calls
2. Visual identification (plant_disease_identification) when image is clear plant tissue
3. Optional detection preprocessing (closed_set_leaf_detection, open_set_object_detection) for complex scenes
4. Knowledge retrieval MANDATORY when user requests guidance/treatment/management
5. Fallback to knowledge_base or web_search when primary tools show low confidence (<0.5) or contradiction
6. Cross-validate outputs when uncertainty is high
7. Skip tools on non-plant images or unclear photos
</Agent Design Principles>

<Acceptable Variations>
- Detection tools before classification (for cluttered scenes, multiple leaves)
- web_search vs knowledgebase_search (both are retrieval, interchangeable)
- Multiple retrieval calls for cross-validation
- Rejecting low-confidence tool outputs
</Acceptable Variations>

<Penalty-Worthy Issues>
- Calling tools on non-plant images
- No retrieval when user asks "how to treat" or "what to do"
- Not using fallback when primary tool fails
- Excessive redundant calls (same tool, same parameters, no new information)
- Accepting contradictory outputs without validation
</Penalty-Worthy Issues>

<Evaluation Steps>
1. Identify the user's request type: identification-only vs guidance-requested (treatment, management, prevention)
2. Check if image is present and contains plant tissue (based on agent's observations)
3. Evaluate tool selection: is plant_disease_identification called when appropriate?
4. Check if detection tools are used appropriately (complex scenes) or skipped (simple images)
5. If user requested guidance: verify knowledgebase_search OR web_search was called
6. Assess fallback behavior: does agent use retrieval when visual tools show low confidence or errors?
7. Check for redundancy: are there unnecessary duplicate tool calls?
8. Evaluate logical flow: sensible tool ordering and evidence-based decision making
</Evaluation Steps>

<Rubric>
Score 0.0:
- Critical failure - calls tools on non-plant images, provides guidance without retrieval when required, or completely illogical trajectory

Score 0.2:
- Poor trajectory - missing mandatory tools (retrieval for guidance requests), no fallback for failed tools, or many redundant calls

Score 0.5:
- Acceptable - reaches correct conclusion but with inefficiencies (redundant calls, suboptimal ordering, minor logical issues)

Score 0.75:
- Good trajectory - mostly logical flow, appropriate tool selection, minor inefficiencies (e.g., one unnecessary call), proper fallback usage

Score 1.0:
- Excellent trajectory - optimal tool progression, efficient path, appropriate fallback/validation, follows all agent design principles
</Rubric>

<User Input>
{input}
</User Input>

<Actual Trajectory>
{outputs}
</Actual Trajectory>

First, identify the user's goal from the input. Then, evaluate whether the trajectory efficiently and logically achieves that goal given the agent's design principles.

Provide your score (0.0, 0.2, 0.5, 0.75, or 1.0) and detailed reasoning.
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
                            # Add plant name
                            if "plant_name" in metadata:
                                context_parts.append(f"Plant: {metadata['plant_name']}")
                            
                            # Add document type/section
                            if "section" in metadata:
                                context_parts.append(f"Section: {metadata['section']}")
                            elif "doc_type" in metadata:
                                context_parts.append(f"Type: {metadata['doc_type']}")
                            
                            # Add content type
                            if "content_type" in metadata:
                                context_parts.append(f"Content Type: {metadata['content_type']}")
                            
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
    
    # Detect healthy plant cases: class ends with "leaf" and no pathogen type
    is_healthy_case = (
        expected_disease.endswith("leaf") or 
        expected_disease.endswith("healthy")
    ) and not expected_pathogen_type
    
    # Combine reference answer with structured metadata for DAG parsing
    if is_healthy_case:
        expected_output = f"""{reference_answer}

METADATA:
- expected_disease: HEALTHY PLANT ({expected_disease})
- expected_pathogen_type: None (healthy plant)
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
            # Create MLLMImage object
            # Note: local=False assumes URL, local=True assumes local file path
            # We'll try to detect if it's a local path or URL
            is_local = not (image_url.startswith('http://') or image_url.startswith('https://'))
            image = MLLMImage(url=image_url, local=is_local)
            
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
        "image_url": image_url,
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
        fallback_models = [deepeval_model_gpt, deepeval_model_flash]
    
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
    
    result = await aevaluate(
        target,
        data=dataset_name,
        evaluators=[goal_achievement_with_ref_evaluator, faithfulness_evaluator],
        experiment_prefix=experiment_prefix,
        description=description,
        max_concurrency=max_concurrency,
        metadata={
            "dataset_type": dataset_name,
            "evaluation_timestamp": timestamp,
            "agent_version": "v1",
        },
    )
    
    print(f"\nEvaluation complete: {experiment_prefix}")
    print(f"View in LangSmith: https://smith.langchain.com")
    print("=" * 80 + "\n")
    
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


async def batch_process_experiments(
    experiment_ids: List[str],
    metrics: List[str] = None,
    max_concurrency: int = 2,
    delay_between_experiments: float = 5.0
):
    """
    Process multiple existing experiments with specified metrics.
    
    Useful for when you have several experiments that need additional metrics added.
    
    Args:
        experiment_ids: List of experiment IDs to process
        metrics: List of metrics to run on each experiment
        max_concurrency: Concurrent evaluations per metric
        delay_between_experiments: Seconds to wait between experiments
    
    Returns:
        dict: Results for each experiment
    
    Example:
        experiments = [
            "eval-vqa-type1-detailed-20260109-0536-05bd7f69",
            "eval-vqa-type2-species-20260109-0600-12345678"
        ]
        
        results = await batch_process_experiments(
            experiments,
            metrics=["faithfulness", "context_relevance"]
        )
    """
    print("=" * 80)
    print("Batch Process Experiments")
    print("=" * 80)
    print(f"Experiments: {len(experiment_ids)}")
    print(f"Metrics per experiment: {metrics if metrics else 'All'}")
    print(f"Max concurrency: {max_concurrency}")
    print("=" * 80)
    
    all_results = {}
    
    for i, exp_id in enumerate(experiment_ids, 1):
        print(f"\n[{i}/{len(experiment_ids)}] Processing: {exp_id}")
        print("-" * 40)
        
        try:
            results = await run_metrics_sequentially(
                experiment_id=exp_id,
                metrics=metrics,
                max_concurrency=max_concurrency,
                delay_between_metrics=1.0
            )
            
            all_results[exp_id] = {
                "status": "success",
                "metrics": results
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            all_results[exp_id] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Wait before next experiment
        if i < len(experiment_ids):
            print(f"\n  Waiting {delay_between_experiments}s before next experiment...")
            await asyncio.sleep(delay_between_experiments)
    
    # Summary
    print("\n" + "=" * 80)
    print("Batch Processing Summary")
    print("=" * 80)
    
    successful = sum(1 for r in all_results.values() if r["status"] == "success")
    failed = sum(1 for r in all_results.values() if r["status"] == "failed")
    
    print(f"Experiments processed: {len(experiment_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("=" * 80 + "\n")
    
    return all_results


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


def find_missing_metrics(experiment_id: str, client: Client = None):
    """
    Find which metrics are missing from an experiment.
    
    Args:
        experiment_id: The experiment ID
        client: LangSmith client
    
    Returns:
        list: Names of metrics that are missing
    """
    if client is None:
        client = Client()
    
    all_evaluators = get_all_evaluators()
    existing_metrics = list_experiment_metrics(experiment_id, client)
    
    missing = []
    for metric_name in all_evaluators.keys():
        if metric_name not in existing_metrics:
            missing.append(metric_name)
    
    if missing:
        print(f"\nMissing metrics for {experiment_id}:")
        for metric in missing:
            print(f"  - {metric}")
    else:
        print(f"\nAll metrics present for {experiment_id}")
    
    return missing


def recommend_metrics_for_experiment(experiment_id: str, client: Client = None):
    """
    Recommend which metrics to run based on experiment data analysis.
    
    This function analyzes an experiment to determine which metrics are appropriate
    based on whether retrieval context exists, reference data is available, etc.
    
    Args:
        experiment_id: The experiment ID
        client: LangSmith client
    
    Returns:
        dict: Recommended metrics and their reasoning
    """
    if client is None:
        client = Client()
    
    print(f"\nAnalyzing experiment: {experiment_id}")
    print("-" * 40)
    
    # Get some examples from the experiment to analyze
    try:
        runs = list(client.list_runs(
            project_name=experiment_id,
            limit=5  # Check first 5 runs
        ))
        
        if not runs:
            print("No runs found in experiment")
            return {}
        
        has_retrieval = False
        has_reference_goal = False
        has_reference_tools = False
        
        for run in runs:
            # Check for retrieval context in outputs
            if hasattr(run, 'outputs') and run.outputs:
                outputs = run.outputs
                if isinstance(outputs, dict):
                    # Check for trace messages that might contain retrieval
                    trace_messages = outputs.get("trace_messages", {})
                    if trace_messages:
                        # Simple heuristic: check if there are any retrieval-related keys
                        if any("retriev" in str(k).lower() for k in trace_messages.keys()):
                            has_retrieval = True
            
            # Check for reference data in inputs
            if hasattr(run, 'reference_example_id') and run.reference_example_id:
                try:
                    example = client.read_example(run.reference_example_id)
                    if hasattr(example, 'outputs') and example.outputs:
                        ref_outputs = example.outputs
                        if isinstance(ref_outputs, dict):
                            if ref_outputs.get("reference_goal"):
                                has_reference_goal = True
                            if ref_outputs.get("reference_tool_calls"):
                                has_reference_tools = True
                except:
                    pass
        
        print(f"  Has retrieval context: {has_retrieval}")
        print(f"  Has reference goal: {has_reference_goal}")
        print(f"  Has reference tool calls: {has_reference_tools}")
        
        # Recommend metrics based on analysis
        recommendations = {}
        
        # Always recommended
        recommendations["disease_accuracy"] = {
            "reason": "Core metric - always applicable",
            "priority": "high"
        }
        
        if has_reference_goal:
            recommendations["goal_with_ref"] = {
                "reason": "Reference goal available",
                "priority": "high"
            }
        else:
            recommendations["goal_without_ref"] = {
                "reason": "No reference goal - use without reference version",
                "priority": "medium"
            }
        
        if has_reference_tools:
            recommendations["trajectory_with_ref"] = {
                "reason": "Reference tool calls available",
                "priority": "medium"
            }
        else:
            recommendations["trajectory_without_ref"] = {
                "reason": "No reference tools - use without reference version",
                "priority": "low"
            }
        
        if has_retrieval:
            recommendations["context_relevance"] = {
                "reason": "Retrieval context detected",
                "priority": "high"
            }
            recommendations["faithfulness"] = {
                "reason": "Retrieval context detected",
                "priority": "high"
            }
        
        print("\nRecommended metrics:")
        for metric, info in recommendations.items():
            print(f"  {metric:25s} [{info['priority']:6s}] - {info['reason']}")
        
        return recommendations
        
    except Exception as e:
        print(f"Error analyzing experiment: {e}")
        return {}

# ============================================================================
# Main Execution
# ============================================================================

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
    
    # Setup datasets
    setup_datasets()
    
    # Run evaluations on all dataset types
    print("\nStarting evaluations...")
    
    # # Type 1: Detailed/descriptive queries
    # print("\n" + "=" * 60)
    # print("Type 1: Detailed/Descriptive Queries")
    # print("=" * 60)
    # exp_type1 = await run_evaluation(
    #     dataset_name="thesis_vqa_type1_detailed",
    #     experiment_name="eval-vqa-type1-detailed",
    #     description="Evaluation on Type 1: Detailed/descriptive queries (vague symptoms + condition description)",
    #     max_concurrency=4,
    # )
    
    # ========================================================================
    # EXAMPLES: How to use the new evaluation management functions
    # ========================================================================
    
    # Example 1: Run a complete evaluation with sequential metrics (SAFE MODE)
    # This avoids rate limits by running metrics one-by-one after the agent
    # results = await run_full_evaluation_workflow(
    #     dataset_name="thesis_vqa_type3_minimal",
    #     experiment_name="eval-vqa-type2-minimal",
    #     description="Thesis evaluation on Type 3: minimal queries",
    #     max_concurrency=2,
    #     run_all_metrics_at_once=False  # Sequential mode
    # )
    
    # Example 2: Run all metrics at once (FAST MODE - may hit rate limits)
    # results = await run_full_evaluation_workflow(
    #     dataset_name="thesis_vqa_type1_detailed",
    #     experiment_name="eval-vqa-type1-detailed",
    #     description="Complete evaluation with all metrics at once",
    #     max_concurrency=4,
    #     run_all_metrics_at_once=True  # All at once mode
    # )
    
    # Example 3: Add metrics to an existing experiment (SEQUENTIAL)
    # This is what you described - running metrics one-by-one on existing data
    # results = await run_metrics_sequentially(
    #     experiment_id="eval-vqa-type1-detailed-20260109-0536-05bd7f69",
    #     metrics=["faithfulness", "context_relevance", "disease_accuracy"],
    #     max_concurrency=2,
    #     delay_between_metrics=1.5
    # )
    
    # Example 4: Process multiple experiments at once
    # experiments = [
    #     "eval-vqa-type1-detailed-20260109-0536-05bd7f69",
    #     "eval-vqa-type2-species-20260109-0600-12345678"
    # ]
    # results = await batch_process_experiments(
    #     experiment_ids=experiments,
    #     metrics=["faithfulness", "context_relevance"],
    #     max_concurrency=2
    # )
    
    # Example 5: Check what metrics are missing from an experiment
    # missing = find_missing_metrics("eval-vqa-type1-detailed-20260109-0536-05bd7f69")
    # print(f"Missing metrics: {missing}")
    
    # Example 6: Continue an incomplete experiment (original functionality)
    # exp_result = await continue_evaluation(
    #     experiment_id="eval-vqa-type1-detailed-20260109-0536-05bd7f69",
    #     description="Continuing incomplete experiment",
    #     max_concurrency=4
    # )
    
    # ========================================================================
    # ACTIVE EXAMPLE: Run this to test the new sequential evaluation
    # ========================================================================
    
    # print("\n" + "=" * 80)
    # print("TESTING NEW SEQUENTIAL EVALUATION FUNCTIONS")
    # print("=" * 80 + "\n")
    
    # # First, let's run a quick test to get an experiment ID
    # print("Step 1: Creating a test experiment with baseline metrics...")
    
    # # Run a small evaluation to create an experiment
    # test_result = await run_evaluation(
    #     dataset_name="thesis_vqa_type1_detailed",
    #     experiment_name="test-sequential-workflow",
    #     description="Test experiment for sequential metric evaluation",
    #     max_concurrency=2
    # )
    
    # # Get the experiment ID from the result
    # print("Variable results type:\n")
    # print(type(results))
    # print()
    # print("Variable results value:\n")
    # print(results)
    experiment_id = "eval-vqa-type2-minimal-20260110-0657-9384b4e8"  # This should give us the experiment name
    
    print(f"\n✓ Test experiment created: {experiment_id}")
    print("\nStep 2: Adding metrics sequentially to avoid rate limits...")
    experiment_name=experiment_id
    async def rerun_examples(
        ls_client,
        target,
        experiment_name=experiment_name,
    ):
        print("\n" + "=" * 80)
        print("Rerunning Failed Examples")
        print("=" * 80 + "\n")
        # 1) find failed root runs
        failed_runs = list(
            ls_client.list_runs(
                project_name=experiment_name,
                is_root=True,
                error=True,
                select=["reference_example_id"],
            )
        )

        # 2) collect + dedupe example ids
        failed_example_ids = sorted({
            r.reference_example_id
            for r in failed_runs
            if getattr(r, "reference_example_id", None) is not None
        })

        if not failed_example_ids:
            return {"message": "No failed examples found."}

        # 3) fetch Examples
        failed_examples = list(ls_client.list_examples(example_ids=failed_example_ids))

        # 4) extend the existing experiment/project
        exp = ls_client.read_project(project_name=experiment_name)

        result = await aevaluate(
            target,
            data=failed_examples,
            client=ls_client,
            evaluators=[
                disease_accuracy_evaluator,
            ],
            max_concurrency=4,
            experiment=exp.id,                 
            metadata={"rerun": "true"},         
            # error_handling="ignore",          # optional
        )
        return result
    
    # exp_result = await rerun_examples(ls_client, target)
    
    # # Now add metrics sequentially
    sequential_results = await run_metrics_sequentially(
        experiment_id=experiment_name,
        metrics=["faithfulness", "context_relevance", "goal_with_ref", "trajectory_with_ref"],
        max_concurrency=2,
        delay_between_metrics=1.5
    )
    
    # print("\n" + "=" * 80)
    # print("Sequential Evaluation Test Complete!")
    # print("=" * 80)
    # print(f"Experiment ID: {experiment_id}")
    # print("You can now view the results in LangSmith")
    # print("=" * 80 + "\n")


    
    # Type 2: Species-only queries
    # print("\n" + "=" * 60)
    # print("Type 2: Species-Only Queries")
    # print("=" * 60)
    # exp_type2 = await run_evaluation(
    #     dataset_name="thesis_vqa_type2_species",
    #     experiment_name="eval-vqa-type2-species",
    #     description="Evaluation on Type 2: Species-only queries",
    #     max_concurrency=10,
    # )
    
    # Type 3: Minimal queries
    # print("\n" + "=" * 60)
    # print("Type 3: Minimal Queries")
    # print("=" * 60)
    # exp_type3 = await run_evaluation(
    #     dataset_name="thesis_vqa_type3_minimal",
    #     experiment_name="eval-vqa-type3-minimal",
    #     description="Evaluation on Type 3: Minimal queries (brief, no species mentioned)",
    #     max_concurrency=8,
    # )
    
    # print("\n" + "=" * 60)
    # print("Continuing Incomplete Experiment: eval-vqa-type2-species")
    # print("=" * 60)
    # exp_continue = await continue_evaluation(
    #     experiment_id="eval-vqa-type2-species-20260108-1215-0dccab9d",
    #     description="Continuing incomplete experiment for Type 2: Species-only queries",
    #     max_concurrency=8,
    # )
    
    print("\n" + "=" * 80)
    print("All Evaluations Complete!")
    print("=" * 80)
    print("\nView results in LangSmith: https://smith.langchain.com")
    print("Filter by tags: 'evaluation' and 'thesis-agent'")
    print("=" * 80 + "\n")


# ============================================================================
# Command Line Interface
# ============================================================================

async def cli_mode():
    """
    Command-line interface for common evaluation tasks.
    
    Usage:
        python evaluation.py --help
        python evaluation.py --analyze-exp "experiment-id"
        python evaluation.py --add-metrics "experiment-id" --metrics faithfulness context_relevance
        python evaluation.py --full-workflow "dataset-name" --name "experiment-name"
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plant Disease Agent Evaluation Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze an existing experiment
  python evaluation.py --analyze-exp eval-vqa-type1-detailed-20260109-0536-05bd7f69
  
  # Add specific metrics to an experiment
  python evaluation.py --add-metrics eval-vqa-type1-detailed-20260109-0536-05bd7f69 --metrics faithfulness context_relevance
  
  # Run full evaluation workflow (sequential mode)
  python evaluation.py --full-workflow thesis_vqa_type1_detailed --name my-eval --sequential
  
  # Run full evaluation workflow (all-at-once mode)
  python evaluation.py --full-workflow thesis_vqa_type1_detailed --name my-eval --all-at-once
        """
    )
    
    parser.add_argument("--analyze-exp", type=str, help="Analyze an experiment and recommend metrics")
    parser.add_argument("--add-metrics", type=str, help="Add metrics to existing experiment")
    parser.add_argument("--metrics", nargs="+", help="List of metrics to add")
    parser.add_argument("--full-workflow", type=str, help="Run full evaluation workflow on dataset")
    parser.add_argument("--name", type=str, help="Experiment name for full workflow")
    parser.add_argument("--sequential", action="store_true", help="Use sequential metrics (safe mode)")
    parser.add_argument("--all-at-once", action="store_true", help="Use all metrics at once (fast mode)")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrency (default: 2)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between metrics in seconds (default: 1.5)")
    
    args = parser.parse_args()
    
    if not any([args.analyze_exp, args.add_metrics, args.full_workflow]):
        parser.print_help()
        return
    
    client = Client()
    
    # Analyze experiment
    if args.analyze_exp:
        print(f"\nAnalyzing experiment: {args.analyze_exp}")
        recommend_metrics_for_experiment(args.analyze_exp, client)
        missing = find_missing_metrics(args.analyze_exp, client)
        if missing:
            print(f"\nSuggested command to add missing metrics:")
            print(f"  python evaluation.py --add-metrics {args.analyze_exp} --metrics {' '.join(missing)}")
    
    # Add metrics to existing experiment
    elif args.add_metrics:
        if not args.metrics:
            print("Error: --metrics required when using --add-metrics")
            return
        
        print(f"\nAdding metrics to experiment: {args.add_metrics}")
        print(f"Metrics: {args.metrics}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Delay: {args.delay}s")
        
        results = await run_metrics_sequentially(
            experiment_id=args.add_metrics,
            metrics=args.metrics,
            max_concurrency=args.concurrency,
            delay_between_metrics=args.delay
        )
        
        print("\nResults:")
        for metric, result in results.items():
            status = "✓" if result["status"] == "success" else "✗"
            print(f"  {status} {metric}: {result['status']}")
    
    # Full workflow
    elif args.full_workflow:
        if not args.name:
            print("Error: --name required for full workflow")
            return
        
        if not (args.sequential or args.all_at_once):
            print("Error: Choose --sequential or --all-at-once mode")
            return
        
        mode = "sequential" if args.sequential else "all-at-once"
        print(f"\nRunning full workflow:")
        print(f"  Dataset: {args.full_workflow}")
        print(f"  Experiment: {args.name}")
        print(f"  Mode: {mode}")
        print(f"  Concurrency: {args.concurrency}")
        
        results = await run_full_evaluation_workflow(
            dataset_name=args.full_workflow,
            experiment_name=args.name,
            description=f"CLI workflow - {mode} mode",
            max_concurrency=args.concurrency,
            run_all_metrics_at_once=args.all_at_once
        )
        
        print(f"\nWorkflow complete: {results['experiment_id']}")


async def quick_add_metrics():
    """
    Quick helper for the user's specific use case: adding metrics to existing experiments.
    
    This is the function you can call directly when you have an experiment ID
    and want to add metrics one-by-one to avoid rate limits.
    """
    # Example usage - replace with your actual experiment ID
    experiment_id = "eval-vqa-type1-detailed-20260109-0536-05bd7f69"
    
    print(f"\nQuick metric addition for: {experiment_id}")
    print("This will run metrics one-by-one to avoid rate limits")
    print("=" * 60)
    
    # First, check what's missing
    missing = find_missing_metrics(experiment_id)
    
    if not missing:
        print("No missing metrics found!")
        return
    
    # Run missing metrics sequentially
    results = await run_metrics_sequentially(
        experiment_id=experiment_id,
        metrics=missing,
        max_concurrency=2,
        delay_between_metrics=1.5
    )
    
    print("\nQuick addition complete!")
    return results


if __name__ == "__main__":
    import sys
    
    # Check if we're in CLI mode or main mode
    if len(sys.argv) > 1 and sys.argv[1] in ["--analyze-exp", "--add-metrics", "--full-workflow", "--help"]:
        asyncio.run(cli_mode())
    else:
        # Run the main evaluation script
        asyncio.run(main())
