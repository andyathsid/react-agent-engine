from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langsmith import Client
import os
from dotenv import load_dotenv
load_dotenv()

client = Client()

# Load prompts from LangSmith for full agent
system_prompt_template = client.pull_prompt("thesis-prompt")  

# Placeholder functions for other prompts - these will be loaded from files
def load_prompt_from_file(filename: str) -> str:
    """Load prompt content from a text file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Return a basic fallback prompt if file not found
        return """You are a plant health identification agent focused on plant disease identification from user-provided photos and details.

**Domain scope:**
- You operate exclusively in plant health
- Primary job: identify likely plant disease or issue class from visible symptoms
- Secondary (only if user asks or if needed to avoid wrong disease call): differentiate pest vs nutrient vs abiotic stress; basic plant ID

{status}

**Response style:**
- Be concise and direct
- Focus on observations and evidence
- Acknowledge uncertainty when appropriate
"""

@dynamic_prompt
def get_system_prompt(request: ModelRequest) -> str:
    """Generate system prompt for plant disease identification agent (full tools)."""
    has_image = bool(request.state.get("image_url"))
    status = (
        "Image file is available and tools are ready to use."
        if has_image
        else "Image file is not available, tools cannot be used."
    )
    
    return system_prompt_template.format(status=status)

@dynamic_prompt
def get_system_prompt_no_detection(request: ModelRequest) -> str:
    """Generate system prompt for agent without detection tools."""
    has_image = bool(request.state.get("image_url"))
    status = (
        "Image file is available but detection tools are not available. Use full-image classification only."
        if has_image
        else "Image file is not available, classification tools cannot be used."
    )
    
    prompt_template = load_prompt_from_file("no_detection_prompt.txt")
    return prompt_template.format(status=status)

@dynamic_prompt
def get_system_prompt_no_retrieval(request: ModelRequest) -> str:
    """Generate system prompt for agent without retrieval tools."""
    has_image = bool(request.state.get("image_url"))
    status = (
        "Image file is available but knowledge retrieval tools are not available. Rely on model knowledge only."
        if has_image
        else "Image file is not available, detection tools cannot be used."
    )
    
    prompt_template = load_prompt_from_file("no_retrieval_prompt.txt")
    return prompt_template.format(status=status)

@dynamic_prompt
def get_system_prompt_no_tools(request: ModelRequest) -> str:
    """Generate system prompt for agent with no tools (LLM only)."""
    has_image = bool(request.state.get("image_url"))
    status = (
        "Image file is available but no external tools are available. Rely on model knowledge only."
        if has_image
        else "Image file is not available, no tools available."
    )
    
    prompt_template = load_prompt_from_file("no_tools_prompt.txt")
    return prompt_template.format(status=status)