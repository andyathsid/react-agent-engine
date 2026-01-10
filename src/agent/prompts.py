from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langsmith import Client

client = Client()

system_prompt_template = client.pull_prompt("thesis")  

@dynamic_prompt
def get_system_prompt(request: ModelRequest) -> str:
    """Generate system prompt for plant disease identification agent."""
    has_image = bool(request.state.get("image_url"))
    status = (
        "Image file is available and tools are ready to use."
        if has_image
        else "Image file is not available, tools cannot be used."
    )

    
    return system_prompt_template.format(status=status)