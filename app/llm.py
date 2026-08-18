import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from app.config import PROJECT_ROOT, LLM_TEMP

load_dotenv(PROJECT_ROOT / ".env")

def setup_openai_client():
    """
    Set up a vLLM based chat client.

    Returns:
        ChatOpenAI: The configured chat client.
    """

    print(f'Trying to connect to model hosted on {os.environ["OPENAI_BASE_URL"]}...')

    if os.getenv("OPENAI_SELECTED_MODEL") == "auto":
        
        # auto-select the first available model from the OpenAI API
        client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

        models = client.models.list()
        model_name = models.data[0].id
    
    else:
        model_name = os.getenv("OPENAI_SELECTED_MODEL")
        
    print(f"Using model: {model_name}")

    client = ChatOpenAI(
        model=model_name,
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=LLM_TEMP,
    )

    return client


chat_model = setup_openai_client()