"""
Azure OpenAI clients for chat and embeddings.
"""

from openai import AzureOpenAI
#from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, CHAT_MODEL, EMB_MODEL
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_VERSION = os.getenv("API_VERSION")
OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME")
OPENAI_EMBED_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBED_DEPLOYMENT_NAME")


def create_azure_openai_client():
    """
    Create and return an Azure OpenAI client.
    """
    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT
    )
    return client


def get_embedding_client():
    """
    Create and return an Azure OpenAI embeddings client configured for LangChain.
    
    Returns:
        AzureOpenAIEmbeddings: A configured embedding client
    """
    from langchain_openai import AzureOpenAIEmbeddings
    
    embedding_client = AzureOpenAIEmbeddings(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        deployment=OPENAI_EMBED_DEPLOYMENT_NAME,
        model=OPENAI_EMBED_DEPLOYMENT_NAME,
    )
    
    return embedding_client


def get_chat_completion(client, messages, temperature=0.5):
    """
    Get a chat completion from Azure OpenAI.
    
    Args:
        client: Azure OpenAI client
        messages: List of message dictionaries
        temperature: Temperature for response generation
        
    Returns:
        Generated response text
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_DEPLOYMENT_NAME,
            messages=messages,
            temperature=temperature
        )
        content = response.choices[0].message.content
        
        # Handle None response
        if content is None:
            return "I apologize, but I'm having trouble generating a response right now. Please continue."
            
        return content
        
    except Exception as e:
        print(f"[ERROR] Azure OpenAI call failed: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please continue."


def get_embeddings(client, texts):
    """
    Get embeddings for a list of texts from Azure OpenAI.
    
    Args:
        client: Azure OpenAI client
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    response = client.embeddings.create(
        model=OPENAI_EMBED_DEPLOYMENT_NAME,
        input=texts
    )
    return [item.embedding for item in response.data]
