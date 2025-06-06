"""
LLM Client for GAIA Agent using Groq's API
"""

import os
from typing import Optional
from dotenv import load_dotenv
import groq

from gaia_system_prompt import GAIA_SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Global variables
client = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma2-9b-it")


def initialize_client() -> None:
    """Initialize the Groq client."""
    global client
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set")
    client = groq.Client(api_key=GROQ_API_KEY)


def get_llm_response(
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Get a response from the LLM.

    Args:
        prompt: The input prompt
        temperature: Sampling temperature (0.0 to 1.0)
        max_new_tokens: Maximum number of tokens to generate
        system_prompt: Optional system prompt to use. If None, uses GAIA_SYSTEM_PROMPT.

    Returns:
        The model's response as a string
    """
    if not client:
        initialize_client()

    final_system_prompt = (
        system_prompt if system_prompt is not None else GAIA_SYSTEM_PROMPT
    )

    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens if max_new_tokens else 1000,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error getting LLM response: {str(e)}")


if __name__ == "__main__":
    # Test the client
    try:
        initialize_client()
        test_prompt = "What is 25% of 400?"
        response = get_llm_response(test_prompt, temperature=0.1)
        print(f"Test Prompt: {test_prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Test failed: {e}")
