import os
import math
from typing import List, Tuple, Optional, Any, Dict, Union
# import torch
from openai import OpenAI
import google.generativeai as genai
import anthropic



def retry_with_exponential_backoff(
    func,
    initial_delay: float = 5,
    exponential_base: float = 1.5,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        import random
        import time
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except Exception as e:
                print(e, flush=True)
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)


    return wrapper

class ModelHandler:
    SUPPORTED_BACKENDS = ['openai', 'vllm', 'sglang', 'gemini', 'anthropic']

    def __init__(self, model_name: str = "gpt-4o-mini", backend_type: str = "openai"):
        if backend_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        self.model_type = backend_type
        self.model_name = model_name
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if self.model_type == "openai":
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not found in environment")
            self.client = OpenAI(api_key=self.api_key)
        elif self.model_type == "gemini":
            self.api_key = os.getenv('GEMINI_API_KEY')
            if not self.api_key:
                raise ValueError("GEMINI API key not found in environment")
            genai.configure(api_key=self.api_key)
        elif self.model_type == "anthropic":
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC API key not found in environment")
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
            )
        else:
            raise NotImplementedError("not implemented")

    @retry_with_exponential_backoff
    def generate_answer(
        self,
        prompt: Union[str, List[str]],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        if self.model_type == "openai":
            return self._get_openai_response(prompt, **kwargs)
        elif self.model_type == "gemini":
            return self._get_gemini_response(prompt, **kwargs)
        elif self.model_type == "anthropic":
            return self._get_anthropic_response(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported backend type: {self.model_type}")

    def _get_openai_response(self, prompt: Union[str, List[str]],
                            max_tokens: int = 4096,
                            temperature = None,
                            **kwargs) -> str:
        messages = [
            # {"role": "system", "content": "You are a helpful assistant focused on providing accurate and detailed responses."},
            {"role": "user", "content": prompt}
        ] if type(prompt) == str else prompt
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=600.0,  # 300 second timeout for individual requests
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content.strip()


    def _get_gemini_response(self, prompt: Union[str, List[str]],
                            max_tokens: int = 4096,
                            temperature = None,
                            **kwargs) -> str:
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
                "response_mime_type": "text/plain",
            }

            messages = [
                {"role": "user", "content": prompt}
            ] if type(prompt) == str else prompt

            def convert_chat_history(openai_messages):
                """
                Converts OpenAI-style chat history to Gemini-style chat history.

                Args:
                    openai_messages: A list of dictionaries representing the OpenAI chat history.

                Returns:
                    A tuple containing the Gemini-style chat history (list of dictionaries) and the system prompt (string).
                """
                gemini_history = []
                system_prompt = None

                role_map = {"assistant": "model", "user": "user"}

                for message in openai_messages[:-1]:
                    if message["role"] == "system":
                        system_prompt = message["content"]
                    else:
                        gemini_history.append(
                            {
                                "role": role_map[message["role"]],
                                "parts": [message["content"]],
                            }
                        )
                user_input = openai_messages[-1]["content"]

                return gemini_history, system_prompt, user_input
            
            gemini_history, system_prompt, user_input = convert_chat_history(messages)

            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                system_instruction=system_prompt
            )

            chat_session = model.start_chat(
                history=gemini_history
            )

            response = chat_session.send_message(user_input)
            return response.text
        except Exception as e:
            print(f"Error in Gemini response: {str(e)}")
            raise

    def _get_anthropic_response(self, prompt: Union[str, List[str]],
                            max_tokens: int = 4096,
                            temperature = None,
                            **kwargs) -> str:
        try:
            messages = [
                {"role": "user", "content": prompt}
            ] if type(prompt) == str else prompt

            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                timeout=600.0,  # 600 second timeout for individual requests
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error in Anthropic response: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'client') and self.client:
            if hasattr(self.client, 'close'):
                self.client.close()
            self.client = None

def main():
    prompt = "What is the capital of France? Please describe it."

    # try:
    handler = ModelHandler(model_name="gpt-4o-mini", backend_type="openai")
    # handler = ModelHandler(model_name="gemini-2.0-flash-exp", backend_type="gemini")
    # handler = ModelHandler(model_name="claude-3-5-haiku-20241022", backend_type="anthropic")
    ret = handler.generate_answer(
        prompt,
        max_tokens=4000,
        temperature=0.7,
    )
    print(f"OpenAI Response: {ret}")

if __name__ == "__main__":
    main()
