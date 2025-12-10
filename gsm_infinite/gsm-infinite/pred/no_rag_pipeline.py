"""NoRAG pipeline implementation."""

from typing import Dict, Any, List, Optional, Tuple, Union
# from ..base.pipeline import BasePipeline
from model_handler import ModelHandler
from concurrent.futures import ThreadPoolExecutor

class NoRAGPipeline():
    """Pipeline that sends entire context to model without retrieval."""

    def __init__(self, model_handler: ModelHandler, temperature: float = 0.0, max_tokens=512):
        """Initialize pipeline with model handler and temperature.

        Args:
            model_handler: ModelHandler instance
            temperature: Temperature for model sampling (default 0.0 for consistent outputs)
        """
        self.model_handler = model_handler
        self.temperature = temperature
        self.config = {'max_tokens': max_tokens}  # Default configuration

    def process_single(self, query: str, contexts: str) -> str:
        """Process a single query with given context.

        Args:
            query: Question string
            context: Context string

        Returns:
            Model response string
        """
        response = self.get_llm_response(query)
        return response

    def _validate_batch_inputs(self, queries: List[str], contexts: List[str]) -> None:
        if len(queries) != len(contexts):
            raise ValueError("Number of queries must match number of contexts")
        
    def process_batch_with_executor(self, queries: List[str], contexts: List[str], max_workers=200) -> List[str]:
        self._validate_batch_inputs(queries, contexts)
        responses = []
        
        # Check if the progress bar is enabled and if the number of queries is greater than 2
        show_progress_bar = self.config.get('progress_bar', True) and len(queries) > 2

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for query, context in zip(queries, contexts):
                future = executor.submit(self.process_single, query, context)
                futures.append(future)

            if show_progress_bar:
                from tqdm import tqdm
                for future in tqdm(futures, total=len(queries), desc="Processing queries"):
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as e:
                        responses.append(f"Error: {str(e)}")
            else:
                for future in futures:
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as e:
                        responses.append(f"Error: {str(e)}")

        return responses

    def process_batch(self, queries: List[str], contexts: List[str]=None, max_workers=200) -> List[str]:
        """Process a batch of queries with corresponding contexts.

        Args:
            queries: List of question strings
            contexts: List of context strings

        Returns:
            List of model responses
        """
        if contexts is None:
            contexts = queries
        self._validate_batch_inputs(queries, contexts)
        return self.process_batch_with_executor(queries, contexts, max_workers)

    def get_llm_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Get response from LLM with token probabilities.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic output)
            **kwargs: Additional arguments to pass to the model handler

        Returns:
            Tuple of (response text, token probabilities, tokens)
        """
        try:
            if temperature is None:
                temperature = self.temperature

            if max_tokens is None:
                max_tokens = self.config.get('max_tokens', 1000)

            response = self.model_handler.generate_answer(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"Error in get_llm_response: {str(e)}")
            raise
