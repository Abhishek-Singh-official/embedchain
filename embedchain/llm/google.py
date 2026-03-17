import logging
import os
from collections.abc import Generator
from typing import Any, Optional, Union

try:
    from google import genai
except ImportError:
    raise ImportError("GoogleLlm requires extra dependencies. Install with `pip install google-generativeai`") from None

from google.genai import types
from embedchain.config import BaseLlmConfig
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.llm.base import BaseLlm

logger = logging.getLogger(__name__)


@register_deserializable
class GoogleLlm(BaseLlm):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        if not self.config.api_key and "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it in the config.")

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def get_llm_model_answer(self, prompt):
        if self.config.system_prompt:
            raise ValueError("GoogleLlm does not support `system_prompt`")
        response = self._get_answer(prompt)
        return response

    def _get_answer(self, prompt: str) -> Union[str, Generator[Any, Any, None]]:
        model_name = self.config.model or "gemini-pro"
        # logger.info(f"Using Google LLM model: {model_name}")

        generation_config = types.GenerateContentConfig(
            candidate_count = 1,
            max_output_tokens = self.config.max_tokens,
            temperature = self.config.temperature or 0.5,
            top_p=self.config.top_p if (0.0 < self.config.top_p <= 1.0) else None
        )

        if self.config.stream:
            return self.client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=generation_config,
        )
        else:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )
            return response.text
