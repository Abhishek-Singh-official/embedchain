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

from sarvamai import SarvamAI

logger = logging.getLogger(__name__)

########## THIS CODE IS MODIFIED CODE FOR SARVAM AI CHAT COMPLITION 09/05/2026


# ============================================================================
# CLIENT MANAGEMENT
# ============================================================================

# Client instance (singleton)
_sarvam_client: Optional[SarvamAI] = None

def _get_sarvam_client() -> SarvamAI:
    """
    Initializes and returns singleton SARVAM client with lazy initialization.
    
    Uses global caching to prevent repeated initialization. Connection is
    established lazily on first use, checking API key validity immediately.
    
    Returns:
        Initialized SARVAM AI client ready for async operations
        
    Raises:
        RuntimeError: If API key is not configured or client initialization fails
        
    Example:
        client = _get_sarvam_client()
        response = client.chat.completions(...)
    """
    global _sarvam_client

    if _sarvam_client is None:
        try:
            # Initialize with API key from environment/config
            _sarvam_client = SarvamAI(api_subscription_key="sk_ipb7a89p_Pv1Qog2Nnjtl3PpCvkTnav4c")
            logger.info("[INTENT] SARVAM AI client initialized (sarvam-30b, async-ready)")
        except Exception as e:
            logger.error(
                f"[INTENT] Failed to initialize SARVAM client: {e}",
                exc_info=True
            )
            raise RuntimeError(f"SARVAM client initialization failed: {e}") from e
    
    return _sarvam_client

@register_deserializable
class GoogleLlm(BaseLlm):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        # if not self.config.api_key and "GOOGLE_API_KEY" not in os.environ:
        #     raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it in the config.")

        # api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        # self.client = genai.Client(api_key=api_key)

    def get_llm_model_answer(self, prompt):
        # if self.config.system_prompt:
        #     raise ValueError("GoogleLlm does not support `system_prompt`")
        response = self._get_answer(prompt)
        return response

    def _get_answer(self, prompt: str) -> Union[str, Generator[Any, Any, None]]:
        try:
            # Step 1: Get SARVAM client (with validation)
            client = _get_sarvam_client()

            # Step 4: Call SARVAM API with tool forcing (structured outputs)
            response = client.chat.completions(
                    model="sarvam-30b",
                    messages=[
                        {
                            "role": "system",
                            "content": self.config.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=self.config.temperature or 0.4,
                    max_tokens=self.config.max_tokens,
                    reasoning_effort="low",
                )
        except Exception as e:
            logger.info(f"Error during query process: {e}")
        
        return  response.choices[0].message.content

        # if self.config.stream:
        #     return self.client.models.generate_content_stream(
        #     model=model_name,
        #     contents=prompt,
        #     config=generation_config,
        # )
        # else:
        #     response = self.client.models.generate_content(
        #         model=model_name,
        #         contents=prompt,
        #         config=generation_config,
        #     )
        #     return response.text












####### BELLOW CODE IS THE ORIGINAL CODE - ABOVE CODE IS MODIFY FOR SARVAR AI FOR CHAT COMPITION - 09/05/2026



# import logging
# import os
# from collections.abc import Generator
# from typing import Any, Optional, Union

# try:
#     from google import genai
# except ImportError:
#     raise ImportError("GoogleLlm requires extra dependencies. Install with `pip install google-generativeai`") from None

# from google.genai import types
# from embedchain.config import BaseLlmConfig
# from embedchain.helpers.json_serializable import register_deserializable
# from embedchain.llm.base import BaseLlm

# logger = logging.getLogger(__name__)


# @register_deserializable
# class GoogleLlm(BaseLlm):
#     def __init__(self, config: Optional[BaseLlmConfig] = None):
#         super().__init__(config)
#         if not self.config.api_key and "GOOGLE_API_KEY" not in os.environ:
#             raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it in the config.")

#         api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
#         self.client = genai.Client(api_key=api_key)

#     def get_llm_model_answer(self, prompt):
#         if self.config.system_prompt:
#             raise ValueError("GoogleLlm does not support `system_prompt`")
#         response = self._get_answer(prompt)
#         return response

#     def _get_answer(self, prompt: str) -> Union[str, Generator[Any, Any, None]]:
#         model_name = self.config.model or "gemini-pro"
#         # logger.info(f"Using Google LLM model: {model_name}")

#         generation_config = types.GenerateContentConfig(
#             candidate_count = 1,
#             max_output_tokens = self.config.max_tokens,
#             temperature = self.config.temperature or 0.5,
#             top_p=self.config.top_p if (0.0 < self.config.top_p <= 1.0) else None
#         )

#         response = self.client.models.generate_content(
#             model=model_name,
#             contents=prompt,
#             config=generation_config,
#         )
        
#         return response.text

#         # if self.config.stream:
#         #     return self.client.models.generate_content_stream(
#         #     model=model_name,
#         #     contents=prompt,
#         #     config=generation_config,
#         # )
#         # else:
#         #     response = self.client.models.generate_content(
#         #         model=model_name,
#         #         contents=prompt,
#         #         config=generation_config,
#         #     )
#         #     return response.text
