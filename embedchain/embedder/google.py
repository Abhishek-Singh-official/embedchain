from typing import Optional, Union

from google import genai
from chromadb import EmbeddingFunction, Embeddings

from embedchain.config.embedder.google import GoogleAIEmbedderConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.models import VectorDimensions

import os

class GoogleAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, config: Optional[GoogleAIEmbedderConfig] = None) -> None:
        super().__init__()
        self.config = config or GoogleAIEmbedderConfig()

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Union[list[str], str]) -> Embeddings:
        model = self.config.model
        title = self.config.title
        task_type = self.config.task_type
        vector_dimension = self.config.vector_dimension

        if isinstance(input, str):
            input_ = [input]
        else:
            input_ = input

        response = self.client.models.embed_content(
            model=model,
            contents=input_,
            config={
                "task_type": task_type,
                "title": title,
                "output_dimensionality": vector_dimension,
            }
        )

        embeddings = response.embeddings[0].values

        return embeddings


class GoogleAIEmbedder(BaseEmbedder):
    def __init__(self, config: Optional[GoogleAIEmbedderConfig] = None):
        super().__init__(config)
        embedding_fn = GoogleAIEmbeddingFunction(config=config)
        self.set_embedding_fn(embedding_fn=embedding_fn)

        vector_dimension = self.config.vector_dimension or VectorDimensions.GOOGLE_AI.value
        self.set_vector_dimension(vector_dimension=vector_dimension)
