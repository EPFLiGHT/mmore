import re
from typing import Optional

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....type import MultimodalSample


class MultimodalEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        super().__init__()
        float16: torch.dtype = torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = self.model.device

        """Interface for embedding models.

        This is an interface meant for implementing text embedding models.

        Text embedding models are used to map text to a vector (a point in n-dimensional
        space).

        Texts that are similar will usually be mapped to points that are close to each
        other in this space. The exact details of what's considered "similar" and how
        "distance" is measured in this space are dependent on the specific embedding model.

        This abstraction contains a method for embedding a list of documents and a method
        for embedding a query text. The embedding of a query text is expected to be a single
        vector, while the embedding of a list of documents is expected to be a list of
        vectors.

        Usually the query embedding is identical to the document embedding, but the
        abstraction allows treating them independently.

        In addition to the synchronous methods, this interface also provides asynchronous
        versions of the methods.

        By default, the asynchronous methods are implemented using the synchronous methods;
        however, implementations may choose to override the asynchronous methods with
        an async native implementation for performance reasons.
        """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        embeddings = []

        for text in texts:
            # replace <attachment> with empty string
            text = text.replace("<attachment>", "")
            prompt, images = MultimodalEmbeddings._extract_multimodal_inputs(
                text, proc_token="<|image|>"
            )
            images = [Image.open(image) for image in images]

            if images:
                inputs = self.processor(
                    text=prompt, images=images, return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(
                    self.device
                )

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[0]
                embedding = last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
                embedding = embedding.cpu().numpy().squeeze(0).astype(np.float32)
                embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.embed_documents([text])[0]

    @staticmethod
    def _multimodal_to_text(sample: MultimodalSample):
        s = "\n".join(
            [
                f"<|{modality.type}|>{modality.value}<|{modality.type}|>"
                for modality in sample.modalities
            ]
        )
        s += sample.text
        return s

    @staticmethod
    def _multimodal_to_doc(sample: MultimodalSample) -> Document:
        return Document(
            MultimodalEmbeddings._multimodal_to_text(sample), metadata=sample.metadata
        )

    @staticmethod
    def _extract_multimodal_inputs(
        text, proc_token: str, pattern: Optional[str] = None
    ) -> tuple[str, list[str]]:
        pattern = pattern or rf"{re.escape(proc_token)}(.*?){re.escape(proc_token)}"

        # Find all matches in the input string
        extracted_strings = re.findall(pattern, text)

        # Replace all matches with an empty string
        cleaned_string = re.sub(pattern, proc_token, text)

        # Return a tuple with the cleaned string and the list of extracted strings
        return (cleaned_string, extracted_strings)
