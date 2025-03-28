from typing import Union, cast

import PIL.Image
import pytesseract
import torch
from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    EmbeddingFunction,
    Embeddings,
    Image,
    Images,
    is_document,
    is_image,
)
from sentence_transformers import SentenceTransformer


class MultimodalEmbeddingFunction(EmbeddingFunction[Union[Documents, Images]]):
    def __init__(self) -> None:
        print("Loading model...")
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __call__(
        self,
        input: Union[Documents, Images],  # pylint: disable=redefined-builtin
    ) -> Embeddings:
        embeddings: Embeddings = []
        for item in input:
            if is_image(item):
                embeddings.append(self._encode_image(cast(Image, item)))
            elif is_document(item):
                embeddings.append(self._encode_text(cast(Document, item)))
        return embeddings

    def _encode_image(self, image: Image) -> Embedding:
        # Convert the image to a PIL image
        pil_image = PIL.Image.fromarray(image)

        # OCR the image using PyTesseract
        text = pytesseract.image_to_string(pil_image)

        # Encode the text using the model
        with torch.no_grad():
            return self._model.encode(text)

    def _encode_text(self, text: Document) -> Embedding:
        with torch.no_grad():
            return self._model.encode(text)
