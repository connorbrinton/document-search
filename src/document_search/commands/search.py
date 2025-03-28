import hashlib
from pathlib import Path
from typing import Annotated

import chromadb
import gradio
import numpy as np
import platformdirs
import typer
from datasets import load_dataset
from diskcache import Cache
from rich import print  # pylint: disable=redefined-builtin
from tqdm.auto import tqdm

from document_search.embedding.embedding_function import MultimodalEmbeddingFunction


def main() -> None:
    """Launch the command based on the arguments in `sys.argv`."""
    typer.run(_cli)


def _cli() -> None:
    """Launch the search application on the OmniAI dataset."""
    # Create a ChromaDB instance and image collection
    print("Setting up ChromaDB...")
    db = chromadb.Client()
    collection = db.create_collection(
        "documents",
        embedding_function=MultimodalEmbeddingFunction(),
    )

    # Download the images from the OmniAI dataset
    print("Loading dataset...")
    dataset = load_dataset("getomni-ai/ocr-benchmark")["test"]

    # Initialize a cache to store the embeddings across invocations of this command
    cache_directory = platformdirs.user_cache_dir(
        appname="image-search",
        appauthor="connorbrinton",
        ensure_exists=True,
    )
    cache = Cache(Path(cache_directory) / "embeddings.db")

    # Index the images by their ID
    images_by_id = {example["id"]: example["image"] for example in dataset}

    # Index all of the images in the directory
    for image_id, image in tqdm(images_by_id.items(), desc="Indexing images"):
        # Load the image into memory
        image_array = np.asarray(image)

        # Hash the image into a key for our cache
        image_hash = hashlib.blake2b(image_array.tobytes()).digest()

        # Attempt to use a cached embedding
        if image_hash in cache:
            # Use the cached embedding
            collection.add(ids=[str(image_id)], embeddings=[cache[image_hash]])
        else:
            # No cached embedding is available, so compute a new one
            collection.add(ids=[str(image_id)], images=[image_array])

            # Retrieve and store the computed embedding for the image
            embedding = collection.get(ids=[str(image_id)], include=["embeddings"])[
                "embeddings"
            ][0]
            cache[image_hash] = embedding

    # Define the Gradio dashboard render function
    print("Launching Gradio dashboard...")
    with gradio.Blocks() as dashboard:
        input_text = gradio.Textbox(label="Query", placeholder="HR documents")

        @gradio.render(inputs=input_text)
        def search_images(query: str) -> None:
            results = collection.query(query_texts=[query], n_results=10)
            ids = results["ids"][0]
            distances = results["distances"][0]
            for image_id, distance in zip(ids, distances):
                gradio.Image(images_by_id[int(image_id)], label="Image")
                gradio.Number(distance, label="Distance")

    # Launch the Gradio dashboard
    dashboard.launch(inbrowser=True, share=False)


if __name__ == "__main__":
    main()
