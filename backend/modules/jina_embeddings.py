import os
import requests

class JinaEmbeddings:
    def __init__(self, model: str = "jina-clip-v2", dimensions: int = 1024):
        # Retrieve the API key from the environment variable.
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is not set.")
        self.model = model
        self.dimensions = dimensions
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        data = {
            "model": self.model,
            "dimensions": self.dimensions,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": text} for text in texts]
        }
        response = requests.post(self.api_url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Error calling Jina embeddings API: {response.text}")
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
