"""Free local semantic embeddings for reproducible Workshop 1-4 checks."""

from pathlib import Path


class LocalONNXEmbeddings:
    """Run all-MiniLM-L6-v2 on CPU; first use downloads its public model."""

    dimension = 384

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(".model-cache/all-MiniLM-L6-v2").resolve()
        self.model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self.model(texts)]
