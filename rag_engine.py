import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class AcademicRAG:
    def __init__(self, data_path="data/academic_faqs.jsonl"):
        self.data_path = Path(data_path)
        self.documents = [
            json.loads(line)
            for line in self.data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        texts = [
            doc["title"] + ". " + doc["text"]
            for doc in self.documents
        ]

        embeddings = self.model.encode(
            texts, normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, question, top_k=3):
        query_vector = self.model.encode(
            [question], normalize_embeddings=True
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = dict(self.documents[int(idx)])
                item["score"] = float(score)
                results.append(item)
        return results

    def answer(self, question, top_k=3):
        results = self.retrieve(question, top_k)
        if not results:
            return "No relevant source was found.", []

        evidence = "\n\n".join(
            f"[{item['source_id']}] {item['title']}: {item['text']}"
            for item in results
        )
        response = (
            "Retrieved evidence (not a generated answer):\n\n" + evidence
        )
        return response, results
