
import unittest
from rag_engine import AcademicRAG


class TestAcademicRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = AcademicRAG("data/academic_faqs.jsonl")

    def test_retrieval_returns_results(self):
        results = self.assistant.retrieve(
            "How do I protect student records?", top_k=3
        )
        self.assertGreater(len(results), 0)

    def test_results_have_source_ids(self):
        results = self.assistant.retrieve(
            "Who can help with course planning?", top_k=3
        )
        self.assertTrue(
            all("source_id" in item for item in results)
        )


if __name__ == "__main__":
    unittest.main()
