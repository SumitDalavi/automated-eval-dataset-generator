import json
import os

class DatasetGenerator:
    def __init__(self, document_text):
        self.document = document_text

    def extract_chunks(self, chunk_size=100):
        """Split document into rough word chunks."""
        words = self.document.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def generate_qa_pair(self, chunk):
        """Simulate LLM generating a question and answer from a chunk."""
        # Mock logic. In reality, call LLM.
        return {
            "prompt": f"Based on the text: '{chunk[:50]}...', what is the main idea?",
            "completion": f"The main idea relates to {chunk.split()[0]}."
        }

    def process(self):
        chunks = self.extract_chunks()
        dataset = []
        for c in chunks:
            if c.strip():
                dataset.append(self.generate_qa_pair(c))
        return dataset

    def export_jsonl(self, dataset, output_file="dataset.jsonl"):
        """Export the dataset to JSONL format for fine-tuning."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        return output_file

if __name__ == "__main__":
    sample_doc = "Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence of humans or animals. It is a field of study in computer science that develops and studies intelligent machines."
    generator = DatasetGenerator(sample_doc)
    dataset = generator.process()
    filepath = generator.export_jsonl(dataset)
    print(f"Generated {len(dataset)} examples. Saved to {filepath}")
