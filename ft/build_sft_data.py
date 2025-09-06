import json
from typing import List, Dict

# Import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

class SFTDataBuilder:
    def __init__(self):
        self.data = []
    
    def add_conversation(self, instruction: str, input_text: str = "", output: str = ""):
        conversation = {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
        self.data.append(conversation)
    
    def add_chat_format(self, messages: List[Dict[str, str]]):
        formatted_conversation = {
            "conversations": messages
        }
        self.data.append(formatted_conversation)
    
    def load_from_csv(self, csv_path: str, instruction_col: str = "instruction", 
                      input_col: str = "input", output_col: str = "output"):
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV loading. Install with: pip install pandas")
        
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.add_conversation(
                instruction=row[instruction_col],
                input_text=row.get(input_col, ""),
                output=row[output_col]
            )
    
    def save_jsonl(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def save_json(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def to_dataset(self):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        return Dataset.from_list(self.data)
    
    def validate_data(self):
        if not self.data:
            raise ValueError("No data to validate")
        
        for i, item in enumerate(self.data):
            if "conversations" in item:
                if not isinstance(item["conversations"], list):
                    raise ValueError(f"Item {i}: conversations must be a list")
            else:
                required_fields = ["instruction", "output"]
                for field in required_fields:
                    if field not in item:
                        raise ValueError(f"Item {i}: missing required field '{field}'")
        
        print(f"Validation passed for {len(self.data)} items")

if __name__ == "__main__":
    builder = SFTDataBuilder()
    
    builder.add_conversation(
        instruction="What is the capital of France?",
        output="The capital of France is Paris."
    )
    
    builder.add_conversation(
        instruction="Explain the concept of machine learning.",
        output="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
    )
    
    builder.validate_data()
    builder.save_jsonl("sft_data.jsonl")
    print("SFT data saved to sft_data.jsonl")