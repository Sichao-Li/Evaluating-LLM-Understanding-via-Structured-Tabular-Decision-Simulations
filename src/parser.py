import re
import ast
import json
import numpy as np
from typing import List, Union, Optional
from src.models import LLM

class OutputParser:
    """
    Uses a helper LLM to extract structured data (lists, rankings) from 
    messy unstructured model outputs.
    """
    
    def __init__(self, extraction_model: LLM):
        """
        Args:
            extraction_model: An instance of LLM (OpenAI, Gemini, or HF) 
                              used to perform the extraction.
        """
        self.llm = extraction_model

    def extract_predictions(self, raw_output: str, expected_count: int) -> List[int]:
        """
        Extracts a list of integer class predictions from raw text.
        """
        prompt = f"""
        You are a data extraction assistant.
        
        ### Task
        Extract the integer prediction labels from the text below.
        - The output MUST be a valid Python list of integers (e.g., [0, 1, 1, 0]).
        - The list MUST contain exactly {expected_count} integers.
        - Do NOT skip any row. If a row is missing a prediction, use -1.
        - Output ONLY the list. No markdown, no explanations.

        ### Raw Text
        ---
        {raw_output}
        ---
        
        ### Response:
        """
        
        cleaned_text = self.llm.generate(prompt, temperature=0.0)
        result = self._parse_python_list(cleaned_text)
        
        # Validation
        if len(result) != expected_count:
            print(f"Warning: Extracted {len(result)} predictions, expected {expected_count}.")
            # Optional: Pad or truncate logic could go here if desired
            
        return result

    def extract_attributions(self, raw_output: str) -> List[str]:
        """
        Extracts a ranked list of feature names (attribution) from raw text.
        """
        prompt = f"""
        You are a data extraction assistant.

        ### Task
        Extract the ranked list of feature names (attributes) from the text below.
        - The output MUST be a valid Python list of strings (e.g., ['age', 'income', 'sex']).
        - Ignore any numbering, bullet points, or explanations.
        - Use the exact feature names found in the text.
        - Output ONLY the list.

        ### Raw Text
        ---
        {raw_output}
        ---

        ### Response:
        """
        
        cleaned_text = self.llm.generate(prompt, temperature=0.0)
        return self._parse_python_list(cleaned_text)

    def _parse_python_list(self, text: str) -> List:
        """
        Robustly finds and parses a Python list [...] from a string.
        """
        # 1. Try simple regex for [ ... ]
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            # Fallback: sometimes models forget the brackets, just comma separated
            # If it looks like "0, 1, 0", wrap it
            if re.match(r"^\s*(\d+\s*,\s*)*\d+\s*$", text):
                text = f"[{text}]"
                match = re.search(r"\[.*?\]", text, re.DOTALL)
            else:
                print(f"Parser Error: No list found in response: {text[:100]}...")
                return []
        
        list_str = match.group(0)
        
        # 2. Safe Eval
        try:
            # ast.literal_eval is safer than eval()
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return parsed
            # If it parsed to a tuple, convert to list
            if isinstance(parsed, tuple):
                return list(parsed)
        except (ValueError, SyntaxError):
            # 3. Last Resort: Manual splitting for simple integer lists
            # useful if the model put comments inside the list like [0, 1, # comment]
            try:
                clean_content = list_str.strip("[]")
                # Split by comma, strip quotes/spaces
                items = [x.strip().strip("'\"") for x in clean_content.split(",")]
                # Try converting to int if they look like numbers
                final_list = []
                for item in items:
                    if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
                        final_list.append(int(item))
                    elif item: # non-empty string
                        final_list.append(item)
                return final_list
            except Exception as e:
                print(f"Parser Failed completely: {e}")
                return []
        return []