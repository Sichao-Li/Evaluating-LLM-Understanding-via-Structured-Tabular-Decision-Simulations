import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional, Any
from pathlib import Path

def load_and_clean_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a CSV and performs basic cleaning (strip whitespace, drop NaNs).
    """
    try:
        df = pd.read_csv(path).dropna()
        # Clean string columns
        df = df.map(lambda x: str(x).strip() if isinstance(x, str) else x)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {path}")

def stratified_sample(df: pd.DataFrame, target_col: str, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Downsamples the dataframe while maintaining class distribution.
    """
    if len(df) <= max_rows:
        return df
    
    n_classes = df[target_col].nunique()
    if n_classes == 0:
        return df.sample(n=max_rows, random_state=seed)
        
    rows_per_class = max_rows // n_classes
    return df.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), rows_per_class), random_state=seed)
    ).reset_index(drop=True)

def drop_attributes(df: pd.DataFrame, attrs: List[Union[str, int]]) -> pd.DataFrame:
    """
    Removes columns by name or index. Used for Leave-Attribute-Out (LAO).
    """
    if not attrs:
        return df
        
    col_names = []
    for attr in attrs:
        if isinstance(attr, int):
            if attr < 0 or attr >= len(df.columns):
                raise IndexError(f"Column index {attr} is out of bounds.")
            col_names.append(df.columns[attr])
        elif isinstance(attr, str):
            if attr not in df.columns:
                # Warn but continue, or raise error depending on preference
                print(f"Warning: Attribute '{attr}' not found in DataFrame to drop.")
                continue
            col_names.append(attr)
            
    return df.drop(columns=col_names)

def format_row_as_text(row: pd.Series) -> str:
    """Converts a pandas row to 'col1=val1, col2=val2' format."""
    return ", ".join([f"{col}={val}" for col, val in row.items()])

def prepare_prediction_context(
    df: pd.DataFrame, 
    target_col: str = "class", 
    mask_train: bool = True, 
    mask_test: bool = True,
    move_unknowns_to_end: bool = True
) -> Tuple[str, List[Any], List[int]]:
    """
    Core logic for preparing the LLM input table.
    
    1. Iterates rows.
    2. Masks the target column (e.g., 'class=?') based on train/test flags.
    3. Optionally moves all 'class=?' rows to the bottom of the string (Prompt Engineering best practice).
    
    Returns:
        table_str (str): The formatted text table.
        ground_truths (list): The actual labels for the masked rows.
        masked_indices (list): Original indices of the masked rows.
    """
    
    formatted_rows = []
    ground_truths = []
    masked_indices = []
    
    # We need to iterate via index to keep track of original positions
    for idx, row in df.iterrows():
        dataset_type = row.get('dataset', 'unknown') # Expects 'train' or 'test' column
        
        should_mask = False
        if target_col in row:
            if dataset_type == 'train' and mask_train:
                should_mask = True
            elif dataset_type == 'test' and mask_test:
                should_mask = True
        
        # Construct the row string
        items = []
        for col in df.columns:
            if col == 'dataset': 
                continue # Don't print the 'dataset' metadata column
            
            val = row[col]
            if col == target_col and should_mask:
                items.append(f"{col}=?")
            else:
                items.append(f"{col}={val}")
        
        row_str = f"Row {idx+1}: " + ", ".join(items)
        
        if should_mask:
            ground_truths.append(row[target_col])
            masked_indices.append(idx)
            # Mark this row struct as "masked" for sorting later
            formatted_rows.append({"text": row_str, "is_masked": True})
        else:
            formatted_rows.append({"text": row_str, "is_masked": False})

    # Sort rows: Put knowns first, unknowns (masked) last
    if move_unknowns_to_end:
        formatted_rows.sort(key=lambda x: x["is_masked"])

    # Join into final string
    table_str = "\n".join([r["text"] for r in formatted_rows])
    
    return table_str, ground_truths, masked_indices

