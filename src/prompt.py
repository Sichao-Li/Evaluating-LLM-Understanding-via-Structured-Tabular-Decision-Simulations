import yaml
import copy
import jinja2
from pathlib import Path
from typing import List, Optional, Dict, Any

class PromptEngine:
    """
    Handles the loading of dataset metadata and the rendering of 
    instruction prompts using Jinja2 templates.
    """
    
    def __init__(self, template_dir: str = "templates", metadata_dir: str = "data/"):
        self.metadata_dir = Path(metadata_dir)
        
        # Initialize Jinja2 Environment
        # This replaces the global 'env' variable in your original script
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Cache loaded YAMLs to reduce disk I/O during loops
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    def _get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Loads metadata from YAML or retrieves from cache."""
        if dataset_id in self._metadata_cache:
            return self._metadata_cache[dataset_id]

        path = self.metadata_dir/dataset_id/f"{dataset_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        self._metadata_cache[dataset_id] = data
        return data

    def render_prediction_instruction(
        self, 
        dataset_id: str, 
        few_shot: bool = False, 
        drop_attributes: Optional[List[str]] = None
    ) -> str:
        """
        Renders the instruction for the prediction task (Task 1 & LAO).
        
        Args:
            dataset_id: The name of the dataset (e.g., 'iris').
            few_shot: Whether to inject few-shot context (affects template logic).
            drop_attributes: List of attribute names to remove from the glossary
                             (used for Leave-Attribute-Out experiments).
        """
        # 1. Load Metadata
        meta = self._get_metadata(dataset_id)
        
        # 2. Create a context copy to avoid mutating the cached metadata
        # This is crucial for LAO loops where we drop different attributes each time
        context = copy.deepcopy(meta)
        
        # 3. Handle Attribute Dropping (LAO Logic)
        if drop_attributes:
            glossary = context.get("attribute_glossary", {})
            for attr in drop_attributes:
                if attr in glossary:
                    del glossary[attr]
            # We update the context's glossary with the pruned version
            context["attribute_glossary"] = glossary

        # 4. Set Flags
        context["few_shot"] = few_shot

        # 5. Render
        template = self.env.get_template("instruction.txt.j2")
        return template.render(**context)

    def render_attribution_instruction(self, dataset_id: str) -> str:
        """
        Renders the instruction for the self-attribution task (Task 2).
        """
        meta = self._get_metadata(dataset_id)
        template = self.env.get_template("attribution.txt.j2")
        return template.render(**meta)