# Let's create the repository structure and key files for the respiratory disease detection project

import os
import json
from datetime import datetime

# Create the repository structure
repo_structure = {
    "respiratory-disease-detection": {
        "README.md": "",
        ".gitignore": "",
        "requirements.txt": "",
        "setup.py": "",
        "LICENSE": "",
        "hardware/": {
            "README.md": "",
            "bill_of_materials.md": "",
            "assembly_instructions.md": "",
            "wiring_diagrams/": {
                "stethoscope_3.5mm_connection.md": "",
                "usb_c_connection_diagram.md": "",
                "circuit_schematic.md": ""
            },
            "3d_models/": {
                "stethoscope_adapter.stl": "",
                "microphone_housing.stl": ""
            }
        },
        "src/": {
            "__init__.py": "",
            "preprocessing/": {
                "__init__.py": "",
                "audio_preprocessing.py": "",
                "noise_reduction.py": "",
                "feature_extraction.py": ""
            },
            "models/": {
                "__init__.py": "",
                "cnn_model.py": "",
                "resnet_model.py": "",
                "ensemble_model.py": ""
            },
            "utils/": {
                "__init__.py": "",
                "audio_utils.py": "",
                "visualization.py": "",
                "config.py": ""
            }
        },
        "data/": {
            "README.md": "",
            "datasets/": {
                "README.md": "",
                "download_datasets.py": ""
            },
            "preprocessed/": {},
            "models/": {
                "pretrained/": {},
                "trained/": {}
            }
        },
        "notebooks/": {
            "01_data_exploration.ipynb": "",
            "02_preprocessing_pipeline.ipynb": "",
            "03_model_training.ipynb": "",
            "04_evaluation_analysis.ipynb": ""
        },
        "web_app/": {
            "__init__.py": "",
            "streamlit_app.py": "",
            "fastapi_backend.py": "",
            "static/": {
                "css/": {
                    "styles.css": ""
                },
                "js/": {
                    "audio_recorder.js": ""
                }
            },
            "templates/": {
                "index.html": ""
            }
        },
        "scripts/": {
            "train_model.py": "",
            "evaluate_model.py": "",
            "download_data.py": "",
            "preprocess_data.py": ""
        },
        "tests/": {
            "__init__.py": "",
            "test_preprocessing.py": "",
            "test_models.py": "",
            "test_utils.py": ""
        },
        "docs/": {
            "installation.md": "",
            "usage.md": "",
            "api_documentation.md": "",
            "dataset_info.md": "",
            "model_architecture.md": ""
        },
        "docker/": {
            "Dockerfile": "",
            "docker-compose.yml": "",
            "requirements-docker.txt": ""
        }
    }
}

def create_structure(structure, base_path=""):
    for name, content in structure.items():
        path = os.path.join(base_path, name) if base_path else name
        
        if isinstance(content, dict):
            # It's a directory
            print(f"Directory: {path}")
            create_structure(content, path)
        else:
            # It's a file
            print(f"File: {path}")

print("Repository Structure:")
print("====================")
create_structure(repo_structure)
print("\nTotal directories and files mapped out for the respiratory disease detection project!")