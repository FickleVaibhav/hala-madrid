import plotly.graph_objects as go
import plotly.express as px

# Create data for the repository structure as a treemap
# Using hierarchical data structure for treemap
data = {
    'ids': [
        # Root
        'respiratory-disease-detection',
        
        # Root level files
        'README.md', '.gitignore', 'requirements.txt', 'setup.py', 'LICENSE',
        
        # Main folders
        'hardware', 'src', 'data', 'web_app', 'scripts', 'notebooks', 'tests', 'docs', 'docker',
        
        # Hardware subfolder contents
        'hardware/README.md', 'hardware/bill_of_materials.md', 'hardware/assembly_instructions.md',
        'hardware/wiring_diagrams', 'hardware/3d_models',
        
        # Src subfolder contents
        'src/preprocessing', 'src/models', 'src/utils',
        
        # Data subfolder contents
        'data/datasets', 'data/preprocessed', 'data/models',
        
        # Web app subfolder contents
        'web_app/streamlit_app.py', 'web_app/fastapi_backend.py', 'web_app/static',
        
        # Scripts subfolder contents
        'scripts/train_model.py', 'scripts/evaluate_model.py', 'scripts/download_data.py'
    ],
    
    'labels': [
        # Root
        '📁 respiratory-disease-detection/',
        
        # Root level files
        '📄 README.md', '📄 .gitignore', '📄 requirements.txt', '📄 setup.py', '📄 LICENSE',
        
        # Main folders
        '📁 hardware/', '📁 src/', '📁 data/', '📁 web_app/', '📁 scripts/', 
        '📁 notebooks/', '📁 tests/', '📁 docs/', '📁 docker/',
        
        # Hardware subfolder contents
        '📄 README.md', '📄 bill_of_materials.md', '📄 assembly_instructions.md',
        '📁 wiring_diagrams/', '📁 3d_models/',
        
        # Src subfolder contents
        '📁 preprocessing/', '📁 models/', '📁 utils/',
        
        # Data subfolder contents
        '📁 datasets/', '📁 preprocessed/', '📁 models/',
        
        # Web app subfolder contents
        '🐍 streamlit_app.py', '🐍 fastapi_backend.py', '📁 static/',
        
        # Scripts subfolder contents
        '🐍 train_model.py', '🐍 evaluate_model.py', '🐍 download_data.py'
    ],
    
    'parents': [
        # Root
        '',
        
        # Root level files
        'respiratory-disease-detection', 'respiratory-disease-detection', 'respiratory-disease-detection', 
        'respiratory-disease-detection', 'respiratory-disease-detection',
        
        # Main folders
        'respiratory-disease-detection', 'respiratory-disease-detection', 'respiratory-disease-detection', 
        'respiratory-disease-detection', 'respiratory-disease-detection', 'respiratory-disease-detection', 
        'respiratory-disease-detection', 'respiratory-disease-detection', 'respiratory-disease-detection',
        
        # Hardware subfolder contents
        'hardware', 'hardware', 'hardware', 'hardware', 'hardware',
        
        # Src subfolder contents
        'src', 'src', 'src',
        
        # Data subfolder contents
        'data', 'data', 'data',
        
        # Web app subfolder contents
        'web_app', 'web_app', 'web_app',
        
        # Scripts subfolder contents
        'scripts', 'scripts', 'scripts'
    ],
    
    'values': [
        # Root
        100,
        
        # Root level files (smaller values)
        2, 1, 2, 2, 1,
        
        # Main folders (larger values to show hierarchy)
        15, 12, 10, 8, 6, 4, 4, 4, 3,
        
        # Hardware subfolder contents
        2, 3, 3, 3, 4,
        
        # Src subfolder contents
        4, 4, 4,
        
        # Data subfolder contents
        3, 3, 4,
        
        # Web app subfolder contents
        3, 3, 2,
        
        # Scripts subfolder contents
        2, 2, 2
    ]
}

# Define colors based on file types
colors = []
for label in data['labels']:
    if '📁' in label:  # Folders - blue
        colors.append('#B3E5EC')
    elif '🐍' in label:  # Python files - green  
        colors.append('#A5D6A7')
    elif '📄' in label:  # Documentation - yellow
        colors.append('#FFEB8A')
    else:
        colors.append('#9FA8B0')  # Default light blue-gray

# Create treemap
fig = go.Figure(go.Treemap(
    ids=data['ids'],
    labels=data['labels'],
    parents=data['parents'],
    values=data['values'],
    marker=dict(
        colors=colors,
        line=dict(width=2)
    ),
    textfont=dict(size=12),
    maxdepth=3,
    branchvalues="total"
))

fig.update_layout(
    title="Respiratory Disease Detection Repository Structure",
    font_size=12
)

# Save the chart
fig.write_image("repository_structure.png")
fig.write_image("repository_structure.svg", format="svg")

print("Repository structure treemap saved successfully!")