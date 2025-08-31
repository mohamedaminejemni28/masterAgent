from nbconvert import ScriptExporter
import nbformat

def convert_ipynb_to_py(ipynb_file, output_file="masterAgent.py"):
    # Read the notebook
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Convert to .py
    exporter = ScriptExporter()
    (body, resources) = exporter.from_notebook_node(notebook)
    
    # Write to output file
    if output_file is None:
        output_file = ipynb_file.replace('.ipynb', '.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(body)

# Example usage
convert_ipynb_to_py('master-agent (4).ipynb')