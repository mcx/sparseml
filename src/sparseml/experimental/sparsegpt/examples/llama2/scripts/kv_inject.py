import click
import os
import onnx
from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector
from sparseml.onnx.utils import ONNXGraph
@click.command()
@click.option('--input-file', help='Path to the input ONNX model file')
@click.option('--output-file', help='Output path for the modified model')

def modify_model(input_file, output_file):
    model = onnx.load(input_file, load_external_data=False)
    model = KeyValueCacheInjector(model_path=os.path.dirname(input_file)).apply(model)
    graph = ONNXGraph(model)
    graph.delete_orphaned_node_branches()
    onnx.save(model, output_file)
    print(f"Modified model saved to: {output_file}")

if __name__ == '__main__':
    modify_model()
