import onnx
import onnxruntime as ort
import numpy as np


def check_onnx(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Check that the model is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))
    

def main(onnx_path):
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider']
    )

    outputs = ort_session.run(
        None,
        {"input": np.random.randn(8, 3, 32, 160).astype(np.float32)},
    )
    print(outputs)
    
    
if __name__ == "__main__":
    onnx_path = "output/baseline/onnx/best_line_acc_53.2698-epoch:172.onnx"
    check_onnx(onnx_path)
    main(onnx_path)
