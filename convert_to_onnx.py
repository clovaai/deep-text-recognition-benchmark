import numpy as np
import torch
from torch import nn
import torch.onnx
from model import MyModel


def get_model():
    arch_dict = {
        "trans": "None",
        "feat": "CRNN",
        "seq": "BILSTM",
        "head": "CTC"
    }
    feat_dict = {
        "input_c": 3,
        "output_c": 512
    }
    bilstm_dict = {
        "hidden_size": 256,
    }
    head_dict = {
        "num_classes": 74,  # CTC+1, Attn+2?
    }
    model = MyModel(arch_dict, feat_dict=feat_dict, bilstm_dict=bilstm_dict, head_dict=head_dict)
    return model

def load_params(model, ckpt_file):
    saved_state_dict = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(saved_state_dict)
    
def main(ckpt_path, onnx_path):
    batch_size = 8
    
    model = get_model()
    load_params(model, ckpt_path)
    model.eval()
    
    torch_input = torch.randn(batch_size, 3, 32, 160, requires_grad=False)
    torch_output = model(torch_input)

    # Export the model
    torch.onnx.export(
        model,                     # model being run
        torch_input,               # model input (or a tuple for multiple inputs)
        onnx_path,                 # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        verbose=True,              # prints a description of the model being exported to stdout
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size', 3: 'width'},    # variable length axes
                    'output' : {0 : 'batch_size', 2: 'sequence_length'}}
    )
    
    
if __name__ == "__main__":
    import os
    ckpt_path = "output/baseline/pth/best_line_acc_53.2698-epoch:172.pth"
    onnx_path = ckpt_path.replace("pth", "onnx")
    print(ckpt_path, onnx_path)
    assert os.path.exists(ckpt_path)
    assert os.path.exists(os.path.dirname(onnx_path))
    
    main(ckpt_path, onnx_path)
    print("end!")    
    
    
    