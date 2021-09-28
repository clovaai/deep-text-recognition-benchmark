import io
import onnx
import onnxruntime
import numpy as np
import torch.onnx
from torchvision import models, transforms
from model import load_model
from global_args import get_cfgs

batch_size = 64
model_dir = "/home/revenuemonster/triton/newwww"
model_file = "region3"
model_in = f"{model_dir}/{model_file}.pt"
model_out = f"{model_dir}/{model_file}.onnx"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

opt = get_cfgs()
torch_model = load_model(opt)[0].eval()


# sample input
x = torch.randn(batch_size, 1, opt.imgH, opt.imgW, requires_grad=True).to(device)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model, x, model_out,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  training=torch.onnx.TrainingMode.EVAL,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})


# validate the output
onnx_model = onnx.load(model_out)
onnx.checker.check_model(onnx_model)

# compute ONNX Runtime output prediction
ort_session = onnxruntime.InferenceSession(model_out)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(
    to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")