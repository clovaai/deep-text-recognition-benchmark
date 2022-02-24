
import json
import sys
from attrdict import AttrDict
from dptr.trbaOcr import TrbaOCR
import numpy as np
from functools import partial


import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import torch


class TRITON_OCR_FLAGS():
    """
    Initializes the configuration parameters for Region Inference Triton Client.  
    """
    def __init__(self, detector_config:json):
        
        self.url        = detector_config["TRITON_OCR_SERVER_URL"]
        self.verbose    = detector_config["TRITON_OCR_FLAG_VERBOSE"]                                                           # 'Enable verbose output'
        self.protocol   = detector_config["TRITON_OCR_PROTOCOL"]                  # Protocol (HTTP/gRPC) used to communicate with server
        self.model_name = detector_config["TRITON_OCR_MODEL_NAME"]
        
        self.model_version = ""
        #self.model_version = str(detector_config("TRITON_REGION_INFERENCE_MODEL_VERSION"))                                                        # Version of model. Default is to use latest version
        self.batch_size    = int(detector_config["TRITON_OCR_BATCH_SIZE"]) 
        self.classes       = int(detector_config["TRITON_OCR_CLASSES"])                   # Number of class results to report. Default is 1
        self.scaling       = None                                                             # Type of scaling to apply to image pixels. Default is NONE
        self.async_set     = detector_config["TRITON_OCR_ASYNC_SET"]    # 'Use asynchronous inference API'
        self.streaming     = detector_config["TRITON_OCR_STREAMING"]                                                       # Use streaming inference API. The flag is only available with gRPC protocol.
       


from trba_triton_client import TRBATritonClient
class TRBATritonDetector:

    def __init__(self, detector_config : json):   
        FLAGS = TRITON_OCR_FLAGS(detector_config = detector_config)
        self.triton_client = TRBATritonClient(FLAGS)     
        self.trbaOCR = TrbaOCR() # trba models / utils / packages   


  
    def parse_trba_model(self):
        """
        A model specific function to check the configuration of a model to make sure it meets the
        requirements for a YOLO object detection network (as expected by
        this client)
        """

        model_metadata, model_config = self.triton_client.get_model_metadata_and_model_config()      
      
        input_image_metadata = model_metadata.inputs[0]
        input_text_metadata = model_metadata.inputs[1]
        input_config = model_config.input[0]  
        output_metadata = model_metadata.outputs[0]
        

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)    
            
        self.c = input_image_metadata.shape[1]
        self.h  = input_image_metadata.shape[2]
        self.w = input_image_metadata.shape[3]

        self.max_batch_size = model_config.max_batch_size
        self.input_image_name = input_image_metadata.name
        self.input_text_name = input_text_metadata.name
        self.output_name = output_metadata.name
        
 
        self.format = input_config.format
        self.input_image_dtype = input_image_metadata.datatype
        self.input_text_dtype = input_text_metadata.datatype
        self.output_dtype = output_metadata.datatype



    def recognize_ocr(self, image):
        
        # image loader converts image to torch tensor, preprocesses to model input size
        # and returns as a list of tensors
        image_loader = self.trbaOCR.img_to_image_loader(pil_image) 

        preprocessed_image_data = []
        # Triton server expects input tensor to be a numpy array
        for image_tensors, image_path_list in image_loader:                
                
            image = image_tensors.numpy()
            print("image shape " , image.shape)
            preprocessed_image_data.append(image)

        preds = self.triton_client.send_requests_to_triton(self.input_image_name,
                    self.input_image_dtype,
                    self.input_text_name,
                    self.input_text_dtype,
                    self.output_name,
                    self.output_dtype,
                    preprocessed_image_data, 
                    self.max_batch_size)
      
        return preds

   
      
    
    

    
    

    
    
if __name__ == "__main__":

        

    # read test image
    from PIL import Image
    image_path = 'demo_image/ean.jpg'
    pil_image = Image.open(image_path)#   
   
  
    ## Read triton configs

    import json
    with open("triton.json") as f:
        triton_config = json.load(f)
    ocr_component = "EANs"
    detector_config = triton_config[ocr_component]
   

    # intialize client
    trba_triton_detector = TRBATritonDetector(detector_config = detector_config)

    
    trba_triton_detector.parse_trba_model()   

    # preprocessed_batched_images = triton_client.
    preds = trba_triton_detector.recognize_ocr(pil_image)
    print("pred shape ", preds.shape)
    
 
    # for image_tensors, image_path_list in image_loader:
    #     batch_size = image_tensors.size(0)
        
    #     image = image_tensors.to(device)
    #     print("image shape " ,image.shape)
    #     # For max length prediction
    #     length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    #     text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        
        
        #preds = self.model(image, text_for_pred, is_train=False)
        
        

       

    #triton_client.parse_model()
    
    
    # batched_images = _batch_images(img)           
    
    # preprocessed_batched_images = triton_client.ingest_batch_of_images(batched_images)        
    # bboxes_batch = triton_client.get_infer_results(preprocessed_batched_images)        
        
    # bboxes_array = np.array(bboxes_batch)        

    # detection_results = []
    # # check if output is 3 dimensional # batch, detections, 7 values
    # if len(bboxes_array.shape) == 3 : 
    
    #     detection_results = _build_detection_results_from_triton_output(img, bboxes_array, self.detection_classes)
    #     return detection_results
    # else:
    #     return None