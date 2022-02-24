
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


def _batch_images(img):
    """
    Can be extended to a batch of images. 
    """
    
    image_data = []
    image_data.append(img)
    return image_data


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
       

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def requestGenerator(batched_image_data,
                 input_image_name, input_image_dtype,
                  input_text_name, input_text_dtype,
                   output_name, output_dtype, FLAGS):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    text_tensor = torch.LongTensor(1, 26).fill_(0).numpy()
    text_array = np.zeros((1,26),dtype=np.int64) 

    inputs = [client.InferInput(input_image_name, batched_image_data.shape, input_image_dtype),
                client.InferInput(input_text_name, text_tensor.shape, input_text_dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)
    inputs[1].set_data_from_numpy(text_array)

    outputs = [
        client.InferRequestedOutput(output_name, binary_data=False, class_count=0)
    ]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version
    

def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config

class TRBATritonClient:

    def __init__(self, detector_config : json):   
        
        
        self.FLAGS = TRITON_OCR_FLAGS(detector_config = detector_config)
     

    

    def initialize_triton_client(self):
        if self.FLAGS.streaming and self.FLAGS.protocol.lower() != "grpc":
            raise Exception("Streaming is only allowed with gRPC protocol")

        try:
            if self.FLAGS.protocol.lower() == "grpc":
                # Create gRPC client for communicating with the server
                self.triton_client = grpcclient.InferenceServerClient(
                    url=self.FLAGS.url, verbose=self.FLAGS.verbose)
                #return triton_client
            else:
                # Specify large enough concurrency to handle the
                # the number of requests.
                concurrency = 20 if self.FLAGS.async_set else 1
                self.triton_client = httpclient.InferenceServerClient(
                    url=self.FLAGS.url, verbose=self.FLAGS.verbose, concurrency=concurrency)
                
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)


    def get_model_metadata_and_model_config(self):
        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=self.FLAGS.model_name, model_version=self.FLAGS.model_version) 
            self.model_config = self.triton_client.get_model_config(
                model_name=self.FLAGS.model_name, model_version=self.FLAGS.model_version)  
            
            if self.FLAGS.protocol.lower() == "grpc":
                self.model_config = self.model_config.config
            else:
                self.model_metadata, self.model_config = convert_http_metadata_config(
                self.model_metadata, self.model_config)

            print("Model Metadata : ", self.model_metadata)
            print("Model config: ", self.model_config)

            return self.model_metadata, self.model_config
        except InferenceServerException as e:
            print("failed to retrieve the metadata or config: " + str(e))
            sys.exit(1)  


  
    def parse_model(self):
        """
        A model specific function to check the configuration of a model to make sure it meets the
        requirements for a YOLO object detection network (as expected by
        this client)
        """

        model_metadata, model_config = self.get_model_metadata_and_model_config()
       
      
        input_image_metadata = model_metadata.inputs[0]
        input_text_metadata = model_metadata.inputs[1]
        input_config = model_config.input[0]  
        output_metadata = model_metadata.outputs[0]
        


        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
    
            
        c = input_image_metadata.shape[1]
        h = input_image_metadata.shape[2]
        w = input_image_metadata.shape[3]

        print("max_batch_size : ", model_config.max_batch_size, "input_image_name : ", input_image_metadata.name, 
        "input_text_name : ", input_text_metadata.name , " output_name_boxes : ",
         output_metadata.name, "c : ", c, " h : ", h , " w: ", w, " format : " , input_config.format, " dtype : ", input_image_metadata.datatype) 

        self.max_batch_size = model_config.max_batch_size
        self.input_image_name = input_image_metadata.name
        self.input_text_name = input_text_metadata.name
        self.output_name = output_metadata.name
        
        self.c = c
        self.h = h
        self.w = w
        self.format = input_config.format
        self.input_image_dtype = input_image_metadata.datatype
        self.input_text_dtype = input_text_metadata.datatype
        self.output_dtype = output_metadata.datatype

    def ingest_batch_of_images(self,image_loader):
        preprocessed_image_data = []

        ### input tensor must be a numpy array
        for image_tensors, image_path_list in image_loader:
            
            batch_size = image_tensors.size(0)
        
            image = image_tensors.numpy()
            print("image shape " , image.shape)
            preprocessed_image_data.append(image)
       
        return preprocessed_image_data

    def get_infer_results(self, image_data):

        preds = self.send_requests_to_triton(self.input_image_name,
                     self.input_image_dtype,
                     self.input_text_name,
                     self.input_text_dtype,
                     self.output_name,
                     self.output_dtype,
                       image_data)
      
        return preds
    
    

    
    def send_requests_to_triton(self, input_image_name, input_image_dtype, input_text_name, input_text_dtype, output_name, output_dtype, image_data):
        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        requests = []
        responses = []
        result_filenames = []
        request_ids = []
        image_idx = 0
        last_request = False
        user_data = UserData()

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []

        sent_count = 0

        if self.FLAGS.streaming:
            self.triton_client.start_stream(partial(completion_callback, user_data))

        while not last_request:
            #input_filenames = []
            repeated_image_data = []

            for idx in range(self.FLAGS.batch_size):
                #input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            if self.max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request
            try:
                for inputs, outputs, model_name, model_version in requestGenerator(
                        batched_image_data,
                         input_image_name, input_image_dtype,
                          input_text_name, input_text_dtype,
                           output_name, output_dtype,
                           self.FLAGS):
                    sent_count += 1
                    if self.FLAGS.streaming:
                        self.triton_client.async_stream_infer(
                            self.FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=self.FLAGS.model_version,
                            outputs=outputs)
                    elif self.FLAGS.async_set:
                        if self.FLAGS.protocol.lower() == "grpc":
                            self.triton_client.async_infer(
                                self.FLAGS.model_name,
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=self.FLAGS.model_version,
                                outputs=outputs)
                        else:
                            async_requests.append(
                                self.triton_client.async_infer(
                                    self.FLAGS.model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=self.FLAGS.model_version,
                                    outputs=outputs))
                    else:
                        responses.append(
                            self.triton_client.infer(self.FLAGS.model_name,
                                                inputs,
                                                request_id=str(sent_count),
                                                model_version=self.FLAGS.model_version,
                                                outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if self.FLAGS.streaming:
                    self.triton_client.stop_stream()
                sys.exit(1)

        if self.FLAGS.streaming:
            self.triton_client.stop_stream()

        if self.FLAGS.protocol.lower() == "grpc":
            if self.FLAGS.streaming or self.FLAGS.async_set:
                processed_count = 0
                while processed_count < sent_count:
                    (results, error) = user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        print("inference failed: " + str(error))
                        sys.exit(1)
                    responses.append(results)
        else:
            if self.FLAGS.async_set:
                # Collect results from the ongoing async requests
                # for HTTP Async requests.
                for async_request in async_requests:
                    responses.append(async_request.get_result())

        for response in responses:
            if self.FLAGS.protocol.lower() == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]
            print("Request {}, batch size {}".format(this_id, self.FLAGS.batch_size))
            
            response_output_array = self.postprocess(response, output_name, self.max_batch_size > 0)
            return response_output_array
    
    def postprocess(self, results, output_name, batching):
        """
        Post-process results to show classifications.
        """

        
        response_output = results.get_output(output_name)
        
        output_array = results.as_numpy(output_name)
       
        print(output_array.shape)
        if len(output_array) != self.FLAGS.batch_size:
            raise Exception("expected {} results, got {}".format(
                self.FLAGS.batch_size, len(output_array)))
    
        return output_array
    
    
    
    def save_region_image_from_yolo_output(self,original_image, yolo_output,savename):
            plot_boxes_cv2(original_image, yolo_output, savename, class_names=self.FLAGS.class_names)   
    
    
    
if __name__ == "__main__":

        

    # read test image
    from PIL import Image
    image_path = 'demo_image/ean.jpg'
    pil_image = Image.open(image_path)#   
   
    # image to image-loader

    trbaOCR = TrbaOCR(device='cuda')
    image_loader = trbaOCR.img_to_image_loader(pil_image)

    # predicted_text = trbaOCR.img_to_ean(pil_image)
    # print("predicted_text : ", ean['pred'], ean['score'])
   
  
    ## Read triton configs

    import json
    with open("triton.json") as f:
        triton_config = json.load(f)
    ocr_component = "EANs"
    detector_config = triton_config[ocr_component]
   

    # intialize client
    triton_client = TRBATritonClient(detector_config = detector_config)

    triton_client.initialize_triton_client()

    triton_client.parse_model()

   

    preprocessed_batched_images = triton_client.ingest_batch_of_images(image_loader)
    preds = triton_client.get_infer_results(preprocessed_batched_images)
    print("pred shape ", preds.shape)
    # inference

    # opt = trbaOCR.opt             
    # device = opt.device

    # predict
    #trbaOCR.model.eval()
   #output = {}
    #with torch.no_grad():
 
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