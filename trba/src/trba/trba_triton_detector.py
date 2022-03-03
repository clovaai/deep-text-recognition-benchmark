
import json
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F


from trba.core.trbaOcr import TrbaOCR
from trba.core.dataset import PillowImageDataset, AlignCollate
from trba.triton.trba_triton_client import TRBATritonClient

class TRITON_OCR_FLAGS():
    """
    Initializes the configuration parameters for Region Inference Triton Client.  
    """
    def __init__(self, triton_flags:json):
        
        self.url        = triton_flags["TRITON_OCR_SERVER_URL"]
        self.verbose    = triton_flags["TRITON_OCR_FLAG_VERBOSE"]                                                           # 'Enable verbose output'
        self.protocol   = triton_flags["TRITON_OCR_PROTOCOL"]                  # Protocol (HTTP/gRPC) used to communicate with server
        self.model_name = triton_flags["TRITON_OCR_MODEL_NAME"]
        
        self.model_version = ""
        #self.model_version = str(triton_flags("TRITON_REGION_INFERENCE_MODEL_VERSION"))                                                        # Version of model. Default is to use latest version
        self.batch_size    = int(triton_flags["TRITON_OCR_BATCH_SIZE"]) 
        self.classes       = int(triton_flags["TRITON_OCR_CLASSES"])                   # Number of class results to report. Default is 1
        self.scaling       = None                                                             # Type of scaling to apply to image pixels. Default is NONE
        self.async_set     = triton_flags["TRITON_OCR_ASYNC_SET"]    # 'Use asynchronous inference API'
        self.streaming     = triton_flags["TRITON_OCR_STREAMING"]                                                       # Use streaming inference API. The flag is only available with gRPC protocol.
       


class TRBATritonDetector:

    def __init__(self, triton_flags : json):   
        FLAGS = TRITON_OCR_FLAGS(triton_flags = triton_flags)
        self.triton_client = TRBATritonClient(FLAGS)   
          
        ## required for opt opreations in image loader ToDo - Remove dependency
        self.trba_ocr = TrbaOCR()  
        self.opt  = self.trba_ocr.opt
        self.converter = self.trba_ocr.converter 
       

  
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

    def img_to_image_loader(self, image):
        '''
        Image loader converts image to torch tensor, preprocesses to model input size
        and returns as a list of tensors
        '''
        

        AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
                   
        if image is not None:
            
            image_data = PillowImageDataset(image, self.opt)
            image_loader = torch.utils.data.DataLoader(
                image_data, batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=int(self.opt.workers),
                collate_fn=AlignCollate_demo, pin_memory=True)

        else:
            print("Could not find image path for inference.")
        
        return image_loader

    def recognize_ocr(self, image):
        
        
        image_loader = self.img_to_image_loader(image) 

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


    def post_process_preds(self, preds):

        # convert to torch tensor
        preds = torch.from_numpy(preds)
        print("pred shape :", preds.shape)

         # select max probabilty (greedy decoding) then decode index to character
  
        batch_size = 1
        output = {}

        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.opt.device)
        
        
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)

                            
    
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]                  
        
            
            #print(f'\t{pred:25s}\t{confidence_score:0.4f}')
            output['pred'] = pred
            output['score'] = np.array(confidence_score.cpu())

        return output
                    

   
      
    
    

    
    

    
    
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
    
    triton_flags = triton_config[ocr_component]
    trba_model_config = TrbaOCR()

    # intialize client
    trba_triton_detector = TRBATritonDetector(triton_flags = triton_flags, trba_model_config= trba_model_config)

    
    trba_triton_detector.parse_trba_model()   

    # preprocessed_batched_images = triton_client.
    preds = trba_triton_detector.recognize_ocr(pil_image)
    print("pred shape ", preds.shape)
    
 
   