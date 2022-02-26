import json
import sys
from attrdict import AttrDict
from trba.trbaOcr import TrbaOCR
import numpy as np
from functools import partial

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import torch

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


class TRBATritonClient():

    def __init__(self, FLAGS ):

        self.FLAGS = FLAGS
        self.initialize_triton_client()


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


    def postprocess(self, results, output_name, batching):
        """
        Post-process triton results 
        """

        response_output = results.get_output(output_name)
        
        output_array = results.as_numpy(output_name)
    
        print(output_array.shape)
        if len(output_array) != self.FLAGS.batch_size:
            raise Exception("expected {} results, got {}".format(
                self.FLAGS.batch_size, len(output_array)))
    
        return output_array
    

    def send_requests_to_triton(self, input_image_name, input_image_dtype, input_text_name, input_text_dtype, output_name, output_dtype, image_data, max_batch_size):
        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        self.max_batch_size = max_batch_size

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