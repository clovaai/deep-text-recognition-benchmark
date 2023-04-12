"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention, TransformerDecoder, TorchDecoderWrapper
from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
import torch.nn.init as init
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        else:
            # self.SequenceModeling = nn.Conv2d(
            #     in_channels=self.FeatureExtraction_output, out_channels=opt.hidden_size,kernel_size=1
            # )
            self.SequenceModeling = nn.Conv1d(
                in_channels=self.FeatureExtraction_output, out_channels=opt.hidden_size,kernel_size=1
            )
            print('No SequenceModeling module specified')
            # self.SequenceModeling_output = self.FeatureExtraction_output

        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        elif opt.Prediction == 'TransformerDecoder':
            # seq_length + 2 to include <start> and <end> characters
            if opt.use_torch_transformer:
                self.Prediction = TorchDecoderWrapper(
                    d_model=opt.hidden_size, num_layers=opt.decoder_layers,
                    num_output=opt.num_class, embedding_dim=opt.hidden_size,
                    seq_length=opt.batch_max_length + 1,
                    learnable_embeddings=opt.learnable_pos_embeddings,
                )
            else:
                self.Prediction = TransformerDecoder(
                    learnable_embeddings=opt.learnable_pos_embeddings, num_output=opt.num_class, 
                    seq_length = opt.batch_max_length + 1,
                    embedding_dim=opt.hidden_size, dim_model=opt.hidden_size,
                    num_layers=opt.decoder_layers
                )


        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True, debug=False):
        batch_size = input.shape[0]
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        if debug:
            print(f'before pool{visual_feature.shape = }')
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        if debug:
            print(f'after pool{visual_feature.shape = }')
        visual_feature = visual_feature.squeeze(3)
            
        if debug:
            print(f'{visual_feature.shape = }')
            
        """ Sequence modeling stage """
        if self.stages['Seq'] not in ['BiLSTM']:
            visual_feature = visual_feature.permute(0, 2, 1)
            if debug:
                print(f' after permute{visual_feature.shape = }')
        contextual_feature = self.SequenceModeling(visual_feature)
        if self.stages['Seq'] not in ['BiLSTM']:
            contextual_feature = contextual_feature.permute(0, 2, 1)

        # if self.stages['Seq'] == 'BiLSTM':
        #     contextual_feature = self.SequenceModeling(visual_feature)
        # else:
        #     contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        if debug:
            print(f'{contextual_feature.shape = }')

        """ Prediction stage """
        if self.stages['Pred'] in ['CTC']:
            prediction = self.Prediction(contextual_feature.contiguous())
        elif self.stages['Pred'] in ['TransformerDecoder']:
            # target_tensor = torch.Tensor(
            #     [[0] * self.opt.batch_max_length for _ in range(batch_size)]
            # ).int()
            # + 1 because it'll be first text.... <EOS>
            if type(self.Prediction) is TransformerDecoder:
                mask = self.Prediction.generate_attn_mask(self.opt.batch_max_length + 1)
            else:
                mask = nn.Transformer.generate_square_subsequent_mask(self.Prediction.seq_length, device= 'cuda' if torch.cuda.is_available() else 'cpu')

            # mask = torch.ones((1,1))
            target_tensor = text

            if debug:
                print(f'{target_tensor.shape = }')
                print(f'{mask.shape = }')

            prediction = self.Prediction(target_tensor, contextual_feature.contiguous(), mask, debug=debug)
            # prediction = self.Prediction(contextual_feature.contiguous(), target_tensor, mask)
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

def load_model(opt):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        state_dict = torch.load(opt.saved_model, map_location=device)
        if opt.FT:
            last_layer_params = [
                "module.Prediction.generator.weight",
                "module.Prediction.generator.bias",
                "module.Prediction.attention_cell.rnn.weight_ih"
            ]
            state_dict = {k: v for k,v in state_dict.items() if k not in last_layer_params}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    print("Model:")
    print(model)
    return model, converter