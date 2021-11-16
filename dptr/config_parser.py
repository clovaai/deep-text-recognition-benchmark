
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import argparse
import configparser
class ConfigParser:

    def __init__(self):
        ## migrate to config parser
        self.config = configparser.ConfigParser()

    def initialize_config(self, config_file):
        self.config.read(config_file)
        parser = argparse.ArgumentParser()
 
        opt = parser.parse_args()
        opt.image_folder = self.config['App']['image_folder']
        opt.image_path = self.config['App']['image_path']
        opt.workers = int(self.config['App']['workers'])
        opt.batch_size = int(self.config['App']['batch_size'])
        opt.saved_model = self.config['App']['saved_model']
        
        opt.batch_max_length = int(self.config['Model']['batch_max_length'])
        opt.imgH = int(self.config['Model']['imgH'])
        opt.imgW = int(self.config['Model']['imgW'])
        opt.rgb = self.config['Model'].getboolean('rgb')
        opt.character = self.config['Model']['character']
        opt.sensitive = self.config['Model'].getboolean('sensitive')
        opt.PAD = self.config['Model'].getboolean('PAD')
        opt.Transformation = self.config['Model']['Transformation']
        opt.FeatureExtraction = self.config['Model']['FeatureExtraction']
        opt.SequenceModeling = self.config['Model']['SequenceModeling']
        opt.Prediction = self.config['Model']['Prediction']
        opt.num_fiducial = int(self.config['Model']['num_fiducial'])
        opt.input_channel = int(self.config['Model']['input_channel'])
        opt.output_channel = int(self.config['Model']['output_channel'])
        opt.hidden_size = int(self.config['Model']['hidden_size'])

        """ vocab / character number configuration """
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        return opt



 # parser.add_argument('--image_folder', required= False, help='path to image_folder which contains text images')
    # parser.add_argument('--image_path', required = False, help = 'path to image for testing')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    # """ Data processing """
    # parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    # parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    # parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    # parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    # parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    # """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    # parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    # parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    # parser.add_argument('--output_channel', type=int, default=512,
    #                     help='the number of output channel of Feature extractor')
    # parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # opt = parser.parse_args()

    # """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    # cudnn.benchmark = True
    # cudnn.deterministic = True
    # opt.num_gpu = torch.cuda.device_count()