## This application initiates a TRBA model based on Deep-Text-Recognition-Framework for a single
## image inference. 



import argparse


if __name__ == '__main__':
    
    from dptr.trbaOcr import TrbaOCR 
    
    saved_model = 'models/TPS-ResNet-BiLSTM-Attn.pth'
    trbaOCR = TrbaOCR(saved_model)    

    image_path = 'demo_image/demo_1.png'
    trbaOCR.img_to_ean(image_path)

