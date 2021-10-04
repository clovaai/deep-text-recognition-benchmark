from setuptools import setup


setup(

    name ='deepTextRecognitionBenchmark', 
    version = '0.1',
    url='https://github.com/raki-dedigama', 
    author = 'Raki Dedigama', 
    author_email = 'rakkitha.da@gmail.com', 
    packages = ['deepTextRecognitionBenchmark'], 
    install_requires = ['numpy', 'lmdb', 'natsort', 'Pillow', 'six', 'torch', 'torchvision','typing-extensions'], 
    
    license = 'MIT', 
    description = "A package for TRBA model for OCR",
)