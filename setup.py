from setuptools import setup, find_packages




setup(

    name ="dptr", 
    version = '0.1',    
    packages = find_packages(),     
    install_requires = ['numpy', 'lmdb', 'natsort', 'Pillow', 'six', 'torch', 'torchvision','typing-extensions'], 
    
    
    description = "A package for TRBA model for OCR",
)
