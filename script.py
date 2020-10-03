import os


def extract_all_failed_imgaes():
    folder = './failed'
    failed_image_names = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith('CHS'):
                failed_image_names.append(file)
    return failed_image_names
