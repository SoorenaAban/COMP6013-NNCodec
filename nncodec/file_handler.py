#filer_handler.py
from . import version_code as version

class CompressedFile:
    def __init__(self):
        self.file_signature = b'NNC'
        pass

    def write(self, file_path, data):
        with open(file_path, 'wb') as file:
            file.write(self.file_signature)
            #write the version in 2 bytes
            file.write(version.to_bytes(2, 'big'))
            #write the data
            file.write(data)

    def read(self, file_path):
        with open(file_path, 'rb') as file:
            #check the signature
            if file.read(3) != self.file_signature:
                raise ValueError("Invalid file signature")
            #read the version
            file_version = int.from_bytes(file.read(2), 'big')
            if file_version != version:
                raise ValueError("incompatible version")
            #read the data
            data = file.read()
        return version, data