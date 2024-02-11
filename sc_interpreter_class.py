from my_os import *

class sc_interpreter_class:
    def __init__(self):
        self.data_dict = open_folder()
        self.curdir = os.curdir
        self.sort_data_dict()
