import os

class Parameters ():
    def __init__(self):

        # Directory Params
        self.checkpoints_dir = './checkpoints'
        self.name = 'deploy'
        
        # Model Hyperparams
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.netG = 'unet_512'
        self.norm = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = False
        self.load_size = 512
        self.crop_size = 512
        self.preprocess = 'none'
        self.no_flip = True

        # System Params
        
        self.gpu_ids = '-1'
        self.num_threads = 0
        self.batch_size = 1

