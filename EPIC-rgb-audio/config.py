'''
# train on 4 NVIDIA 1080Ti GPUs
Arguments
'''


class config_func():
    def __init__(self, source_domain, target_domain):
        self.lr_1stStage = 1e-4
        if source_domain == 'D1':
            self.lr_1stStage = 2e-4

        elif source_domain == 'D2':
            self.lr_1stStage = 9e-5


class GlobalPara():
    def __init__(self, root_path='/kaggle/working/', device='cpu') -> None:
        self.device = device
        self.base_path = root_path
        self.pkl_path = root_path + 'pkl/'
        self.model_ckpt = root_path + 'model/'
        self.wav_path = root_path + 'AudioVGGSound/'
        self.rgb_flow = root_path + 'frames_rgb_flow/'
        self.video_path = ''
