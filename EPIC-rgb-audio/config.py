'''
# train on 4 NVIDIA 1080Ti GPUs
Arguments
'''


from collections import OrderedDict
import torch


class config_func():
    def __init__(self, source_domain, target_domain):
        self.lr_1stStage = 1e-4
        if source_domain == 'D1':
            self.lr_1stStage = 2e-4

        elif source_domain == 'D2':
            self.lr_1stStage = 9e-5


class GlobalPara():
    """
    set global arguments
    """
    def __init__(self, root_path='/kaggle/working/', gpu='cuda:0') -> None:
        self.device = gpu if torch.cuda.is_available() else 'cpu'
        self.base_path = root_path
        self.pkl_path = root_path + 'pkl/'
        self.model_ckpt = root_path + 'model/'
        self.wav_path = root_path + 'AudioVGGSound/'
        self.rgb_flow = root_path + 'frames_rgb_flow/'
        self.video_path = ''


def load_para(model_, ckp):
    """
    load parameters for models. mainly from cpu
    """
    # print("Checkpoint keys:", list(checkpoint['state_dict'].keys())[0])
    # print("Model keys:", list(model.state_dict().keys())[0])
    # assuming that 'module.' is the prefix in the keys causing the mismatch
    new_state_dict = OrderedDict()
    for k, v in ckp.items():
        name = k[7:]  # remove 'module.' prefix
        new_state_dict[name] = v

    # load the corrected state_dict
    model_.load_state_dict(new_state_dict)
