import os
import argparse
import math
import numpy as np

from predict_with_sliding_window import Predict

'''
'''

class Generic_Test(Predict):
    def __init__(self, model, opts, test_dataloader):
        self.model = model
        self.opts = opts
        self.test_dataloader = test_dataloader

    def test(self):

        self.opts.output_patch_size = self.opts.model_train_size
        if not self.opts.is_upsample_landcover:
            self.opts.output_patch_size = self.opts.model_train_size * 3 / 10
            assert self.opts.output_patch_size.is_integer()
            self.opts.output_patch_size = int(self.opts.output_patch_size)

        self.opts.output_size = self.opts.load_size
        if not self.opts.is_upsample_landcover:
            self.opts.output_size = self.opts.load_size * 3 / 10
            assert self.opts.output_size.is_integer()
            self.opts.output_size = int(self.opts.output_size)

        self.model.net_G.eval()
        for _, param in self.model.net_G.named_parameters():
            param.requires_grad = False

        _iter = 0
        semantic_metric = StreamSegMetrics(self.model.num_classes)
        semantic_metric.reset()

        with torch.no_grad():
            for data in self.test_dataloader:

                self.model.set_input(data)

                pred_landcover_data = self.predict_landcover(optical_data=self.model.cloudy_data,
                                                             SAR_data=self.model.SAR_data)

                cloudmask_data = self.model.cloudmask_data.cpu().numpy()
                if not self.opts.is_upsample_landcover:
                    cloudmask_data = F.interpolate(self.model.cloudmask_data.unsqueeze(1),
                                                   size=[self.opts.output_size, self.opts.output_size],
                                                   mode='nearest',
                                                   align_corners=None).squeeze(1).cpu().numpy()

                semantic_metric.update(self.model.landcover_data.cpu().numpy(),
                                       pred_landcover_data.detach().max(dim=1)[1].cpu().numpy(),
                                       cloudmask_data)

                _iter += 1
                if _iter % 10 == 0:
                    print(f'{_iter} samples have been processed!')

        print('Testing done. ')

        score = semantic_metric.get_results(semantic_metric.confusion_matrix, True)
        print('All Regions\n', semantic_metric.to_str(score))

        score = semantic_metric.get_results(semantic_metric.confusion_matrix_cloudfree, True)
        print('Cloud-Free Regions\n', semantic_metric.to_str(score))

        score = semantic_metric.get_results(semantic_metric.confusion_matrix_cloudy, True)
        print('Cloudy Regions\n', semantic_metric.to_str(score))

    def predict_landcover(self, optical_data, SAR_data):

        # if crop_size = model_train_size: predict directly
        # if crop_size > model_train_size: predict with sliding window

        if self.opts.crop_size == self.opts.model_train_size:
            pred_landcover_data = self.model.forward(optical_data=optical_data,
                                                     SAR_data=SAR_data,
                                                     output_shape=[self.opts.output_patch_size, self.opts.output_patch_size])
            return pred_landcover_data
        elif self.opts.crop_size < self.opts.model_train_size:
            raise NotImplementedError("This function is not implemented yet.")

        return self._predict_with_sliding_window(optical_data, SAR_data)


'''
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_folder', type=str, default='/remote-home/xufang/data/RS/M3M-CR/test')
    parser.add_argument('--is_load_SAR', type=bool, default=True)
    parser.add_argument('--is_upsample_SAR', type=bool, default=True)  # only useful when is_load_SAR = True
    parser.add_argument('--is_load_landcover', type=bool, default=True)
    parser.add_argument('--is_upsample_landcover', type=bool, default=False)  # only useful when is_load_landcover = True
    parser.add_argument('--lc_level', type=str, default='1')  # only useful when is_load_landcover = True
    parser.add_argument('--is_load_cloudmask', type=bool, default=True)
    parser.add_argument('--load_size', type=int, default=300)
    parser.add_argument('--crop_size', type=int, default=300)
    parser.add_argument('--model_train_size', type=int, default=160)

    parser.add_argument('--test_list_filepath', type=str, default='../M3R-CR/csv/test.csv')
    parser.add_argument('--pretrained_model', type=str, default='../checkpoints/StudentNet.pth')
    parser.add_argument('--gpu_ids', type=str, default='2')

    opts = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    from model_SS_net import ModelSSNet
    import torch
    from torch.nn import functional as F
    from metrics import StreamSegMetrics
    from dataloader import get_filelist, ValDataset

    test_filelist = get_filelist(opts.test_list_filepath)
    test_data = ValDataset(opts, test_filelist)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    print(f'Test set: {len(test_data)}')

    model = ModelSSNet(opts)
    Generic_Test(model, opts, test_dataloader).test()
