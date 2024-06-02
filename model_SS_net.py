import os
import torch
import torch.nn as nn

from model_base import ModelBase

from SSUCC_net import SSUCCNet

from torch.optim import lr_scheduler

class ModelSSNet(ModelBase):
    def __init__(self, opts):
        super(ModelSSNet, self).__init__()
        self.opts = opts

        # assign value to self.num_classes
        if self.opts.lc_level == '1':
            self.num_classes = 6
        elif self.opts.lc_level == '2':
            self.num_classes = 11
        
        # create network
        self.net_G = SSUCCNet(encoder_name='mit_b4',
                              encoder_weights='imagenet',
                              classes=self.num_classes).cuda()
        self.net_G = nn.DataParallel(self.net_G)

        # load pre-trained model
        if self.opts.pretrained_model is not None:
            checkpoint = torch.load(self.opts.pretrained_model)
            self.net_G.load_state_dict(checkpoint['network'])

        # initialize optimizers
        # If self.opts.optimizer exists, the default is the training phase
        if hasattr(self.opts, 'optimizer'): 

            self.print_networks(self.net_G)

            # load model trained on cloud-free images
            from SSUCC_net_CloudFree import SSUCC_net_CloudFree
            self.net_cloudfree_G = SSUCC_net_CloudFree(encoder_name='mit_b4',
                                                       encoder_weights='imagenet',
                                                       classes=self.num_classes).cuda()
            self.net_cloudfree_G = nn.DataParallel(self.net_cloudfree_G)
            checkpoint = torch.load('checkpoints/teacher_net.pth')
            self.net_cloudfree_G.load_state_dict(checkpoint['network'])
            self.net_cloudfree_G.eval()
            for _,param in self.net_cloudfree_G.named_parameters():
                param.requires_grad = False

            self.output_patch_size = self.opts.crop_size 
            if not self.opts.is_upsample_landcover:
                self.output_patch_size = self.opts.crop_size * 3 / 10
                assert self.output_patch_size.is_integer()
                self.output_patch_size = int(self.output_patch_size)
            
            if self.opts.optimizer == 'Adam':
                self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.opts.lr)

            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
            
            self.loss_SS_fn = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            from losses import ECCharbonnierLoss
            self.loss_CR_fn = ECCharbonnierLoss()
            self.CR_weight = 1.0
            from losses import MaskHint
            self.loss_KD_fn = MaskHint()
            self.KD_weight = 1.

            if self.opts.continue_train_checkpoint is not None:
                checkpoint = torch.load(self.opts.continue_train_checkpoint)
                self.net_G.load_state_dict(checkpoint['network'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                self.start_epoch = -1
                        
    def set_input(self, inputs):
        self.cloudy_data = inputs['cloudy_data'].cuda()
        self.cloudfree_data = inputs['cloudfree_data'].cuda()
        self.SAR_data = inputs['SAR_data'].cuda() if self.opts.is_load_SAR else None
        self.cloudmask_data = inputs['cloudmask_data'].cuda() if self.opts.is_load_cloudmask else None
        self.landcover_data = inputs['landcover_data'].long().cuda() if self.opts.is_load_landcover else None
        if not hasattr(self.opts, 'optimizer'):
          self.file_name = inputs['file_name']
          if 'crop_params' in inputs:
              self.crop_params = inputs['crop_params']
          if 'SAR_path' in inputs:
              self.SAR_path = inputs['SAR_path']
        else:
          self.file_name = None
        
    def forward(self, optical_data=None, SAR_data=None, output_shape=None, is_train=False):
        self.pred_landcover_data, self.pred_cloudfree_data, self.pred_feats = self.net_G(optical_data, 
                                                                                         SAR_data, 
                                                                                         output_shape)
        if is_train:
            self.lc_from_cloudfree, self.feats_from_cloudfree = self.net_cloudfree_G(self.cloudfree_data,
                                                                                     SAR_data,
                                                                                     output_shape)
        return self.pred_landcover_data

    def optimize_parameters(self):     
        self.forward(optical_data=self.cloudy_data, 
                     SAR_data=self.SAR_data, 
                     output_shape=[self.output_patch_size, self.output_patch_size],
                     is_train=True)

        loss_SS = self.loss_SS_fn(self.pred_landcover_data, self.landcover_data)
        loss_CR = self.loss_CR_fn(self.pred_cloudfree_data, self.cloudfree_data, self.cloudmask_data)
        loss_KD = self.loss_KD_fn(self.pred_feats["cross_modal_decoder_out"], 
                                  self.feats_from_cloudfree["cross_modal_decoder_out"].detach(), 
                                  self.cloudmask_data) 

        self.loss_total = loss_SS + self.CR_weight*loss_CR + self.KD_weight*loss_KD

        self.optimizer_G.zero_grad()
        self.loss_total.backward()
        self.optimizer_G.step()  

        return self.loss_total.item()

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.opts.save_model_dir, '%s_net.pth' % (str(epoch))))
        self.net_G.load_state_dict(checkpoint['network'])
