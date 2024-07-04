import os
import argparse

if __name__ == "__main__":
    ##===================================================##
    ##********** Configure training settings ************##
    ##===================================================##
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=32, help='batch size used for training')

    parser.add_argument('--input_data_folder', type=str, default='/remote-home/xufang/data/RS/M3R-CR/train')
    parser.add_argument('--train_list_filepath', type=str, default='../M3R-CR/csv/train.csv')
    parser.add_argument('--val_list_filepath', type=str, default='../M3R-CR/csv/val.csv')
    parser.add_argument('--is_load_SAR', type=bool, default=True)
    parser.add_argument('--is_upsample_SAR', type=bool, default=True) # only useful when is_load_SAR = True
    parser.add_argument('--is_load_landcover', type=bool, default=True)
    parser.add_argument('--is_upsample_landcover', type=bool, default=False) # only useful when is_load_landcover = True
    parser.add_argument('--lc_level', type=str, default='1')  # only useful when is_load_landcover = True
    parser.add_argument('--is_load_cloudmask', type=bool, default=False)
    parser.add_argument('--load_size', type=int, default=300)
    parser.add_argument('--crop_size', type=int, default=160)

    parser.add_argument('--optimizer', type=str, default='Adam', help = 'Adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
    parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
    parser.add_argument('--lr_start_epoch_decay', type=int, default=10, help='epoch to start lr decay')
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=2)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--save_model_dir', type=str, default='../checkpoints/TeacherNet')

    parser.add_argument('--continue_train_checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--gpu_ids', type=str, default='1')

    opts = parser.parse_args()

    ##===================================================##
    ##****************** choose gpu *********************##
    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    ##===================================================##
    ##**********************  base  *********************##
    ##===================================================##
    import torch
    from dataloader import get_filelist, TrainDataset, ValDataset
    from model_SS_net import ModelSSNet
    from generic_train import Generic_Train
    from model_base import print_options, seed_torch
    print_options(opts)

    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    seed_torch()

    train_filelist = get_filelist(opts.train_list_filepath)
    val_filelist = get_filelist(opts.val_list_filepath)

    train_data = TrainDataset(opts, train_filelist)
    val_data = ValDataset(opts, val_filelist)
    print("Train set: %d, Val set: %d" % (len(train_data), len(val_data)))

    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_sz,shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=opts.batch_sz,shuffle=False, num_workers=4, drop_last=True)

    ##===================================================##
    ##****************** Create model *******************##
    ##===================================================##
    model=ModelSSNet(opts)

    ##===================================================##
    ##**************** Train the network ****************##
    ##===================================================##
    Generic_Train(model, opts, train_dataloader, val_dataloader).train()
