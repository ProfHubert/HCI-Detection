import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import MaskDataset, mask_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    Cuda            = True
    classes_path    = 'model_data/cls_classes.txt'
    model_path      = 'model_data/yolox_s.pth'
    input_shape     = [640, 640]
    phi             = 's' # yolo version: s m l
    mosaic              = False
    Cosine_scheduler    = False

    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 32
    Freeze_lr           = 1e-3
    UnFreeze_Epoch      = 60
    Unfreeze_batch_size = 5 # s-8; m-5
    Unfreeze_lr         = 1e-4
    Freeze_Train        = True
    num_workers         = 4
    train_annotation_path   = 'train_citystreet.txt'
    val_annotation_path     = 'val_citystreet.txt'

    class_names, num_classes = get_classes(classes_path)

    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v) and 'dark' in k} # backbone or dark
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/")

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_scheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = MaskDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic = mosaic, train = True, mode = 1)
        val_dataset     = MaskDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic = False, train = False, mode = 0)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=mask_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=mask_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_scheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = MaskDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic = mosaic, train = True, mode = 1)
        val_dataset     = MaskDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic = False, train = False, mode = 0)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=mask_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=mask_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
