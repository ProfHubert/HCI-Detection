import torch
from tqdm import tqdm
from torchvision import transforms
from utils.utils import get_lr
import torch.nn.functional as F
import numpy as np

toPIL = transforms.ToPILImage()
itrs = np.arange(100) 
criterion = torch.nn.BCELoss()  
loss_num = torch.nn.MSELoss()

def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    loss        = 0
    val_loss    = 0
    ns          = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets, masks, num_v = batch[0], batch[1], batch[2], batch[3]
            num_v = np.array(num_v)
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    masks  = torch.from_numpy(masks).type(torch.FloatTensor).cuda()
                    num_v = torch.from_numpy(num_v).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    masks  = torch.from_numpy(masks).type(torch.FloatTensor)
                    num_v = torch.from_numpy(num_v).type(torch.FloatTensor)
            
            masks = masks[:, 0, :, :]
            masks = masks.unsqueeze(1)
            
            # pic = toPIL(masks[0, :, :, :])
            # pic.save('e.jpg')
            
            # pic = toPIL(images[0, :, :, :])
            # pic.save('g.jpg')
            
            optimizer.zero_grad()
            outputs, ms, qf  = model_train(images)

            
            num_v[:, 0] = num_v[:, 0] / 34 # the car_max: 34
            num_v[:, 1] = num_v[:, 1] / 16 # the truck_max: 16
            num_v[:, 2] = num_v[:, 2] / 7  # the bus_max: 7
            num_v[:, 3] = num_v[:, 3] / 7  # the motor_max: 7
            num_v[:, 4] = num_v[:, 4] / 25 # the bike_max: 25
            
            loss_global = loss_num(qf, num_v)
        
            loss_mask = torch.mean(torch.pow(masks - ms, 2))

            loss_det = yolo_loss(outputs, targets) 
            
            loss_value = loss_det + loss_mask*10 + loss_global*20
            
            if epoch in itrs and ns < 1:
                pic = toPIL(images[0, :, :, :])
                pic.save('figs/ori_%d.jpg' % epoch)
                
                pic = toPIL(masks[0, :, :, :])
                pic.save('figs/mask_%d.jpg' % epoch)
                
                pic = toPIL(ms[0, :, :, :])
                pic.save('figs/out_%d.jpg' % epoch)
                ns += 1

            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets, masks, num_v = batch[0], batch[1], batch[2], batch[3]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    masks  = torch.from_numpy(masks).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    masks  = torch.from_numpy(masks).type(torch.FloatTensor)
                    
                    
                masks = masks[:, 0, :, :]
                masks = masks.unsqueeze(1)
                
                # pic = toPIL(masks[0, :, :, :])
                # pic.save('ve.jpg')
                
                # pic = toPIL(images[0, :, :, :])
                # pic.save('vg.jpg')
                
                optimizer.zero_grad()
                
                outputs, ms, qf = model_train(images)
                loss_value = yolo_loss(outputs, targets)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
