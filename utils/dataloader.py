from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class MaskDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, mode, mosaic_ratio = 0.9):
        super(MaskDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.train              = train

        self.step_now           = -1
        self.mosaic_ratio       = mosaic_ratio
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        
        self.step_now += 1
        if self.mosaic:
            if self.rand() < 0.5 and self.step_now < self.epoch_length * self.mosaic_ratio * self.length:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
            else:
                image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        else:
            image, box, mask, num_vehicle  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        mask        = np.transpose(preprocess_input(np.array(mask, dtype=np.float32)), (2, 0, 1))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box, mask, num_vehicle

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line    = annotation_line.split()
        l = line[0] + ' ' + line[1]
        if self.mode > 0:
            m = "D:\\Houwang Zhang\\Detection\\Datasets\\BDD\\bdd100k\\images\\100k\\train_citystreet_labels\\" + line[1][55:]
        else:
            m = "D:\\Houwang Zhang\\Detection\\Datasets\\BDD\\bdd100k\\images\\100k\\val\\" + line[1][53:]
        
        num_obj = len(line[2:])
        num_car = 0
        num_bus = 0
        num_truck = 0
        num_bike = 0
        num_motor = 0
        for i in np.arange(num_obj):
            obj_ve = line[i + 2]
            if obj_ve[-1] == '0': num_car += 1
            if obj_ve[-1] == '1': num_truck += 1
            if obj_ve[-1] == '2': num_bus += 1
            if obj_ve[-1] == '3': num_motor += 1
            if obj_ve[-1] == '4': num_bike += 1
            
            # print("\n\n", obj_ve)
        num_vehicle = [num_car, num_truck, num_bus, num_motor, num_bike]
        
        image   = Image.open(l)
        image   = cvtColor(image)
        
        mask = Image.open(m)
        mask   = cvtColor(mask)
        iw, ih  = image.size
        h, w    = input_shape
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            
            mask       = mask.resize((nw,nh), Image.BICUBIC)
            # new_mask   = Image.new('RGB', (w,h), (128,128,128))
            new_mask   = Image.new('RGB', (w,h), (0,0,0))
            new_mask.paste(mask, (dx, dy))
            mask_data  = np.array(new_mask, np.float32)

            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box, mask_data, num_vehicle
                
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        mask = mask.resize((nw,nh), Image.BICUBIC)

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        
        # new_mask = Image.new('RGB', (w,h), (128,128,128))
        new_mask = Image.new('RGB', (w,h), (0,0,0))
        new_mask.paste(mask, (dx, dy))
        mask = new_mask

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if flip: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        
        image_data  = np.array(image, np.float32)
        mask_data   = np.array(mask, np.float32)

        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box, mask_data, num_vehicle
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

def mask_dataset_collate(batch):
    images = []
    bboxes = []
    masks  = []
    num_vehicles = []
    for img, box, mask, num_vehicle in batch:
        images.append(img)
        bboxes.append(box)
        masks.append(mask)
        num_vehicles.append(num_vehicle)
    images = np.array(images)
    masks  = np.array(masks)
    return images, bboxes, masks, num_vehicles
