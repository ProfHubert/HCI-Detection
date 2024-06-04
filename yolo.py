import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression

from torchvision import transforms
toPIL = transforms.ToPILImage()


class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/trained model.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        "input_shape"       : [640, 640],
        "phi"               : 's',
        "confidence"        : 0.5,
        "nms_iou"           : 0.5,
        "letterbox_image"   : True,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
    def generate(self):
        self.net    = YoloBody(self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            outputs, ms, qf = self.net(images)
            cc = ms[:, :, 120:480, :]
            pic = toPIL(cc[0, :, :, :])
            pic.save('mask_d.jpg')
            
            msk = ms.clone()
            msk[ms < ms.mean()] = 0
            pic = toPIL(msk[0, :, :, :])
            pic.save('mask_k1.jpg')
            
            outputs = decode_outputs(outputs, self.input_shape)
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_seg(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs, ms, qf = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            qfs   = qf[0]
            qfs[0] = qfs[0] * 49 # the car_max: 49
            qfs[1] = qfs[1] * 7  # the truck_max: 7
            qfs[2] = qfs[2] * 6  # the bus_max: 6
            qfs[3] = qfs[3] * 3  # the motor_max: 3
            qfs[4] = qfs[4] * 25 # the bike_max: 3
            
        return ms, qfs

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                  
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs, ms, qf = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            results = non_max_suppression(outputs.clone(), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                             
            msk = ms.clone()
            msk[ms < ms.mean()] = 0 # threshold?
            qfs   = qf[0]
            qfs[0] = qfs[0] * 49 # the car_max: 49
            qfs[1] = qfs[1] * 7  # the truck_max: 7
            qfs[2] = qfs[2] * 6  # the bus_max: 6
            qfs[3] = qfs[3] * 3  # the motor_max: 3
            ref_results = non_max_suppression(outputs.clone(), self.num_classes, self.input_shape, 
                        self.input_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                             
                       
            if results[0] is None: 
                return

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
            
            ref_boxes   = ref_results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            ref_box         = ref_boxes[i]
            score           = top_conf[i]
            # score           = str(top_conf[i])

            top, left, bottom, right = box
            
            rtop, rleft, rbottom, rright = ref_box

            rtop     = max(0, np.floor(rtop).astype('int32'))
            rleft    = max(0, np.floor(rleft).astype('int32'))
            rbottom  = min(image.size[1], np.floor(rbottom).astype('int32'))
            rright   = min(image.size[0], np.floor(rright).astype('int32'))
            
            ms_cut = torch.zeros_like(msk)
            img_cut = torch.zeros_like(images)
            
            ms_cut[:, :, rtop:rbottom, rleft:rright] = ms[:, :, rtop:rbottom, rleft:rright]
            img_cut[:, :, rtop:rbottom, rleft:rright] = images[:, :, rtop:rbottom, rleft:rright]
            
            # pic = toPIL(img_cut[0, :, :, :])
            # pic.save('imgcut.jpg')
            # pic = toPIL(ms_cut[0, :, :, :])
            # pic.save('ms_cut.jpg')
            
            ms_cut[ms_cut > 0] = 1
            mask_cut = ms_cut.sum()
            area_cut = (rbottom - rtop) * (rright - rleft)
            radio_cut = mask_cut / area_cut
            
            if radio_cut > 0.5 and predicted_class == 'car' and score < 0.001:
                score = score + 0.001
            if radio_cut < 0.5 and score < 0.001: # ref_thres = conf_thres / 10 = 0.0001
                score = score - 0.001
                # continue
            
            if predicted_class not in class_names:
                continue

            score           = str(score)
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
