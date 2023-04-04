import datetime
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

import glob
import cpnCascade
import skimage
import skimage.transform
import imageio
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import torch.utils.data as data
import cv2
import json
import numpy as np
import math
import logging
from PIL import Image
from display import show_image

sys.path.insert(0, 'src/cpn/')
print(sys.path)
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from tqdm import tqdm

from pipeline_utils import plot_skeleton_2d
# gpuUsage


class cpn:
    def __init__(self):
        """
        All cpn processes that only need to occur once
        """
        start = time.time()
        
        self.PLOT_TEST = True   #True
               
        def add_pypath(path):
            if path not in sys.path:
                sys.path.insert(0, path)

        class Config:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            this_dir_name = cur_dir.split('/')[-1]
            root_dir = os.path.join(cur_dir, '..')

            model = 'CPN101' # option 'CPN50', 'CPN101'

            num_class = 17
#             img_path = args.source

            symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            bbox_extend_factor = (0.1, 0.15) # x, y

            pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
            data_shape = (384, 288) #height, width
            output_shape = (96, 72) #height, width

        self.cfg = Config()
        add_pypath(self.cfg.root_dir)
        add_pypath(os.path.join(self.cfg.root_dir, 'cocoapi/PythonAPI'))
        
        self.flipped_y = -1   #if y-axis is flipped, set to -1. If not, set to 1.
        self.checkpoint = 'src/cpn/checkpoint'
        self.test = 'CPN101_384x288'
        
        # create model
        self.model = network.__dict__[self.cfg.model](self.cfg.output_shape, self.cfg.num_class, pretrained = False)
        # self.model = torch.nn.DataParallel(self.model).cuda()
        self.model = torch.nn.DataParallel(self.model)

        print(self.model)
        # load trainning weights
        dev = torch.device('cpu')
        checkpoint_file = os.path.join(self.checkpoint, self.test+'.pth.tar')
        checkpoint = torch.load(checkpoint_file, map_location=dev)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

        # change to evaluation mode
        self.model.eval()

        self.new_cascade = cpnCascade.cpnCascade(cpnModel=self.model)
        torch.save(self.new_cascade.state_dict(), 'src/cpn/new_model.th')

        ckpt = torch.load('src/cpn/new_model.th')
        print(ckpt)
        self.new_cascade.load_state_dict(ckpt)
        self.new_cascade.eval()
        print(self.new_cascade)


        print("Putting in new Model done")

        torch.save(self.model.state_dict(), 'src/cpn/model.pt')
        logging.info("CPN init time: {:.3f}".format((time.time()-start)))

    
    
    def preprocess(self):
        pass
    
    def detect(self, crop):
        """
        All cpn processes that occur once for every frame
        """
        logging.info("Image resolution: " + str(crop.shape[1]) + " x " + str(crop.shape[0]))

        def to_numpy(tensor):
            return tensor.cpu().detach().numpy() 
            
            # if tensor.requires_grad else tensor.cpu().numpy() 

        def export(input_Var, inputs):
            import onnx
            import onnxruntime
            import onnxmltools
            """
            Converts from EvoSkeleton output to standard joint order & swaps dimension order from (x,z,y) to (x,y,z)
            """
            # # torch_out = torch_out(x)
            self.model.eval()
            inputVar = torch.randn(1, 3, 384, 288, requires_grad=True, device = 'cpu')

            torch_out = self.model(input_Var)
            print("Torch out", torch_out)

            cpn_torch = torch_out[0]
            cpn_list = []
            for v in cpn_torch:
                cpn_list.append(to_numpy(v))

            cpn_list.append(to_numpy(torch_out[1]))
            cpn_list = np.array(cpn_list)

            torch_out = torch_out[0][0]
            torch_out = torch_out.cpu().detach().numpy() 
            batch_size = 1
            torch.onnx.export(self.model.module,               # model being run
                            input_Var,                         # model input (or a tuple for multiple inputs)
                            "src/cpn/cpn_skeleton.onnx",   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=11,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'],
                            dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
            
            print("put in model done")

            onnx_model = onnx.load("src/cpn/cpn_skeleton.onnx")
            onnx.checker.check_model(onnx_model)

            

            ort_session = onnxruntime.InferenceSession("src/cpn/cpn_skeleton.onnx")

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_Var)}
            ort_outs = ort_session.run(None, ort_inputs)
            # print("Onnx output", ort_outs[0])
            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-02, atol=1e-03)

            print("Exported model has been tested with ONNXRuntime, and the result looks good!")



            # refine_output = ort_outs[0]
            score_map = ort_outs[0]
            # score_map = score_map.numpy()
            logging.info("Prediction Retrieved")

#             det_scores = meta['det_scores']
            for b in range(inputs.size(0)):
                details = meta['augmentation_details']
                single_result_dict = {}
                single_result = []

                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
#                 v_score = np.zeros(17)   #Measure of confidence score
                for p in range(17): 
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((self.cfg.output_shape[0] + 2*border, self.cfg.output_shape[1]+2*border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, self.cfg.output_shape[1] - 1))
                    y = max(0, min(y, self.cfg.output_shape[0] - 1))
                    resy = float((4 * y + 2) / self.cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float((4 * x + 2) / self.cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
#                     v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                    single_result.append(resx)
                    single_result.append(resy)
#                     single_result.append(x)   #This version ignored augmentation details, resulting in different scaling & translation
#                     single_result.append(y)   #This version ignored augmentation details, resulting in different scaling & translation
                    single_result.append(1)   
                if len(single_result) != 0:
    #                         single_result_dict['image_id'] = int(ids[b])
                    single_result_dict['keypoints'] = single_result
    #                         single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                    full_result.append(single_result_dict)
            
        
            #Note: This only works for single person detection. This code only saves the highest confidence detection.
            logging.info("People detected: " + str(len(full_result)))
            keypoints = full_result[0]['keypoints']
            logging.info("CPN detection time: {:.3f}".format((time.time()-start)))

            #########################################################################################################
            import onnx

            # Load the ONNX model
            onnx_model = onnx.load("src/cpn/cpn_skeleton.onnx")

            # Run shape inference on the model
            model_inferred = onnx.shape_inference.infer_shapes(onnx_model)

            # Get the output shape
            output_shape = model_inferred.graph.output[0].type.tensor_type.shape

            # Print the output shape
            print("Onnx Output Shape:", output_shape)

            import coremltools as ct

            mlmodel = ct.converters.onnx.convert(
            model = 'src/cpn/cpn_skeleton.onnx',
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',    
            )

            print("Done")

            mlmodel.save("src/cpn/cpn_skeleton.mlmodel")

            coreml_model = ct.models.MLModel("src/cpn/cpn_skeleton.mlmodel")

            output_feature = coreml_model.get_spec().description.output[0]

            # Change the shape of the output tensor to a fixed size
            output_feature.type.multiArrayType.shape[:] = [1,17,96,72]

            # Save the modified model
            coreml_model.save("src/cpn/modified_model.mlmodel")

            print("Resizing Done")

            import tvm
            from tvm.contrib import graph_runtime
            from tvm import relay
            from tvm.relay import frontend, transform

            
            # Load the ONNX model
            onnx_model = onnx.load('src/cpn/cpn_skeleton.onnx')
            input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
            output_shape = onnx_model.graph.output[0].type.tensor_type.shape.dim

            print(f"Input shape: {[dim.dim_value for dim in input_shape]}")
            print(f"Output shape: {[dim.dim_value for dim in output_shape]}")

            print("1")
            # Convert the ONNX model to a TVM relay graph
            mod, params = frontend.from_onnx(onnx_model)
            print("2")
            # Apply optimization passes to the relay graph
            mod = relay.transform.InferType()(mod)
            print("3")
            mod = relay.transform.SimplifyInference()(mod)
            print("4")
            mod = relay.transform.FoldConstant()(mod)
            # Specify the input shape
            print("4.5")
            # shape_dict = {"input": (1, 3, 384, 288)}
            buckets = [1]
            mod = tvm.relay.transform.FlexibleShapeDispatch(buckets = [1], axis = 0, input_indices = [0])(mod)
            print(mod)
            print("5")
            # Compile the relay graph to Core ML format
            model = tvm.contrib.xcode.compile_coreml(mod, model_name='tvm_gen', out_dir='src/cpn')
            print("6")
            # Create a TVM runtime module for the compiled Core ML model
            module = graph_runtime.create(model, tvm.cpu())
            print("7")
            # Set the input data
            module.set_input('input', input_Var)
            print("8")
            # Run the model
            module.run()
            print("9")
            # Get the output data
            output_data = module.get_output(0)
            print("10")
            # Save the Core ML model to disk
            ct.models.MLModel(model).save("src/cpn/tvm_model.mlmodel")

            print("TVM Ended")

            return keypoints

    #######################################################################
        def get_keypoints(input_Var):
            import onnx
            import onnxruntime
            onnx_model = onnx.load("src/cpn/cpn_skeleton.onnx")
            onnx.checker.check_model(onnx_model)

            ort_session = onnxruntime.InferenceSession("src/cpn/cpn_skeleton.onnx")
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_Var)}
            ort_outs = ort_session.run(None, ort_inputs)

            score_map = ort_outs[0]

            # logging.info("Prediction Retrieved")

            for b in range(inputs.size(0)):
                details = meta['augmentation_details']
                single_result_dict = {}
                single_result = []

                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
#                 v_score = np.zeros(17)   #Measure of confidence score
                for p in range(17): 
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((self.cfg.output_shape[0] + 2*border, self.cfg.output_shape[1]+2*border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, self.cfg.output_shape[1] - 1))
                    y = max(0, min(y, self.cfg.output_shape[0] - 1))
                    resy = float((4 * y + 2) / self.cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float((4 * x + 2) / self.cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
#                     v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                    single_result.append(resx)
                    single_result.append(resy)
#                     single_result.append(x)   #This version ignored augmentation details, resulting in different scaling & translation
#                     single_result.append(y)   #This version ignored augmentation details, resulting in different scaling & translation
                    single_result.append(1)   
                if len(single_result) != 0:
    #                         single_result_dict['image_id'] = int(ids[b])
                    single_result_dict['keypoints'] = single_result
    #                         single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                    full_result.append(single_result_dict)
            
        
            #Note: This only works for single person detection. This code only saves the highest confidence detection.
            logging.info("People detected: " + str(len(full_result)))
            keypoints = full_result[0]['keypoints']
            logging.info("CPN detection time: {:.3f}".format((time.time()-start)))

            return keypoints
            
        def augmentationCropImage(img, joints=None):  
            """
            This function crops the image based on the given bbox, then sets it to the standard 288x384 image size, and 
            saves augmentation details.
            
            The input and output images are both BGR.
            """
            inp_res = self.cfg.data_shape
            height, width = inp_res[0], inp_res[1]
            
            #Setting the bbox to the entire image
            bbox = [0, 0, img.shape[1], img.shape[0]]  #img.shape[1] is the width, img.shape[1] is the height 
            bbox = np.array(bbox).reshape(4, ).astype(np.float32)
            
            add = max(img.shape[0], img.shape[1])
            mean_value = self.cfg.pixel_means
            img = img.permute((2,0,1))
            
            # Manual padding
            cuda0 = torch.device('cpu')
            torch_means = torch.tensor((mean_value[0], mean_value[1], mean_value[2]), device=cuda0)
            # torch_means = torch.tensor((mean_value[0], mean_value[1], mean_value[2]))
            torch_means = torch_means[..., None, None]
            
            pad_top = torch_means.repeat(1, img.shape[1], add)            
            bimg = torch.cat((pad_top, img, pad_top), 2)
            
            pad_left = torch_means.repeat(1, add, bimg.shape[2])
            bimg = torch.cat((pad_left, bimg, pad_left), 1)           
#             bimg = torch.nn.functional.pad(img, (add, add, add, add), "constant", 0)       
            
            objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])      
            bbox += add
            objcenter += add
            crop_width = (bbox[2] - bbox[0]) * (1 + self.cfg.bbox_extend_factor[0] * 2)
            crop_height = (bbox[3] - bbox[1]) * (1 + self.cfg.bbox_extend_factor[1] * 2)
            if crop_height / height > crop_width / width:
                crop_size = crop_height
                min_shape = height
            else:
                crop_size = crop_width
                min_shape = width  

            crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
            crop_size = min(crop_size, (bimg.shape[2] - objcenter[0]) / width * min_shape * 2. - 1)
            crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
            crop_size = min(crop_size, (bimg.shape[1] - objcenter[1]) / height * min_shape * 2. - 1)

            min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
            max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
            min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
            max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)                               

            x_ratio = float(width) / (max_x - min_x)
            y_ratio = float(height) / (max_y - min_y)
            
#             img = cv2.resize(bimg[:, min_y:max_y, min_x:max_x], (width, height))
            bimg = torch.unsqueeze(bimg, 0)
            img = torch.nn.functional.interpolate(bimg[:, :, min_y:max_y, min_x:max_x].float(), [height, width], mode='bilinear', align_corners=False)
            details = torch.tensor([min_x - add, min_y - add, max_x - add, max_y - add], dtype=torch.float64)
                        
            return img, details
        
        def getitem(image):
            image = image[:, :, (2,1,0)]   #Convert input image to BGR (faster method)

            image, details = augmentationCropImage(image)
            
            img = torch.true_divide(image, 255) # From 0-255 to 0-1
            img = color_normalize(torch.squeeze(img), self.cfg.pixel_means)

            details = torch.unsqueeze(details, 0)
            meta = {'augmentation_details' : details}   #For now, the imgID doesn't matter since there's only 1 image

            meta['det_scores'] = 1.00   #Used to calculate a final confidence score for the skeleton
            
            img = torch.unsqueeze(img, 0)
            return img, meta
        
            
        start = time.time()
        inputs, meta = getitem(crop)
        full_result = []

        with torch.no_grad():
            # input_var = torch.autograd.Variable(inputs.cuda())
            input_var = torch.autograd.Variable(inputs)
            # onnx_keypoints = export(input_var,inputs)
            # print("Getting keypoints from ONNX:- ")
            onnx_keypoints = get_keypoints(input_var)
            # print(onnx_keypoints)
            # compute output
            global_outputs, refine_output = self.model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            print("Score Map", score_map)
            logging.info("Prediction Retrieved")

#             det_scores = meta['det_scores']
            for b in range(inputs.size(0)):
                details = meta['augmentation_details']
                single_result_dict = {}
                single_result = []

                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
#                 v_score = np.zeros(17)   #Measure of confidence score
                for p in range(17): 
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((self.cfg.output_shape[0] + 2*border, self.cfg.output_shape[1]+2*border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, self.cfg.output_shape[1] - 1))
                    y = max(0, min(y, self.cfg.output_shape[0] - 1))
                    resy = float((4 * y + 2) / self.cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float((4 * x + 2) / self.cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
#                     v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                    single_result.append(resx)
                    single_result.append(resy)
#                     single_result.append(x)   #This version ignored augmentation details, resulting in different scaling & translation
#                     single_result.append(y)   #This version ignored augmentation details, resulting in different scaling & translation
                    single_result.append(1)   
                if len(single_result) != 0:
    #                         single_result_dict['image_id'] = int(ids[b])
                    single_result_dict['keypoints'] = single_result
    #                         single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                    full_result.append(single_result_dict)
            
        
        #Note: This only works for single person detection. This code only saves the highest confidence detection.
        logging.info("People detected: " + str(len(full_result)))
        keypoints = full_result[0]['keypoints']
        logging.info("CPN detection time: {:.3f}".format((time.time()-start)))
        print("Torch Keypoints", keypoints)
        return keypoints, onnx_keypoints
            
    def postprocess(self, keypoints):
        """
        Interpolates Pelvis, Thorax, Spine, and Head and sets skeleton to our Standard order. 
        This process is necessary to get a functional skeleton (by default, the skeleton is COCO format)
        """
        start = time.time()
        
        x = [keypoints[0]] + [keypoints[3]] + [keypoints[6]] + [keypoints[9]] + [keypoints[12]] + [keypoints[15]] + [keypoints[18]] + [keypoints[21]] + [keypoints[24]] + [keypoints[27]] + [keypoints[30]] + [keypoints[33]] + [keypoints[36]] + [keypoints[39]] + [keypoints[42]] + [keypoints[45]] + [keypoints[48]]
        y = [keypoints[1]] + [keypoints[4]] + [keypoints[7]] + [keypoints[10]] + [keypoints[13]] + [keypoints[16]] + [keypoints[19]] + [keypoints[22]] + [keypoints[25]] + [keypoints[28]] + [keypoints[31]] + [keypoints[34]] + [keypoints[37]] + [keypoints[40]] + [keypoints[43]] + [keypoints[46]] + [keypoints[49]]
        skeleton = [x, y]
        skeleton = np.array(skeleton).T
        
        #Reordering the joints
        skeleton = skeleton[[1, 12, 14, 16, 11, 13, 15, 6, 8, 10, 5, 7, 9, 2, 0, 3, 4], :]

        #Interpolate the Pelvis (mid point of left and right hips)
        vector = (skeleton[1]-skeleton[4])/2
        pelvis = skeleton[4] + vector
        skeleton[0,:] = pelvis

    #     #Interpolate the Thorax (mid point of left and right shoulders)

        #Interpolate Thorax (centered between left and right shoulders and shifted up 30%)
        vector_hz = (skeleton[10]-skeleton[7])/2
        vector_hz_x = vector_hz[0]
        vector_hz_y = vector_hz[1]
        midpoint = skeleton[10] + vector_hz
        abdomen_vector = midpoint - pelvis
        x_sign = abdomen_vector[0]
        y_sign = abdomen_vector[1]
        vector_vt = np.array([math.copysign(vector_hz[1], x_sign), math.copysign(vector_hz[0], y_sign)]) * 0.6
        thorax = skeleton[7] + vector_hz + vector_vt
        skeleton[15,:] = thorax

        #Interpolate the Spine (mid point of thorax and pelvis)
        vector = (skeleton[15]-skeleton[0])/2
        spine = skeleton[0] + vector
        skeleton[16,:] = spine

        #Interpolate the Head
        vector = (skeleton[14]-skeleton[15])
        head = skeleton[14] + vector
        skeleton[13,:] = head
    
        logging.info("Joint interpolation and reording complete. Time taken was: {:.3f}".format((time.time()-start)))

        if self.PLOT_TEST:
            plot_skeleton_2d(skeleton, output_filename="data/test_outputs/2d_skel")
    
        return skeleton
    
    def main(self, crop):
        """
        Returns a 2D skeleton (17x2 np array)
        """
        try:          
            coco_keypoints, onnx_keypoint = self.detect(crop)
            print("Done1")
            skel_2d = self.postprocess(coco_keypoints)
            skel_2d_onnx = self.postprocess(onnx_keypoint)

            print("cpn", skel_2d)
            print("onnx_cpn", skel_2d_onnx)
            img = show_image(crop, skel_2d_onnx)
            print("Done2")
#             for string in gpuUsage():
#                 logging.info(string) 
        except Exception as e:
            logging.exception("Exception in CPN main")
            print("Error! Exception in CPN main")
            print(e)
            #If skeletizing fails for whatever reason, None will be returned.
            skel_2d = None
        return skel_2d, img

skeleton2D = cpn()