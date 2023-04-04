import logging
import time
import datetime
import argparse
import cv2
import sys
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import glob
import os
from pipeline_utils import test_plot, plot_skeleton_3d
from pathlib import Path
import torch.onnx
import torch.nn as nn
# from pipeline_utils import gpuUsage

import EvoCascade

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

sys.path.insert(0, 'src/EvoSkeleton')
import libs.model.model as libm
from libs.dataset.h36m.data_utils import unNormalizeData
from examples.inference_func import normalize, get_pred

class EvoSkeleton:
    def __init__(self):
        """
        All evo processes that only need to occur once
        """  
        self.PLOT_TEST = False
        
        start = time.time()
        # Define constants
        self.re_order_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

        # Load model checkpoint
        model_path = 'src/EvoSkeleton/examples/example_model.th'
        ckpt = torch.load(model_path)
        
        # Load statistics used for normalizing & un-normalizing
        self.stats = np.load('src/EvoSkeleton/examples/stats.npy', allow_pickle=True).item()
        self.dim_used_2d = self.stats['dim_use_2d']
        self.mean_2d = self.stats['mean_2d']
        self.std_2d = self.stats['std_2d'] 
        
        # Initialize the model
        self.cascade = libm.get_cascade()
        input_size = 32
        output_size = 48
        for stage_id in range(2):
            # initialize a single deep learner
            stage_model = libm.get_model(stage_id + 1,
                                         refine_3d=False,
                                         norm_twoD=False, 
                                         num_blocks=2,
                                         input_size=input_size,
                                         output_size=output_size,
                                         linear_size=1024,
                                         dropout=0.5,
                                         leaky=False)
            self.cascade.append(stage_model)
        self.cascade.load_state_dict(ckpt)
        self.new_cascade = EvoCascade.CascadeModule(cascade=self.cascade)
        torch.save(self.new_cascade.state_dict(), 'src/EvoSkeleton/examples/new_cascade.th')

        ckpt = torch.load('src/EvoSkeleton/examples/new_cascade.th')
        print(ckpt)
        self.new_cascade.load_state_dict(ckpt)
        self.new_cascade.eval()
        print(self.new_cascade)
        logging.info("EvoSkeleton init time: {:.3f}".format((time.time()-start)))
    
    def preprocess(self, skeleton):
        """
        Joints are passed in standard order. They must be re-ordered to the format that EvoSkeleton expects (Human 3.6M format)
        """
        start = time.time()
        skeleton = skeleton[[0, 1, 2, 3, 4, 5, 6, 16, 15, 14, 13, 10, 11, 12, 7, 8, 9], :]
        logging.info("Preprocessing completed. Time taken was: {:.3f}".format((time.time()-start)))
        
        return skeleton
            
    def detect(self, skeleton):
        """
        All evo processes that occur once for every frame
        """

        #######################################################################

        def to_numpy(tensor):
            return tensor.cpu().detach().numpy() 

        def export(data2):
            import onnx
            import onnxruntime
            """
            Converts from EvoSkeleton output to standard joint order & swaps dimension order from (x,z,y) to (x,y,z)
            """
            self.new_cascade.eval()
            print("Printing Evo Input shape")
            print(data2.shape)
            ct_data = data2
            ct_data = ct_data.astype(np.float64)
            ct_data = torch.from_numpy(ct_data)
            # x = torch.randn(1,17,2, requires_grad=True, device = 'cpu')
            # print(x)
            data = torch.from_numpy(data2)
            print("Converted torch data")
            print(data)
            example_input = torch.randn(1, 32)
            print(example_input)
            torch_out = self.new_cascade(data)
            torch_out = torch_out.cpu().detach().numpy() 
            

            # # Export the model
            print(torch_out)
            torch.onnx.export(self.new_cascade,               # model being run
                            data,                         # model input (or a tuple for multiple inputs)
                            "src/EvoSkeleton/my_evo_skeleton2.onnx",   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=11,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'])
            
            print("put in model done evo")

            onnx_model = onnx.load("src/EvoSkeleton/my_evo_skeleton2.onnx")
            onnx.checker.check_model(onnx_model)

            

            ort_session = onnxruntime.InferenceSession("src/EvoSkeleton/my_evo_skeleton2.onnx")

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: data2}
            ort_outs = ort_session.run(None, ort_inputs)
            # print("Onnx output", ort_outs[0])
            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-02, atol=1e-03)

            print("Exported model has been tested with ONNXRuntime, and the result looks good!")

            import coremltools as ct

            traced_model = torch.jit.trace(self.new_cascade, data)
            out = traced_model(data)

            print("Tracer Done")
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=data.shape)]
            )

            model.save("src/EvoSkeleton/my_evo_skeleton2.mlmodel")


            print("CoreML Conversion Done")

            model = ct.models.MLModel('src/EvoSkeleton/my_evo_skeleton2.mlmodel')

            # Print input description to get input shape.
            print("Evo CoreML")
            print(model.get_spec().description.input)

            output_dict = model.predict({'input.1': data2})
            output_list = np.array([val for key,val in output_dict.items()])
            output_list = output_list[0]

            print(output_list.shape)
            print(torch_out.shape)

            np.testing.assert_allclose(torch_out, output_list, rtol=1e-01, atol=1e-02)
            print("Exported CoreML model has been tested with ONNXRuntime, and the result looks good!")

        #######################################################################
        start = time.time()
        
        # Use the author's normalize function - they remove the Neck/Nose joint (reduce down to a 16 keypoint skeleton) 
       
        norm_ske_gt = normalize(skeleton, self.re_order_indices).reshape(1,-1)
        export(norm_ske_gt.astype(np.float32))
        pred = get_pred(self.cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))

        # Unnormalizes into H36M style - done in original code
        pred = unNormalizeData(pred.data.numpy(),
                           self.stats['mean_3d'],
                           self.stats['std_3d'],
                           self.stats['dim_ignore_3d']
                           )
        
        # This is all post-processing that they do
        pred = pred.reshape(-1,3)
#         permute the order of x,y,z axis  
        pred = pred.reshape(96)
        vals = np.reshape( pred, (32, -1) )
        
        # Select the correct keypoints - maintains same 2D keypoint input the model expects (Human 3.6)
        points = vals[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :]
        logging.info("Detection completed. Time taken was: {:.3f}".format((time.time()-start)))
        return points
            
    def postprocess(self, pred):
        """
        Converts from EvoSkeleton output to standard joint order & swaps dimension order from (x,z,y) to (x,y,z)
        """
        start = time.time()
        out = pred[(0, 4, 5, 6, 1, 2, 3, 14, 15, 16, 11, 12, 13, 10, 9, 8, 7), :]
        logging.info("Post-processing completed. Time taken was: {:.3f}".format((time.time()-start)))
        
        if self.PLOT_TEST:
            plot_skeleton_3d(out, "test_outputs/skel_3d.png")
            
        return out


    def main(self, skel_2d):
        """
        Returns a 3D skeleton (17x3 np array) in our Standard Order
        """
        try:          
            skel_2d = self.preprocess(skel_2d)
            skel_3d = self.detect(skel_2d)
            skel_3d = self.postprocess(skel_3d)
            
        except:
            logging.exception("Exception in EvoSkeleton main")
            print("Error! Exception in EvoSkeleton main")
            skel_3d = None
        return skel_3d    
