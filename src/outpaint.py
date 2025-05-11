from dataclasses import dataclass
from typing import Optional
import argparse
import glob
import os
import sys

import numpy as np
import PIL
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.append("tools/Fooocus")
from fooocus_command import Fooocus


@dataclass
class Frame():
    '''
    rgb: in shape of H*W*3, in range of 0-1
    dpt: in shape of H*W, real depth
    inpaint: bool mask in shape of H*W for inpainting
    intrinsic: 3*3
    extrinsic: array in shape of 4*4

    As a class for:
    initialize camera
    accept rendering result
    accept inpainting result
    All at 2D-domain
    '''
    def __init__(self,
                 H: int = None,
                 W: int = None,
                 rgb: np.array = None,
                 dpt: np.array = None,
                 sky: np.array = None,
                 inpaint: np.array = None,
                 intrinsic: np.array = None,
                 extrinsic: np.array = None,
                 # detailed target
                 ideal_dpt: np.array = None,
                 ideal_nml: np.array = None,
                 prompt: str = None) -> None:
        self.H = H
        self.W = W
        self.rgb = rgb
        self.dpt = dpt
        self.sky = sky
        self.prompt = prompt
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self._rgb_rect()
        self._extr_rect()
        # for inpainting
        self.inpaint = inpaint
        self.inpaint_wo_edge = inpaint
        # for supervision
        self.ideal_dpt = ideal_dpt
        self.ideal_nml = ideal_nml

    def _rgb_rect(self):
        if self.rgb is not None:
            if isinstance(self.rgb, PIL.PngImagePlugin.PngImageFile):
                self.rgb = np.array(self.rgb)
            if isinstance(self.rgb, PIL.JpegImagePlugin.JpegImageFile):
                self.rgb = np.array(self.rgb)
            if np.amax(self.rgb) > 1.1:
                self.rgb = self.rgb / 255

    def _extr_rect(self):
        if self.extrinsic is None: self.extrinsic = np.eye(4)
        self.inv_extrinsic = np.linalg.inv(self.extrinsic)


class Llava():
    def __init__(self,device='cuda',
                 llava_ckpt='llava-hf/bakLlava-v1-hf') -> None:
        self.device = device
        self.model_id = llava_ckpt
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def __call__(self, image:PIL.Image, prompt: Optional[str] = None):

        # input check
        if not isinstance(image,PIL.Image.Image):
            if np.amax(image) < 1.1:
                image = image * 255
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)

        prompt = '<image>\n USER: Detaily imagine and describe the scene this image taken from? \n ASSISTANT: This image is taken from a scene of ' if prompt is None else prompt
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return answer


class Inpaint_Tool():
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self):
        self.fooocus = Fooocus()
        self.llava = Llava(device='cpu',llava_ckpt="llava-hf/bakLlava-v1-hf")

    def _llava_prompt(self):
        prompt = '<image>\n \
                USER: Detaily imagine and describe the scene this image taken from? \
                \n ASSISTANT: This image is taken from a scene of '
        return prompt

    def __call__(self, frame:Frame, outpaint_selections=[], outpaint_extend_times=0.0):
        '''
        Must be Frame type
        '''
        # conduct reconstuction
        # ----------------------- LLaVA -----------------------
        if frame.prompt is None:
            print('Inpaint-Caption[1/3] Move llava.model to GPU...')
            self.llava.model.to('cuda')
            print('Inpaint-Caption[2/3] Llava inpainting instruction:')
            query  = self._llava_prompt()
            prompt = self.llava(frame.rgb,query)
            split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') + len(f'ASSISTANT: This image is taken from a scene of ')
            prompt = prompt[split:]
            print(prompt)
            print('Inpaint-Caption[3/3] Move llava.model to CPU...')
            self.llava.model.to('cpu')
            torch.cuda.empty_cache()
            frame.prompt = prompt
        else:
            prompt = frame.prompt
            print(f'Using pre-generated prompt: {prompt}')
        # --------------------- Fooocus ----------------------
        print('Inpaint-Fooocus[1/2] Fooocus inpainting...')
        image = frame.rgb
        mask = np.zeros_like(image,bool) if len(outpaint_selections)>0 else frame.inpaint
        fooocus_result = self.fooocus(image_number=1,
                            prompt=prompt + ' 8K, perspective, natural, roomy, wide, good visibility, no large circles, no cameras, no fisheye.',
                            negative_prompt='fisheye, large circles, blurry, unrealistic, occluded, cluttered, many people.',
                            outpaint_selections=outpaint_selections,
                            outpaint_extend_times=outpaint_extend_times,
                            origin_image=image,
                            mask_image=mask,)[0]
        torch.cuda.empty_cache()

        # reset the frame for outpainting
        if len(outpaint_selections) > 0.:
            assert len(outpaint_selections) == 4
            small_H, small_W = frame.rgb.shape[0:2]
            large_H, large_W = fooocus_result.shape[0:2]
            if frame.intrinsic is not None:
                # NO CHANGE TO FOCAL
                frame.intrinsic[0,-1] = large_W//2
                frame.intrinsic[1,-1] = large_H//2
            # begin sample pixel
            frame.H = large_H
            frame.W = large_W
            begin_H = (large_H-small_H)//2
            begin_W = (large_W-small_W)//2
            inpaint = np.ones_like(fooocus_result[...,0])
            inpaint[begin_H:(begin_H+small_H),begin_W:(begin_W+small_W)] *= 0.
            frame.inpaint = inpaint > 0.5
        frame.rgb = fooocus_result

        print('Inpaint-Fooocus[2/2] Assign Frame...')
        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--outpaint_extend_times", type=float, default=0.4)
    args = parser.parse_args()

    rgb_inpaintor = Inpaint_Tool()

    image_paths = []
    if os.path.isdir(args.image_path):
        image_paths += glob.glob(os.path.join(args.image_path, "*"))
    else:
        image_paths.append(args.image_path)

    for imgpath in image_paths:
        rgb = PIL.Image.open(imgpath)
        rgb = np.array(rgb)[:,:,:3]
        # conduct outpainting on rgb and change cu,cv
        outpaint_frame = rgb_inpaintor(
            Frame(rgb=rgb),
            outpaint_selections=['Left','Right','Top','Bottom'],
            outpaint_extend_times=args.outpaint_extend_times)
        outpaint_rgb = np.clip(255 * outpaint_frame.rgb, 0, 255).astype(np.uint8)

        os.makedirs(args.save_dir, exist_ok=True)
        save_name = os.path.splitext(os.path.basename(imgpath))
        save_name = save_name[0] + "_outpaint" + save_name[1]
        save_path = os.path.join(args.save_dir, save_name)
        PIL.Image.fromarray(outpaint_rgb).save(save_path)
