import os, sys
from cog import BasePredictor, Input
from cog import Path as CogPath

sys.path.append('/content/StreamingT2V')
sys.path.append('/content/StreamingT2V/t2v_enhanced')
sys.path.append('/content/StreamingT2V/t2v_enhanced/thirdparty')
os.chdir('/content/StreamingT2V')

from typing import List
from pathlib import Path
from os.path import join as opj
import torch
from inference_utils import *
from model_init import *
from model_func import *

class Options:
    def __init__(self):
        self.prompt = "A cat running on the street"
        self.image = ""
        self.base_model = "ModelscopeT2V"
        self.num_frames = 24
        self.negative_prompt = ""
        self.negative_prompt_enhancer = None
        self.num_steps = 50
        self.image_guidance = 9.0
        self.output_dir = "results"
        self.device = "cuda"
        self.seed = 33
        self.chunk = 24
        self.overlap = 8
args = Options()

class Predictor(BasePredictor):
    def setup(self) -> None:
        directory = "/content/StreamingT2V/results"
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.result_fol = Path("/content/StreamingT2V/results")
        self.device = args.device
        self.ckpt_file_streaming_t2v = Path("/content/StreamingT2V/t2v_enhanced/checkpoints/streaming_t2v.ckpt")
        self.cfg_v2v = {'downscale': 1, 'upscale_size': (1024, 1024), 'model_id': '/content/Video-to-Video', 'pad': True}
        self.stream_cli, self.stream_model = init_streamingt2v_model(self.ckpt_file_streaming_t2v, self.result_fol)
        if args.base_model == "ModelscopeT2V":
            self.model = init_modelscope(self.device)
        elif args.base_model == "AnimateDiff":
            self.model = init_animatediff(self.device)
        elif args.base_model == "SVD":
            self.model = init_svd(self.device)
            self.sdxl_model = init_sdxl(self.device)
        self.msxl_model = init_v2v_model(self.cfg_v2v)
    def predict(
        self,
        prompt: str = Input(default="A cat running on the street"),
        negative_prompt: str = Input(default=""),
        num_frames: int = Input(default=24),
        seed: int = Input(default=33),
        num_steps: int = Input(default=50),
        image_guidance: int = Input(default=9),
        chunk: int = Input(default=24),
        overlap: int = Input(default=8),
        enhance: bool = False,
    ) -> List[CogPath]:
        args.prompt = prompt
        args.negative_prompt = negative_prompt
        args.num_frames = num_frames
        args.seed = seed
        args.num_steps = num_steps
        args.image_guidance = image_guidance
        args.chunk = chunk
        args.overlap = overlap
        name = 'output'
        inference_generator = torch.Generator(device="cuda")
        inference_generator.manual_seed(args.seed)
        if args.base_model == "ModelscopeT2V":
            short_video = ms_short_gen(args.prompt, self.model, inference_generator)
        elif args.base_model == "AnimateDiff":
            short_video = ad_short_gen(args.prompt, self.model, inference_generator)
        elif args.base_model == "SVD":
            short_video = svd_short_gen(args.image, args.prompt, self.model, self.sdxl_model, inference_generator)
        n_autoreg_gen = (args.num_frames-8)//8
        stream_long_gen(args.prompt, short_video, n_autoreg_gen, args.negative_prompt, args.seed, args.num_steps, args.image_guidance, name, self.stream_cli, self.stream_model)
        args.negative_prompt_enhancer = args.negative_prompt_enhancer if args.negative_prompt_enhancer is not None else args.negative_prompt
        if enhance:
            video2video_randomized(args.prompt, opj(self.result_fol, name+".mp4"), self.result_fol, self.cfg_v2v, self.msxl_model, chunk_size=args.chunk, overlap_size=args.overlap, negative_prompt=args.negative_prompt_enhancer)
            return [CogPath('/content/StreamingT2V/results/output.mp4'), CogPath('/content/StreamingT2V/results/output_enhanced.mp4')]
        else:
            return [CogPath('/content/StreamingT2V/results/output.mp4')]
