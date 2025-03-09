import gradio
import logging
import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video


if __name__ == '__main__':
    print('getting t2v-1.3B config')
    t2v_1_3B = WAN_CONFIGS['t2v-1.3B']
    ckpt_dirs = {
        'i2v_14b_480p': '../Wan2.1-I2V-14B-480P',
        'i2v_1_3b_480p': '../Wan2.1-I2V-14B-200P',
        't2v_1_3b': '../Wan2.1-T2V-1.3B'
    }
    resolution = {'x':832, 'y':480}
    frame_num = 33
    sampling_steps = 25
    prompt = 'a penguin walking in the snow'
    neg_prompt = ('overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, '
                  'overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, '
                  'poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, '
                  'still picture, cluttered background, three legs, many people in the background, walking backwards')
    print('instantiating WanT2V')
    device = torch.device('mps')
    print(device)
    wan_t2v = wan.WanT2V(
        config=t2v_1_3B,
        checkpoint_dir=ckpt_dirs['t2v_1_3b'],
        device_id=device,
        rank=0,  # Single device, so use rank 0
        t5_fsdp=False,  # Disable FSDP (not supported on MPS)
        dit_fsdp=False,  # Disable FSDP (not supported on MPS)
        use_usp=False,  # Disable Ulysses/ring parallelism (single device)
        t5_cpu=True,
    )

    print('generating video')
    video = wan_t2v.generate(
        input_prompt=prompt,
        size=(
            resolution['x'],
            resolution['y']
        ),
        frame_num=frame_num,
        shift=None,
        sample_solver='unipc',
        sampling_steps=sampling_steps,
        guide_scale=5.0,
        seed=1234,
        n_prompt=neg_prompt,
        offload_model=True
    )
    print('saving video')
    cache_video(
        tensor=video[None],
        save_file="example.mp4",
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))

