# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")

        prompts = [
            "In a medium wide shot, during daytime with soft sunlight filtering through a window, a white, rectangular box is placed on a reflective surface against a light blue background. The box is initially empty. As the video progresses, small, multicolored spheres begin to appear inside the box, gradually increasing in number until the box is filled with a dense network of spheres connected by thin white wires. The spheres, in various colors such as shades of pink, purple, blue, green, and yellow, float gently within the box. The wires connecting the spheres form a complex web, and the spheres are evenly distributed throughout the box. Sunlight creates a warm, gentle glow, highlighting the intricate details of the spheres and wires.",
            "Day time, top lighting, wide shot, warm colors. A person dressed in a dark blue suit with a white shirt and a pink pocket square walks confidently on a rooftop. The sky is a clear, bright blue with wisps of clouds drifting by. The individual, with their hair styled in a bun, stands out against the grayish-brown surface of the rooftop. Buildings stretch into the distance behind them, adding depth to the scene. The person's shadow is cast onto the rooftop as they walk, creating a dynamic interplay of light and shadow.",
            "Day time, sunny lighting, cool colors, wide shot, center composition. The video begins with a view of a vibrant coral reef underwater, with a diver in the foreground. The diver, equipped with scuba gear including a yellow buoyancy control device (BCD) and a yellow mask, swims gracefully through the reef. Various coral formations and colorful marine life surround the diver. Clear water allows sunlight to filter through, casting a serene blue hue on the scene. As the diver moves, a sleek gray shark with a white underbelly appears, swimming close to the reef and the diver. The diver maintains a safe distance from the shark, continuing to explore the reef. The interaction between the diver and the shark highlights the beauty and tranquility of the underwater world. Gentle currents cause coral and seaweed to sway softly in the background.",
            "Day time, sunny lighting, soft lighting, cool colors, wide shot, center composition. The video captures a serene coastal landscape with a clear blue sky. In the foreground, a red and white windsock flutters in the wind, attached to a metal pole on the right side of the frame. In the background, a person paraglides with a white parachute visible against the blue sky, moving from left to right across the frame. The sea is visible in the far background, with a mountainous coastline. The overall color palette is dominated by blues, greens, and whites, with gentle waves lapping at the shore and clouds drifting lazily across the sky.",
            "Early morning, daylight, side lighting, wide shot, balanced composition. A person skates on a concrete sidewalk in a residential area, wearing a plaid shirt, jeans, and rollerblades. They skate towards a set of stairs, executing a jump over the railing, and land gracefully on the opposite side. The environment includes houses with neatly trimmed lawns and trees casting early morning shadows. The sky is a bright, clear blue. Gentle sunlight casts a soft glow on the scene, highlighting the individual's athletic form as they perform the jump. Leaves on the trees rustle gently in the breeze.",
            "Side lighting, soft lighting, warm colors, medium shot, daylight. The video captures a serene moment of a baby lying on a white, fluffy blanket. The baby is dressed in a white onesie and appears to be in a state of deep sleep. Natural light filters in from a large window to the side, casting a soft glow on the baby's face. The baby's hands are gently resting on the blanket, and the overall atmosphere is calm and peaceful. In the background, a wooden rocking chair and a few toys can be seen, adding warmth to the scene.",
            "First-person perspective, day time, sunlight, medium shot, balanced composition. A person wearing a white helmet, blue jacket, and pink gloves stands on a grassy field, adjusting a yellow paragliding wing spread out on the ground. The sky is a clear blue with a few white clouds drifting by. Houses, trees, and distant mountains provide a scenic backdrop. The individual bends down to check the wing's tension, then straightens up and takes a final look at the surroundings before preparing for takeoff. The sunlight casts gentle shadows on the grass and creates a warm, inviting atmosphere.",
            "Night time, artificial lighting, soft lighting, cool colors, medium wide shot, center composition. The video opens with a darkened street, illuminated by the fire truck's lights and surrounding street lamps. Firefighters, dressed in full gear, walk towards the truck, their helmets and flashlights casting shadows on the pavement. The environment is dimly lit, with the fire truck parked on the side of the road, its visible markings and equipment clearly shown. As the video progresses, the firefighters move around the truck, preparing for an operation. Their actions are methodical and coordinated, indicating a professional response to an emergency situation. The ambient light creates a cool, calm atmosphere, highlighting the seriousness of the scenario.",
            "Day time, sunlight, soft lighting, warm colors, medium wide shot, balanced composition. A group of individuals gathers on the banks of a river, with some standing and others in the water. Those in the water wear bright orange life jackets, suggesting a safety measure. The riverbank is lined with small wooden boats, their sails gently fluttering in the breeze. Flags wave in the background, indicating a festive event. Individuals in the water splash and interact joyfully, celebrating together. The sun casts a gentle glow, creating a warm and inviting atmosphere. Trees along the riverbank sway gently, adding movement to the scene.",
            "Close-up shot, soft lighting, cool colors, artificial lighting. A woman with blonde hair wearing a light-colored top is captured in a close-up. She is in a room with a blue wall in the background. Her expression changes subtly, shifting from calm to intrigued as she reacts to something or someone off-camera. Artificial lights hanging from the ceiling provide soft illumination, casting gentle shadows on her face. The background remains static, emphasizing her subtle facial expressions."
        ]
        prompts = [
            "Late afternoon, golden hour, side lighting, warm colors, wide shot, center composition. The scene opens on a quiet countryside road lined with tall sunflowers swaying gently in the breeze. A vintage red bicycle rests against a wooden fence on the left side of the frame. The low sun casts long shadows across the dirt path, creating a warm, nostalgic glow. In the distance, rolling hills stretch beneath a soft orange sky. Dust particles glimmer in the sunlight, adding texture and depth to the peaceful scene.",
            "Day time, natural lighting, cool tones, wide shot, symmetrical composition. Inside a modern glass greenhouse, rows of lush green plants extend into the distance. A person wearing a white lab coat and gloves tends to the plants using a small watering can. Sunlight filters through the glass panels above, creating dynamic patterns of light and shadow on the floor. The humidity inside causes a light mist to hang in the air, giving the environment a fresh and serene atmosphere.",
            "Night time, neon lighting, strong contrast, cool colors, medium wide shot. A bustling city street glows with reflections of neon signs in puddles from recent rain. A woman in a red trench coat walks with an umbrella, her reflection trailing her in the wet pavement. Cars pass by, casting streaks of light. The camera tracks her movement as she crosses the street, surrounded by the hum of the city. Steam rises from a nearby manhole, enhancing the cinematic urban mood.",
            "Early morning, foggy lighting, muted colors, wide shot, minimal composition. The scene shows a lone rowboat drifting quietly on a mist-covered lake. The boatman, wearing a dark coat and hat, slowly paddles through the still water. The surface reflects the pale sky, almost blending with the fog, creating a dreamlike, monochromatic look. Occasional ripples from the oar break the reflection, adding gentle motion to the tranquil atmosphere.",
            "Sunset, warm lighting, medium wide shot, diagonal composition. A train glides along a seaside railway as the sun dips below the horizon. The orange and pink hues of the sunset reflect off the metal of the train and the calm ocean beside it. Seagulls fly overhead as waves gently crash against the rocky shore. The scene captures a fleeting moment of transition between day and night, filled with warmth and motion.",
            "Day time, soft sunlight, warm tones, medium shot, balanced composition. A young artist paints at an easel in a cozy studio filled with plants. Sunlight streams in from a nearby window, highlighting specks of dust in the air. Brushes, palettes, and unfinished canvases lie around the room, hinting at creative chaos. The artist dips their brush into a jar of water, creating ripples that catch the light, evoking a sense of focus and calm productivity.",
            "Night time, candlelight, warm colors, close-up shot. A person writes in a journal at a wooden desk illuminated only by the soft flicker of a candle. The pen moves slowly across the page as wax drips down the candleholder. The surrounding room fades into darkness, with faint outlines of books and a cup of tea visible. The warm glow emphasizes the intimacy and introspection of the moment.",
            "Afternoon, bright sunlight, high contrast, wide shot, dynamic composition. A skateboarder performs tricks at an urban skate park. Graffiti covers the concrete ramps and walls, bursting with color. The camera follows the skateboarder as they launch off a ramp, spinning mid-air before landing smoothly. The sunlight reflects off the metal edges of the railings, capturing the energy and rhythm of the movement.",
            "Dusk, cool and warm mix lighting, wide shot, cinematic tone. A rural gas station sits alone along an empty highway, its lights flickering on as twilight descends. A vintage car pulls up, and the driver steps out to refuel. The surrounding landscape is flat and open, with purple and orange hues blending in the horizon. Crickets chirp faintly in the background, reinforcing the stillness of early evening.",
            "Morning, indoor ambient light, soft lighting, medium shot, vertical composition. A barista prepares a cup of coffee in a small café. Steam rises from the espresso machine as milk is frothed and poured into a ceramic cup. Light enters through a nearby window, casting a golden hue on the wooden counter. The scene focuses on the subtle motions — the swirl of coffee, the shimmer of steam — capturing the quiet ritual of morning preparation."
        ]
        prompts = [
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
            "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
        ]

        for prompt in prompts:

            if rank == 0:
                logging.info(f"Generating for prompt: {prompt}")
            video = wan_ti2v.generate(
                prompt,
                img=img,
                size=SIZE_CONFIGS[args.size],
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)
            if rank == 0:
                logging.info(f"Generated video shape: {video.shape}")

            if rank == 0:
                
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.mp4'
                save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

                logging.info(f"Saving generated video to {save_file}")
                save_video(
                    tensor=video[None],
                    save_file=save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                if "s2v" in args.task:
                    if args.enable_tts is False:
                        merge_video_audio(video_path=save_file, audio_path=args.audio)
                    else:
                        merge_video_audio(video_path=save_file, audio_path="tts.wav")
            del video

    elif "animate" in args.task:
        logging.info("Creating Wan-Animate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )

        logging.info(f"Generating video ...")
        video = wan_animate.generate(
            src_root_path=args.src_root_path,
            replace_flag=args.replace_flag,
            refert_num = args.refert_num,
            clip_len=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
