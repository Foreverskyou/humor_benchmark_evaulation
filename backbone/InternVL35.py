import io
import os
import cv2
import base64
from typing import List

from PIL import Image

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

# 复用你已有的抽帧工具
from backbone_utils import (
    get_max_frame_and_interval,
    extract_frames_base64,
)

def b64_to_pil_list(b64_list: List[str]) -> List[Image.Image]:
    imgs = []
    for b in b64_list:
        data = base64.b64decode(b)
        imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return imgs


class InternVL35:
    """
    用 InternVL3_5-8B 做视频多帧理解（非流式）。
    继续使用你的抽帧预算策略与提帧实现。
    """
    def __init__(
        self,
        model: str = "OpenGVLab/InternVL3_5-8B",
        tp: int = 1,                 # 38B 用 tp=2；241B-A28B 用 tp=8（这里只是8B示例）
        session_len: int = 32768,    # 根据任务长度可调
    ):
        engine_cfg = PytorchEngineConfig(session_len=session_len, tp=tp)
        self.pipe = pipeline(model, backend_config=engine_cfg)


    #从代码里发现最大分块数为12，每个448*448为一块，占256个token
    def get_completion(
        self,
        video_path: str,
        user_prompt: str,
        # 视觉预算 & 抽帧策略（与你原脚本一致的接口）
        max_tokens: int = 31000,
        model_h: int = 448,
        model_w: int = 448,
        model_max_tokens: int = 3072,
        model_min_tokens: int = 256,
        image_prefix: str = "Frame",
    ) -> str:
        """
        返回：模型输出文本
        """
        # --- 1) 打开视频 & 估算抽帧策略 ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")

        # 估算最多帧数与抽帧间隔（你的逻辑）
        nframes, interval = get_max_frame_and_interval(
            max_tokens, cap, frame, model_h, model_w, model_max_tokens, model_min_tokens
        )

        # 真正抽帧（base64）
        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        cap.release()

        if not base64_frames:
            raise RuntimeError("未抽取到有效帧，请检查视频或抽帧参数。")

        # 2) 转 PIL，并用 lmdeploy 的 load_image 做标准化
        pil_frames = b64_to_pil_list(base64_frames)

        images = [load_image(img) for img in pil_frames]

        # 3) 构造提示词：为每张图插入编号 + IMAGE_TOKEN（官方多图范式）
        header_lines = [f"{image_prefix}-{i+1}: {IMAGE_TOKEN}" for i in range(len(images))]
        prompt = "\n".join(header_lines) + f"\n{user_prompt}"

        # 4) 一次性非流式推理
        #    这里无需 streaming_* 接口，直接 pipe((prompt, images)) 即可
        resp = self.pipe((prompt, images))

        # 5) 返回文本
        return getattr(resp, "text", str(resp))


if __name__ == "__main__":
    """
    示例运行：
    python internvl35_8b_video.py
    """
    video_path="humor_benchmark/video/1716681072748982272.mp4"

    video_description = "The first part of the video says 'Movie Plot', in which the man suddenly turns into a corner and hides, and the people chasing him don't notice him and just run away. The second part says 'Reality', in which the man still turns into a corner, but the people chasing him immediately turn around and see him and catch him."
    user_prompt = (
           f"These are frames from a video. "
                 "And you'll be given a description of the video. " 
                 "Your job is to explain why the video is humorous in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. "
                 "Respond with a 2-3 sentence explanation of the joke and how it relates to the video.\n\n"
                 "Output format:\n"
                 "Explanation: <answer>\n\n"
                 f"Video Description: {video_description}"
             )
    
    backbone = InternVL35(
        model="OpenGVLab/InternVL3_5-8B",
        tp=1,
        session_len=32768
    )

    response = backbone.get_completion(
        video_path=video_path,
        user_prompt=user_prompt,
        max_tokens=31000,   # 继续沿用你的预算策略
        model_h=448, model_w=448,
        model_max_tokens=3072, model_min_tokens=256,
        image_prefix="Frame"
    )

    print("=== MODEL OUTPUT ===")
    print(response)
