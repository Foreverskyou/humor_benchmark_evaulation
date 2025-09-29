import os
import sys
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from backbone_utils import extract_frames_base64, get_max_frame_and_interval,reconstruct_video,encode_video
from moviepy import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# 加载环境变量
load_dotenv()

class Qwen25_Omni_Sound:
    def __init__(self, model_name_or_path="qwen2.5-omni-7b", max_tokens=30000):
        # 确保使用 DashScope 的 API 密钥和基础 URL
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name_or_path
        self.max_tokens = max_tokens

        # model patch height
        self.model_h = 28

        # model patch width
        self.model_w = 28

        self.model_max_tokens = 1280
        self.model_min_tokens = 4

    def get_completion(self, system_prompt, user_prompt, video_path):
        """
        从 DashScope API (Qwen-omni) 生成文本补全
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")
        
        nframes, interval = get_max_frame_and_interval(
            self.max_tokens, cap, frame, self.model_h, self.model_w, self.model_max_tokens, self.model_min_tokens
        )

        if original_fps <= 0:
            original_fps = 30.0  # 默认帧率

        time_between_frames_sec = interval / original_fps
        target_fps = 1.0 / time_between_frames_sec if time_between_frames_sec > 0 else 1.0

        cap.release()

        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        clip = VideoFileClip(video_path)
        audio = clip.audio  # 获取音频对象
        audio_path = None
        if audio is not None:
            audio_path = "extracted_audio.wav"
            audio.write_audiofile(audio_path)

        reconstruct_video(base64_frames, audio_path, target_fps, "reconstruct_video.mp4")

        base64_video = encode_video("reconstruct_video.mp4")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", 
                             "video_url":{
                                 "url": f"data:;base64,{base64_video}"}},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            
            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return full_response if full_response else "No response received from API"

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    video_description = "A man in skates fled, and two security guards couldn't catch him. Then many more people appeared in skates, creating a scene of chaos. The man ran out the door and used a cane to hook a car on the road, while others rushed out trying to chase him but ultimately couldn't catch up and fell down."
    system_prompt = "You are a helpful AI assistant specialized in video understanding and humor analysis. You can explain jokes clearly and naturally based on video content and video description."
    user_prompt = (
                f"These are frames from a video. "
                 "Your job is to explain why the video is humorous in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. "
                 "Respond with a 2-3 sentence explanation of the joke and how it relates to the video.\n\n"
                 "Output format:\n"
                 "Explanation: <answer>\n\n"
             )
    backbone = Qwen25_Omni_Sound()
    response = backbone.get_completion(system_prompt, user_prompt, video_path="1146_clip_17.mp4")
    print(response)
    
    # A correct form of the example is: 
    # Explanation: The humor comes from the stark contrast between the movie plot and reality. In the movie, the protagonist cleverly hides from pursuers without being noticed, showcasing a common trope where the hero outsmarts the bad guys. However, in real life, the same scenario reveals that people are more likely to spot someone hiding right next to them, turning what should be a suspenseful moment into an anticlimactic one. This unexpected twist highlights how our expectations often differ from reality.

# 输入数据csv：data1.csv
# 输入video：humor_benchmark/video
# 不同任务的prompt：
#             system_prompt (str)
#             user_prompt (str)
#         if task_type == "QA":
#             question = instruction.get("question")
#             video_description = instruction.get("video_description")
#             humor_explanation = instruction.get("humor_explanation")
#             system_prompt = "You are a helpful AI assistant. You can analyze videos and answer questions about their content."
#             user_prompt = (
#                 f"These are frames from a video. You will also be given a description of the video. "
#                 f"Based on this information, answer the following question: {question}\n\n"
#                 "Output format:\n"
#                 "Answer: <answer>\n\n"
#                 f"Video Description: {video_description}\n"
#                 f"Humor Explanation: {humor_explanation}\n"
#             )
        
#         elif task_type == "explanation":
#             question = instruction.get("question")
#             video_description = instruction.get("video_description")
#             system_prompt = "You are a helpful AI assistant specialized in video understanding and humor analysis. You can explain jokes clearly and naturally based on video content and video description."
#             user_prompt = (
#                 f"These are frames from a video. "
#                 "And you'll be given a description of the video. " 
#                 "Your job is to explain why the video is humorous in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. "
#                 "Respond with a 2-3 sentence explanation of the joke and how it relates to the video.\n\n"
#                 "Output format:\n"
#                 "Explanation: <answer>\n\n"
#                 f"Video Description: {video_description}"
#             )

#         elif task_type == "matching":
#             question = instruction.get("question")
#             system_prompt = "You are a helpful AI assistant. You can analyze videos and answer questions about their content."
#             user_prompt = (
#                 f"Here's the video URL: {video_path}. {question}"
#             )

# QA输出结果：In the ""Movie Plot"" part of the video, the man suddenly turns into a corner and hides, and the people chasing him don't notice him and just run away."
# explanation输出结果：The humor comes from the stark contrast between the movie plot and reality. In the movie, the protagonist cleverly avoids detection by turning into a corner, leading his pursuers to miss him entirely. However, in real life, the same scenario results in immediate discovery because people are more alert and aware of their surroundings, making the situation less amusing than expected.
# matching输出结果：B
