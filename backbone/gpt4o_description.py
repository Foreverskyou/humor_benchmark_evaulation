import os
import sys
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from backbone_utils import extract_frames_base64, get_max_frame_and_interval

# 加载环境变量
load_dotenv()

class GPT4o_Description:
    def __init__(self, model_name_or_path="gpt-4o", max_tokens=120000):
        # 从环境变量读取 OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model_name_or_path
        self.max_tokens = max_tokens

    def get_completion(self, system_prompt, user_prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    video_description = "The first part of the video says 'Movie Plot', in which the man suddenly turns into a corner and hides, and the people chasing him don't notice him and just run away. The second part says 'Reality', in which the man still turns into a corner, but the people chasing him immediately turn around and see him and catch him."
    system_prompt = "You are a helpful AI assistant specialized in video understanding and humor analysis. You can explain jokes clearly and naturally based on video content and video description."
    user_prompt = (
           f"These are frames from a video. "
                 "And you'll be given a description of the video. " 
                 "Your job is to explain why the video is humorous in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. "
                 "Respond with a 2-3 sentence explanation of the joke and how it relates to the video.\n\n"
                 "Output format:\n"
                 "Explanation: <answer>\n\n"
                 f"Video Description: {video_description}"
             )
    backbone = GPT4o_Description()
    response = backbone.get_completion(system_prompt, user_prompt)
    print(response)
