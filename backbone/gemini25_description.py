import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 加载环境变量
load_dotenv()

class Gemini25_Description:
    def __init__(self, model_name_or_path="google/gemini-2.5-flash", max_tokens=30000):
        # 确保使用 DashScope 的 API 密钥和基础 URL
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1" # 用openrouter调用
        # base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name_or_path
        self.max_tokens = max_tokens

    def get_completion(self, system_prompt, user_prompt):
        
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
                        "content": user_prompt
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
    video_description = "The first part of the video says 'Movie Plot', in which the man suddenly turns into a corner and hides, and the people chasing him don't notice him and just run away. The second part says 'Reality', in which the man still turns into a corner, but the people chasing him immediately turn around and see him and catch him."
    system_prompt = "You are a helpful AI assistant specialized in video understanding and humor analysis. You can explain jokes clearly and naturally based on video content and video description."
    user_prompt = (
                 "You'll be given a description of the video. " 
                 "Your job is to explain why the video is humorous in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. "
                 "Respond with a 2-3 sentence explanation of the joke and how it relates to the video.\n\n"
                 "Output format:\n"
                 "Explanation: <answer>\n\n"
                 f"Video Description: {video_description}"
             )
    backbone = Gemini25_Description()
    response = backbone.get_completion(system_prompt, user_prompt)
    print(response)
    
    # A correct form of the example is: 
    # Explanation: The humor comes from the stark contrast between the movie plot and reality. In the movie, the protagonist cleverly hides from pursuers without being noticed, showcasing a common trope where the hero outsmarts the bad guys. However, in real life, the same scenario reveals that people are more likely to spot someone hiding right next to them, turning what should be a suspenseful moment into an anticlimactic one. This unexpected twist highlights how our expectations often differ from reality.
