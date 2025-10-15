Data

我们的数据集下载地址：https://huggingface.co/datasets/Foreverskyou/video/tree/main

Filtering

部署whisper模型，选择videos with less than 10 characters.

Annotation System

我们的标注平台在Label Studio，搭建平台请参考Annotation_Manual和https://github.com/HumanSignal/label-studio

Evaluation

测试三种不同的的setting: Text-Only/Video-Only/Video+Audio，分为QA/explanation/matching，测试脚本可参考scripts，其中MODEL_NAME=['Qwen2.5-Omni','Qwen2.5-VL','Gemini2.5-flash','GPT-4o','InterVL 3.5','Minicpm 2.6-o','video SALMONN 2']
