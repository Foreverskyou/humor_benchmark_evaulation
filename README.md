# V-HUB: A VISUAL-CENTRIC HUMOR UNDERSTANDING BENCHMARK FOR VIDEO LLMS

![VideoQA](https://img.shields.io/badge/Task-VideoQA-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![Video-MME](https://img.shields.io/badge/Dataset-V--HUB-blue)  
![Gemini](https://img.shields.io/badge/Model-Gemini-green) 
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green)

<p align="center">
    <img src="./figures/teaser.png" width="100%" height="100%">
</p>

<font size=7><div align='center' > [[📖 arXiv Paper](https://arxiv.org/pdf/2509.25773)] [[📊 Dataset](https://huggingface.co/datasets/Foreverskyou/video/tree/main)] </div></font>

## 📐 Dataset Examples

<p align="center">
    <img src="./figures/example.png" width="100%" height="100%">
</p>

## 🔍 Dataset

**License**:
```
Video-MME is only used for academic research. Commercial use in any form is prohibited.
The copyright of all videos belongs to the video owners.
If there is any infringement in Video-MME, please email videomme2024@gmail.com and we will remove it immediately.
Without prior approval, you cannot distribute, publish, copy, disseminate, or modify Video-MME in whole or in part. 
You must strictly comply with the above restrictions.
```

Please send an email to **videomme2024@gmail.com**. 🌟

## 🔮 Data Curation and Evaluation Pipeline

<p align="center">
    <img src="./figures/pipline.png" width="100%" height="100%">
</p>

📍 **Filtering**

部署whisper模型，选择videos with less than 10 characters.

📍 **Annotation**

我们的标注平台在Label Studio，搭建平台请参考[Annotation_Manual](https://github.com/Foreverskyou/humor_benchmark_evaulation/tree/main/Annotation_Manual)和[Label Studio](https://github.com/HumanSignal/label-studio)

📍 **Evaluation**: 

Here we provide an example template [output_test_template.json](./evaluation/output_test_template.json). Once you have prepared the model responses in this format, please refer to the evaluation script [eval_your_results.py](https://github.com/thanku-all/parse_answer/blob/main/eval_your_results.py), and you will get the accuracy scores across video_durations, video domains, video subcategories, and task types. 
The evaluation does not introduce any third-party models, such as ChatGPT.

```bash
./scripts/Text_Only/example_QA.sh
```
You can specify multiple types separated by commas or organize them in a list, for example: `short,medium,long` or `["short","medium","long"]`.


测试三种不同的的setting: Text-Only/Video-Only/Video+Audio，分为QA/explanation/matching，测试脚本可参考scripts，其中

MODEL_NAME=`['Qwen2.5-Omni','Qwen2.5-VL','Gemini2.5-flash','GPT-4o','InterVL 3.5','Minicpm 2.6-o','video SALMONN 2']`

## :black_nib: Citation

If you find our work helpful for your research, please consider citing our work. 

```bibtex
@article{shi2025v,
  title={V-HUB: A Visual-Centric Humor Understanding Benchmark for Video LLMs},
  author={Shi, Zhengpeng and Li, Hengli and Zhao, Yanpeng and Zhou, Jianqun and Wang, Yuxuan and Cui, Qinrong and Bi, Wei and Zhu, Songchun and Zhao, Bo and Zheng, Zilong},
  journal={arXiv preprint arXiv:2509.25773},
  year={2025}
}
```
