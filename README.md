# <div align="center"> <i>PILOT</i>: Coherent and Multi-modality Image Inpainting via Latent Space Optimization </div>

<div align="center">

  <a href="https://pilot-page.github.io"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=GitHub&color=blue&logo=github"></a> &ensp;
  <a href="https://arxiv.org/abs/2407.08019"><img src="https://img.shields.io/static/v1?label=ArXiv&message=2407.08019&color=B31B1B&logo=arxiv"></a> &ensp;

</div>

---


Official Implement of PILOT.

[Lingzhi Pan](https://github.com/Lingzhi-Pan), [Tong Zhang](https://people.epfl.ch/tong.zhang?lang=en), [Bingyuan Chen](https://github.com/Alex-Lord), [Qi Zhou](https://github.com/zaqai), [Wei Ke](https://gr.xjtu.edu.cn/en/web/wei.ke), [Sabine Susstrunk](https://people.epfl.ch/sabine.susstrunk), [Mathieu Salzmann](https://people.epfl.ch/mathieu.salzmann)

<!--
**Code will be coming in two weeks! （before 7.28）**

Abstract: With the advancements in denoising diffusion probabilistic models (DDPMs), image inpainting has undergone a significant evolution, transitioning from filling information based on nearby regions to generating content conditioned on various factors such as text, exemplar images, sketches, etc. However, existing methods often necessitate fine-tuning of the model or concatenation of latent vectors, leading to drawbacks such as generation failure due to overfitting and inconsistent foreground generation. In this paper, we argue that the current large models are powerful enough to generate realistic images without further tuning. Hence, we introduce PILOT (in**P**ainting v**I**a **L**atent **O**p**T**imization), an optimization approach grounded on a novel semantic centralization and background loss to identify latent spaces capable of generating inpainted regions that exhibit high fidelity to user-provided prompts while maintaining coherence with the background region. Crucially, our method seamlessly integrates with any pre-trained model, including ControlNet and DreamBooth, making it suitable for deployment in multi-modal editing tools. Our qualitative and quantitative evaluations demonstrate that our method outperforms existing approaches by generating more coherent, diverse, and faithful inpainted regions to the provided prompts. Our project webpage: https://pilot-page.github.io.
-->


![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/teaser.jpg)


## Method Overview

![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/framework_a.png)
![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/framework_b.png)




## Getting Started
It is recommended to create and use a Torch virtual environment, such as conda. Next, download the appropriate PyTorch version compatible with your CUDA devices, and install the required packages listed in requirements.txt.
```
git clone https://github.com/Lingzhi-Pan/PILOT.git
cd PILOT
conda create -n pilot python==3.9
conda activate pilot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
You can download the `stable-diffusion-v1-5` model from the website "https://huggingface.co/runwayml/stable-diffusion-v1-5" and save it to your local path.

## Run
We provide three types of conditions to guide the inpainting process: text, spatial controls, and reference images. Each condition-control refers to a different configuration in the directory `configs/`.

### Text
```
python run_example.py --config_file configs/t2i_step50.yaml
```
![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/text_add.png)

### Text + Spatial Controls
```
python run_example.py --config_file configs/controlnet_step30.yaml
```
![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/controlNet_results.png)

### Text + Reference Image
```
python run_example.py --config_file configs/ipa_step30.yaml
```
![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/ip_adapter_a.png)

### Text + Spatial Controls + Reference Image
```
python run_example.py --config_file configs/ipa_controlnet_step30.yaml
```
![image](https://github.com/Lingzhi-Pan/PILOT/blob/main/assets/ip_adapter_b.png)
**See our [GitHub pages](https://pilot-page.github.io) for more information!**


## BibTeX
If you find this work helpful, please consider citing:
```bibtex
@article{pan2024coherent,
  title={Coherent and Multi-modality Image Inpainting via Latent Space Optimization},
  author={Pan, Lingzhi and Zhang, Tong and Chen, Bingyuan and Zhou, Qi and Ke, Wei and S{\"u}sstrunk, Sabine and Salzmann, Mathieu},
  journal={arXiv preprint arXiv:2407.08019},
  year={2024}
}
```



