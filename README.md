<div align="center">
  <img src="Logics-Parsing-Omni/imgs/logo.jpg" width="80%" >
</div>


<p align="center">
    🤗 <a href="https://huggingface.co/Logics-MLLM/Logics-Parsing-Omni">Model</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/studios/Alibaba-DT/logix/summary">Demo</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/">Technical Report</a>
</p>

## News
* 2026.01.15: We have released the [**Logics-Parsing-Omni**](https://huggingface.co/Logics-MLLM/Logics-Parsing-Omni). For more details, please check our [**Logics-Parsing-Omni paper**](https://arxiv.org/)
* 2025.09.17: We have released the [**Logics-Parsing**](https://huggingface.co/Logics-MLLM/Logics-Parsing). For more details, please check our [**Logics-Parsing paper**](https://arxiv.org/abs/2509.19760)

## Introduction
Logics-Parsing-Omni is a unified Multimodal Large Language Model (MLLM) designed to bridge the gap between pixel-level structural parsing and semantic-level cognitive captioning. It delivers breakthroughs in fine-grained perception and high-level cognition across documents, images, audio, and video.

<div align="center">
  <img src="Logics-Parsing-Omni/imgs/overview.png" alt="Logics-Parsing-Omni 概览" style="width: 800px; height: 250px;">
</div>


## Key Features

*   **Holistic Document Deconstruction**
    *   It enables holistic document deconstruction by surpassing traditional OCR to precisely restore layouts, convert unstructured charts into structured data, and interpret illustrations contextually.

*   **Enhanced Visual Captioning and Reasoning**
    *   It improves caption accuracy for both general images and complex graphics.
    *   It introduces a difference-perception mechanism that captures subtle visual changes for fine-grained image reasoning and executable editing suggestions.

*   **Advanced Audio–Video Understanding**
    *   It extends these capabilities to audio and video, significantly improving dynamic scene understanding through the joint modeling of video OCR, camera movements, and key events.

*   **State-of-the-Art Performance**
    * xxxxx




## Benchmark

Existing document-parsing benchmarks often provide limited coverage of complex layouts and STEM content. To address this, we constructed an in-house benchmark comprising 1,078 page-level images across nine major categories and over twenty sub-categories. Our model achieves the best performance on this benchmark.



## Quick Start
### 1. Installation
```shell
conda create -n logics-parsing-omni python=3.10
conda activate logics-parsing-omni

pip install -r Logics-Parsing-Omni/requirements.txt

```

### 2. Inference
```shell
python inference.py --image_url PATH_TO_INPUT_IMG --audio_url PATH_TO_INPUT_AUDIO --text_prompt "What can you see and hear? Answer in one short sentence."
```

## Acknowledgments


We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)

