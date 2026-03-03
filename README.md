<div align="center">
  <img src="imgs/logo.png" width="80%" >
</div>


<p align="center">
    🤗 <a href="https://huggingface.co/Logics-MLLM/Logics-Parsing-Omni">Model</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/">Technical Report</a>
</p>

## News
* **`2026.01.15`** : We have released the [**Logics-Parsing-Omni**](https://huggingface.co/Logics-MLLM/Logics-Parsing-Omni). For more details, please check our [**Logics-Parsing-Omni paper**](https://arxiv.org/).
* **`2025.09.17`** : We have released the [**Logics-Parsing**](https://github.com/alibaba/Logics-Parsing/Logics-Parsing/README.md). For more details, please check our [**Logics-Parsing paper**](https://arxiv.org/abs/2509.19760).

## Introduction
Logics-Parsing-Omni is a unified Multimodal Large Language Model (MLLM) designed to bridge the gap between pixel-level structural parsing and semantic-level cognitive captioning. It delivers breakthroughs in fine-grained perception and high-level cognition across documents, images, audio, and video.

<div align="center">
  <img src="imgs/overview.png" alt="Logics-Parsing-Omni 概览" style="width: 800px; height: 250px;">
</div>


## Key Features

*   **Omni-Modal Unified Parsing Framework**
    *   It introduces a unified modeling paradigm that bridges fine-grained structural parsing (pixel-grounded localization) with high-level semantic interpretation (cognition-level understanding).
    *   By employing a joint optimization strategy, it achieves substantial and concurrent performance gains on both perception-centric and cognition-centric tasks.

*   **Holistic Document & Graphic Interpretation**
    *   Surpassing traditional OCR pipelines, it holistically deconstructs multimodal document elements, ranging from layout structure to embedded illustrations and complex charts.
    *   It significantly enhances the knowledge richness and descriptive accuracy of general images and complex graphics, providing robust support for downstream retrieval and QA applications.

*   **Long-Form Video & Educational Content Parsing**
    *   It extends unified parsing to videos by endowing the model with acute perception of dynamic visual cues, including camera language and speech content.
    *   It addresses the challenges of long-form educational videos by capturing missed visual details—such as slides, whiteboards, formulas, and code—that traditional ASR systems overlook.

*   **State-of-the-Art Performance**
    *   Logics-Parsing-Omni sets new standards on both LogicsParsingBench (for documents) and our internal OmniParsingBench (for audio-visual content), demonstrating a superior balance between structural fidelity and semantic depth.




## Experimental Results

As demonstrated by the evaluation results on the LogicsParsingBench and OmniDocBench-v1.5 benchmarks, our proposed Logics-Parsing-Omni achieves overall performance superior to most existing general and specialized document parsing models. It shows notable advantages in multilingual document parsing, particularly for Chinese content, while maintaining balanced and robust capabilities across various structural elements such as text, tables, and formulas. These results validate the effectiveness and generality of the proposed unified single-stage architecture and training strategy for multimodal and diverse document parsing tasks.

## Quick Start

### 1. Installation
```shell
conda create -n logics-parsing-omni python=3.10
conda activate logics-parsing-omni

pip install -r requirements.txt
```

### 2. Inference

We provide a unified multimodal inference script (`inference.py`) that supports **12 pre-defined tasks** across 4 different modalities (Single Image, Multi-Image, Audio, and Video). You can easily test different capabilities using the `--task` argument.

#### Option A: Run a Pre-defined Task
Test a specific capability using built-in prompts and assets by passing the corresponding task name:

```shell
# Example: Run the natural video parsing task
python inference.py --task natural_video_parsing

# Example: Run the document structure parsing task
python inference.py --task document_structure_parsing
```

#### Option B: Run a Custom Task (CLI Mode)
If you want to test your own files and prompts, use the `--task custom` mode along with the specific modality argument:

```shell
# Example 1: Single Image Inference
python inference.py --task custom \
    --image_paths path/to/image.jpg \
    --text_prompt "Describe the content of this image."

# Example 2: Multi-Image Inference
python inference.py --task custom \
    --image_paths path/to/image1.jpg path/to/image2.jpg \
    --text_prompt "What are the differences between these two images?"

# Example 3: Single Audio Inference
python inference.py --task custom \
    --audio_path path/to/audio.wav \
    --text_prompt "Please transcribe this audio."

# Example 4: Single Video Inference (with audio extraction)
python inference.py --task custom \
    --video_path path/to/video.mp4 \
    --use_audio_in_video \
    --text_prompt "Please summarize this video."
```

### 3. Supported Pre-defined Tasks

Here is the complete list of built-in tasks you can pass to the `--task` argument:

| Modality | Task Argument (`--task`) | Description |
| :--- | :--- | :--- |
| **Single Image** | `document_structure_parsing` | Document Structure Parsing |
| | `document_structure_and_semantic_parsing`| Document Structure and Semantic Parsing |
| | `natural_image_parsing` | Natural Image Parsing (Entities, BBoxes, Descriptions) |
| | `chart_image_parsing` | Chart Image Deep Parsing |
| | `geometric_image_parsing` | Geometric Shapes & Text Parsing |
| **Audio** | `audio_parsing` | Audio Segmentation, ASR, and VAD Parsing |
| **Video** | `natural_video_parsing` | Video Semantic Segmentation with Audio Extraction |
| | `camera_aware_video_parsing` | Camera Movement & Visual Tracking Parsing |
| | `text_rich_video_parsing` | OCR & ASR Extraction for Text-Rich Videos |
| | `text_rich_video_in_depth_caption` | In-depth Educational/Course Video Captioning |
| **Multi-Image** | `natural_image_diff_parsing` | Structural Difference Parsing between 2 Natural Images |
| | `geometric_diff_parsing` | Geometric Difference Parsing between 2 Images |



## Acknowledgments


We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)

