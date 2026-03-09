<div align="center">
  <img src="Logics-Parsing-Omni/imgs/logo.png" width="80%" >
</div>

<p align="center">
    🤗 <a href="https://huggingface.co/Logics-MLLM/Logics-Parsing-Omni">Model</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/">Technical Report</a>
</p>

## News
* [2026/03/10] We release the [**Logics-Parsing-Omni**](https://github.com/alibaba/Logics-Parsing/Logics-Parsing/Logics-Parsing-Omni/README.md). For more details, please check our [**Logics-Parsing-Omni paper**](https://arxiv.org/).
* [2026/02/13] 🚀🚀🚀🚀🚀 We release **Logics-Parsing-v2** Model.
* [2025/09/25] 🚀🚀🚀 We have released the [**Logics-Parsing**](https://github.com/alibaba/Logics-Parsing/Logics-Parsing/README.md). For more details, please check our [**Logics-Parsing paper**](https://arxiv.org/abs/2509.19760).

## Introduction
Logics-Parsing-Omni is a unified Multimodal Large Language Model (MLLM) designed to bridge the gap between pixel-level structural parsing and semantic-level cognitive captioning. It delivers breakthroughs in fine-grained perception and high-level cognition across documents, images, audio, and video.

<div align="center">
  <img src="Logics-Parsing-Omni/imgs/overview.png" alt="Logics-Parsing-Omni 概览" style="width: 800px; height: 250px;">
</div>

## Showcase
<div align="center">
  <img src="Logics-Parsing-Omni/imgs/showcase_all.png" alt="Showcase of the multifaceted capabilities of Logics-Parsing-Omni" style="width: 600px; height: 1500px;">
  <p><em>Showcase of the multifaceted capabilities of Logics-Parsing-Omni</em></p>
</div>


## Key Features

*   **Omni-Modal Unified Parsing Framework**
    *   It introduces a **progressive three-level paradigm**—integrating Holistic Detection, Fine-grained Recognition, and Semantic Interpretation—that fundamentally bridges the gap between pixel-based perception and logic-based cognition.
    *   It transforms unstructured multimodal signals into a standardized, machine-readable schema that is inherently **Locatable, Enumerable, and Traceable**, forming an indispensable part of fact-based reasoning chains.

*   **Knowledge-Intensive Document & Graphic Interpretation**
    *   Surpassing traditional OCR and generic LLM pipelines, it jointly parses structural elements (e.g., dense text, layout, tables, formulas) and deep semantics (e.g., complex illustrations) with high layout fidelity.
    *   It overcomes the bottleneck of generic image understanding by explicitly extracting dense, attribute-rich underlying data series, axis labels, and spatial topologies from scientific charts and technical diagrams to support reasoning.

*   **Long-Form Audio-Visual & Educational Content Parsing**
    *   Moving beyond flat ASR linear transcripts and generic video summaries, it dynamically synchronizes audio cues with critical visual contexts, explicitly capturing missed details like slides, whiteboards, and code.
    *   It is specifically optimized for long-form educational videos, successfully extracting structured pedagogical organizations (e.g., chapter hierarchies, key concepts) and dynamic narrative logic while mitigating information redundancy and topic drift.

*   **Data-Centric Optimization & Comprehensive Benchmarking**
    *   Powered by a meticulously constructed omni-modal dataset, the **Logics-Parsing-Omni** model establishes a robust balance between fine-grained structural fidelity and deep semantic interpretation, directly enabling downstream tasks like robust RAG and intelligent tutoring.
    *   Alongside the model, it introduces **OmniParsingBench**, a standardized evaluation infrastructure designed to quantitatively assess the full spectrum of parsing capabilities across documents, images, audio, and videos.

## Experimental Results

### OmniParsingBench Evaluation

To rigorously evaluate the unified parsing capabilities of our model across diverse modalities, we construct **OmniParsingBench**, a comprehensive, large-scale, and high-quality evaluation corpus. Unlike traditional single-task benchmarks, OmniParsingBench is designed to assess the full spectrum of parsing performance—from fundamental signal detection to complex semantic reasoning—across six primary domains: Document, Natural Image, Graphics, Audio, Natural Video, and Text-Rich Video.

Our evaluation framework strictly aligns with the proposed three-stage architecture (L1–L3), systematically assessing performance from L1-Holistic Detection (spatio-temporal grounding and classification), L2-Fine-grained Recognition (symbol extraction, attribute identification, and structural recovery), to L3-Multi-level Interpreting (semantic consistency and hallucination resistance). To provide a concise view of model capabilities, we aggregate these fine-grained metrics into two core scores: **Perception**, which evaluates signal precision and structural fidelity (dominating L1 and L2), and **Cognition**, which evaluates logical reasoning and semantic understanding (dominating L3).

<div align="center">
  <!-- 请将此处的 src 替换为你实际的图片路径 -->
  <img src="Logics-Parsing-Omni/imgs/omniparsingbench_performance.jpg" alt="OmniParsingBench performance of Logics-Parsing-Omni" style="width: 800px; height: 500px;">
  <p><em>OmniParsingBench performance of Logics-Parsing-Omni.</em></p>
</div>

As detailed in the table below, Logics-Parsing-Omni demonstrates highly competitive or state-of-the-art capabilities across all six diverse modalities when evaluated on Overall (Ovr.), Perception (Perc.), and Cognition (Cog.) metrics. 

Notably, our model consistently surpasses all evaluated baselines—including the leading proprietary Gemini-3 Pro—in the Graphics, Audio, and Text-Rich Video domains. This superiority is particularly pronounced in the Cognition metric, where Logics-Parsing-Omni exhibits exceptional logical reasoning and semantic understanding, achieving top-tier scores such as 92.19 in Graphics and 84.22 in Text-Rich Video. While Gemini-3 Pro maintains an advantage in the fundamental perception of Natural Images and Documents, as well as a marginal lead in Natural Video, our model significantly outperforms other open-weight counterparts (e.g., the Qwen series) across the board. These quantitative results validate the efficacy of our L1–L3 architecture, demonstrating that Logics-Parsing-Omni successfully bridges fundamental signal detection with complex multi-modal interpreting.

<br>

<div align="center">
  <strong>Table 1: Performance comparison of various models on OmniParsingBench.</strong>
  <br><br>
  <table>
    <thead>
      <tr>
        <th rowspan="2" align="left">Model</th>
        <th colspan="3" align="center">Natural Image</th>
        <th colspan="3" align="center">Graphics</th>
        <th colspan="1" align="center">Document</th>
        <th colspan="3" align="center">Audio</th>
        <th colspan="3" align="center">Natural Video</th>
        <th colspan="3" align="center">Text-Rich Video</th>
      </tr>
      <tr>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
        <th align="center">Perc.</th>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td align="left">Gemini-3 Pro</td>
        <td align="center"><b>69.72</b></td>
        <td align="center"><b>63.15</b></td>
        <td align="center"><b>76.29</b></td>
        <td align="center"><u>86.66</u></td>
        <td align="center"><b>85.33</b></td>
        <td align="center"><u>87.98</u></td>
        <td align="center"><b>87.01</b></td>
        <td align="center"><u>52.68</u></td>
        <td align="center"><b>73.99</b></td>
        <td align="center"><u>31.37</u></td>
        <td align="center"><b>44.56</b></td>
        <td align="center"><b>52.54</b></td>
        <td align="center"><b>36.57</b></td>
        <td align="center"><u>66.50</u></td>
        <td align="center"><u>53.47</u></td>
        <td align="center"><u>79.53</u></td>
      </tr>
      <tr>
        <td align="left">GPT-5.2</td>
        <td align="center">52.84</td>
        <td align="center">46.07</td>
        <td align="center">59.61</td>
        <td align="center">81.07</td>
        <td align="center">70.34</td>
        <td align="center"><u>91.80</u></td>
        <td align="center">77.43</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
      </tr>
      <tr>
        <td align="left">Qwen3.5-397B-A17B</td>
        <td align="center">54.80</td>
        <td align="center">47.12</td>
        <td align="center">62.48</td>
        <td align="center">78.79</td>
        <td align="center">74.44</td>
        <td align="center">83.15</td>
        <td align="center">81.09</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
      </tr>
      <tr>
        <td align="left">Qwen3-VL-235B-A22B</td>
        <td align="center">59.34</td>
        <td align="center"><u>50.62</u></td>
        <td align="center">68.05</td>
        <td align="center">78.00</td>
        <td align="center">72.15</td>
        <td align="center">83.85</td>
        <td align="center">84.47</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
      </tr>
      <tr>
        <td align="left">Qwen3-VL-30B-A3B</td>
        <td align="center">53.73</td>
        <td align="center">46.47</td>
        <td align="center">60.99</td>
        <td align="center">73.05</td>
        <td align="center">66.31</td>
        <td align="center">79.78</td>
        <td align="center">78.94</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
        <td align="center">--</td>
      </tr>
      <tr>
        <td align="left">Qwen3-Omni-30B-A3B</td>
        <td align="center">51.30</td>
        <td align="center">44.88</td>
        <td align="center">57.72</td>
        <td align="center">74.93</td>
        <td align="center">71.39</td>
        <td align="center">78.47</td>
        <td align="center">73.50</td>
        <td align="center">48.17</td>
        <td align="center">62.86</td>
        <td align="center">33.48</td>
        <td align="center">27.98</td>
        <td align="center">29.35</td>
        <td align="center">26.60</td>
        <td align="center">33.76</td>
        <td align="center">8.40</td>
        <td align="center">59.11</td>
      </tr>
      <tr>
        <td align="left"><b>Logics-Parsing-Omni</b></td>
        <td align="center"><u>62.46</u></td>
        <td align="center">50.53</td>
        <td align="center"><u>74.38</u></td>
        <td align="center"><b>87.43</b></td>
        <td align="center"><u>82.67</u></td>
        <td align="center"><b>92.19</b></td>
        <td align="center"><u>84.90</u></td>
        <td align="center"><b>53.75</b></td>
        <td align="center"><u>70.03</u></td>
        <td align="center"><b>37.48</b></td>
        <td align="center"><u>43.78</u></td>
        <td align="center"><u>51.76</u></td>
        <td align="center"><u>35.79</u></td>
        <td align="center"><b>69.37</b></td>
        <td align="center"><b>54.52</b></td>
        <td align="center"><b>84.22</b></td>
      </tr>
    </tbody>
  </table>
  <p align="left"><em>Note: <b>Bold text</b> indicates the best result, and <u>underlined text</u> indicates the second-best result.</em></p>
</div>

## Quick Start

### 1. Installation
```shell
conda create -n logics-parsing-omni python=3.10
conda activate logics-parsing-omni

pip install -r requirements.txt
```

### 2. Inference

We provide a unified multimodal inference script (`inference_omni.py`) that supports **12 pre-defined tasks** across 4 different modalities (Single Image, Multi-Image, Audio, and Video). 

You can easily test different capabilities using the `--task` argument. Additionally, all pre-defined tasks support bilingual prompts. You can switch between English and Chinese using the `--language` argument (`en` or `ch`, defaults to `en`).

#### Option A: Run a Pre-defined Task
Test a specific capability using built-in prompts and assets by passing the corresponding task name and your preferred language:

```shell
# Example: Run the natural video parsing task with the English prompt (default)
python inference_omni.py --task natural_video_parsing --language en

# Example: Run the document structure parsing task with the Chinese prompt
python inference_omni.py --task document_structure_parsing --language ch
```

#### Option B: Run a Custom Task (CLI Mode)
If you want to test your own files and prompts, use the `--task custom` mode along with the specific modality argument. *(Note: The `--language` argument is ignored in custom mode since you provide the prompt directly).*

```shell
# Example 1: Single Image Inference
python inference_omni.py --task custom \
    --image_paths path/to/image.jpg \
    --text_prompt "Describe the content of this image."

# Example 2: Multi-Image Inference
python inference_omni.py --task custom \
    --image_paths path/to/image1.jpg path/to/image2.jpg \
    --text_prompt "What are the differences between these two images?"

# Example 3: Single Audio Inference
python inference_omni.py --task custom \
    --audio_path path/to/audio.wav \
    --text_prompt "Please transcribe this audio."

# Example 4: Single Video Inference (with audio extraction)
python inference_omni.py --task custom \
    --video_path path/to/video.mp4 \
    --use_audio_in_video \
    --text_prompt "Please summarize this video."
```

### 3. Supported Pre-defined Tasks

Here is the complete list of built-in tasks you can pass to the `--task` argument, along with their corresponding English and Chinese prompts:

| Modality | Task Argument (`--task`) | English Prompt (`--language en`) | Chinese Prompt (`--language ch`) |
| :--- | :--- | :--- | :--- |
| **Single Image** | `document_structure_parsing` | Output the parsing results of this document in JSON format. | 以JSON格式输出此文档的解析结果。 |
| | `document_structure_and_semantic_parsing`| Output the parsing results of this document in JSON format. Include descriptions for illustrations, structurally parse natural images and graphics, and add a global overview at the end. Use the same language as the document text. | 以JSON格式输出此文档的解析结果。若有插图请进行描述，对自然图像和图表进行结构化分析，文末需包含全局文档描述，且语言与文档一致。 |
| | `natural_image_parsing` | Please detect text and entities in the image, extract structured information such as bounding boxes, labels, attributes, and detailed descriptions, and provide a global image description. Output the results in JSON format. | 请检测图中的文本与实体，提取边界框、标签、属性及详细描述等结构化信息，并给出全局图像描述。结果以JSON格式输出。 |
| | `chart_image_parsing` | Perform an in-depth parsing of the image, locate text and charts, extract their bounding boxes, labels, parsing results, and descriptions, and provide a global image description. Please present the results in JSON format. | 对图片进行深度解析，定位文本和图表，提取其边界框、标签、解析结果与描述，并给出全局图像描述，请用JSON格式呈现。 |
| | `geometric_image_parsing` | Please detect the text and geometric shapes in the image, extract bounding boxes, labels, parsing results, and detailed descriptions, and provide a global image description. Output the results in JSON format. | 请检测图中的文本和几何形状，提取边界框、标签、解析结果及详细描述，并提供全局图像描述。结果以JSON格式输出。 |
| **Audio** | `audio_parsing` | Divide the audio into continuous segments primarily based on speaker and VAD (split non-speech parts by audio classification); segments should include timestamps, classification labels, ASR, and speaker IDs, with a global description added at the end, output in JSON format. | 以说话人及VAD为首要依据将音频划分为连续片段（无人声处按音频分类拆分），段内包含时间戳、分类标签、ASR及说话人ID，末尾添加全局描述并以JSON格式输出。 |
| **Video** | `natural_video_parsing` | Split the video into continuous time segments based on visual semantic changes; for each segment, extract timestamps, internal audio split points and classification labels (following the principle of prioritizing human voice VAD, and classifying non-vocal parts by audio type) and video attributes. Finally, integrate a global audio-visual description, ASR (including speaker distinction), and language information. Please output in JSON format. | 基于视觉语义变化将视频分割成连续的时间片段；针对每个片段，提取时间戳、内部音频的切分点与分类标签（划分遵循人声VAD优先，非人声进行音频分类的原则）及视频属性。最后整合全局音视频描述、ASR（含说话人区分）和语言信息。请以JSON格式输出。 |
| | `camera_aware_video_parsing` | Describe the video content and explain its camera movement features, while simultaneously extracting the timestamps and camera movement labels of the visual segments, and output in JSON format. | 描述视频内容并说明其运镜特点，同时提取视觉片段的时间戳与运镜标签，以JSON格式输出。 |
| | `text_rich_video_parsing` | Please analyze the video using OCR information stability as the basis for segmentation, extract the timestamp, OCR, and ASR content of each segment in chronological order, add a global audio-video description at the end, and output the result in JSON format. | 请以OCR信息稳定性为分段依据分析视频，按时间顺序依次提取各分段的时间戳、OCR及ASR内容，并在最后补充全局音视频描述，输出JSON格式结果。 |
| | `text_rich_video_in_depth_caption` | Based on the input course video, generate a course description report that is clearly structured, detailed, and easy for learners to read. | 根据输入的课程视频，生成一份结构清晰、内容详尽、易于学习者阅读的课程描述报告。 |
| **Multi-Image** | `natural_image_diff_parsing` | Generate structured analysis results for the edit from the first image to the second image. List all changed elements item by item, providing corresponding bounding boxes, labels, attributes, and descriptions; finally, provide a global editing description summarizing the overall changes. Output in JSON format. | 生成从第一张图编辑到第二张图的结构化解析结果。逐项列出所有变化元素，并给出对应的边界框、标签、属性及描述等信息；最后给出全局编辑描述总结整体变化。以JSON格式输出。 |
| | `geometric_diff_parsing` | Generate the analysis results of geometric edits from the first image to the second image. The content must include structured parsing of all changed geometric elements, geometric and quantitative relationships, and provide a global editing instruction summarizing the overall changes. Output in JSON format. | 生成从第一张图到第二张图的几何编辑解析结果。内容需包含所有变化几何元素的结构化解析、几何与定量关系，并给出总结整体变化的全局编辑指令。以JSON格式输出。 |

## Acknowledgments

We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)
