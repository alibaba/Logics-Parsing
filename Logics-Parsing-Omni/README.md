<div align="center">
  <img src="imgs/logo.png" width="80%">
</div>

<p align="center">
    🤗 <a href="https://huggingface.co/Logics-MLLM/Logics-Parsing-Omni">Model</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2603.09677">Technical Report</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://github.com/alibaba/Logics-Parsing/tree/main/Logics-Parsing-Omni/README.md"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="18" style="vertical-align: middle;"></a> <a href="https://github.com/alibaba/Logics-Parsing/tree/main/Logics-Parsing-Omni">GitHub</a>
</p>


## News
* [2026/04/08] We release the [**OmniParsingBench**](https://huggingface.co/datasets/Logics-MLLM/OmniParsingBench), a comprehensive multimodal parsing benchmark covering Natural Image, Graphics, Audio, Natural Video, and Text-Rich Video.
* [2026/03/09] We release the [**Logics-Parsing-Omni**](https://github.com/alibaba/Logics-Parsing/tree/main/Logics-Parsing-Omni). For more details, please check our [**Technical Report**](https://arxiv.org/pdf/2603.09677).
* [2026/02/13] 🚀🚀🚀🚀🚀 We release Logics-Parsing-v2 Model.
* [2025/09/25] 🚀🚀🚀 We release Logics-Parsing Model.

## Introduction
Logics-Parsing-Omni is a unified Multimodal Large Language Model (MLLM) designed to bridge the gap between pixel-level structural parsing and semantic-level cognitive captioning. It provides strong performance in both fine-grained perception and high-level semantic understanding across documents, images, audio, and video.

<div align="center">
  <img src="imgs/overview.png" alt="Logics-Parsing-Omni Overview" width="800">
    <p><em>The construction of unified multi-modal parsing corpus and training pipeline of our proposed Logics-Parsing-Omni.</em></p>
</div>

<div align="center">
  <img src="imgs/showcase_all.png" alt="Showcase of the multifaceted capabilities of Logics-Parsing-Omni" width="600">
  <p><em>Showcase of the multifaceted capabilities of Logics-Parsing-Omni</em></p>
</div>


### Key Features

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
    *   Powered by a meticulously constructed omni-modal dataset, the **Logics-Parsing-Omni** model establishes a robust balance between fine-grained structural fidelity and deep semantic interpretation.
    *   Alongside the model, it introduces **OmniParsingBench**, a standardized evaluation infrastructure designed to quantitatively assess the full spectrum of parsing capabilities across documents, images, audio, and videos.



## OmniParsingBench Evaluation

To rigorously evaluate the unified parsing capabilities of our model across diverse modalities, we introduce **OmniParsingBench**—a comprehensive, large-scale evaluation corpus. It assesses the full spectrum of parsing performance across six primary domains: *Natural Image, Graphics, Document, Audio, Natural Video, and Text-Rich Video*. 

To provide a concise view of model capabilities, we aggregate fine-grained metrics into three core scores:
- **Perception (Perc.):** Evaluates signal precision and structural fidelity (e.g., spatio-temporal grounding, symbol extraction).
- **Cognition (Cog.):** Evaluates logical reasoning, semantic understanding, and hallucination resistance.
- **Overall (Ovr.):** The comprehensive performance metric.
Full evaluation details are released in the technical report.

<div align="center">
  <img src="imgs/omniparsingbench_performance.jpg" alt="OmniParsingBench performance of Logics-Parsing-Omni" style="width: 800px; height: 450px;">
  <p><em>OmniParsingBench performance of Logics-Parsing-Omni.</em></p>
</div>




<div align="center">
  <strong>Table 1: OmniParsingBench evaluation results on Image, Graphics, and Document</strong>
  <br><br>
  <table>
    <thead>
      <tr>
        <th rowspan="2" align="left">Model</th>
        <th colspan="3" align="center">Natural Image</th>
        <th colspan="3" align="center">Graphics</th>
        <th colspan="1" align="center">Document</th>
      </tr>
      <tr>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
        <th align="center">Perc.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td align="left">Gemini-3-Pro</td>
        <td align="center"><b>61.20</b></td>
        <td align="center">55.96</td>
        <td align="center"><b>66.44</b></td>
        <td align="center"><u>87.03</u></td>
        <td align="center"><b>84.21</b></td>
        <td align="center">87.43</td>
        <td align="center"><b>87.01</b></td>
      </tr>
      <tr>
        <td align="left">GPT-5.2</td>
        <td align="center">39.94</td>
        <td align="center">37.77</td>
        <td align="center">42.12</td>
        <td align="center">82.71</td>
        <td align="center">69.86</td>
        <td align="center"><u>91.48</u></td>
        <td align="center">77.43</td>
      </tr>
      <tr>
        <td align="left">Qwen3.5-397B-A17B</td>
        <td align="center">57.40</td>
        <td align="center"><b>56.95</b></td>
        <td align="center">57.85</td>
        <td align="center">82.81</td>
        <td align="center">73.77</td>
        <td align="center">83.13</td>
        <td align="center">81.09</td>
      </tr>
      <tr>
        <td align="left">Qwen3-VL-235B-A22B</td>
        <td align="center">58.61</td>
        <td align="center"><u>56.23</u></td>
        <td align="center">60.99</td>
        <td align="center">79.49</td>
        <td align="center">71.51</td>
        <td align="center">83.46</td>
        <td align="center">84.47</td>
      </tr>
      <tr>
        <td align="left">Qwen3-VL-30B-A3B</td>
        <td align="center">50.92</td>
        <td align="center">48.91</td>
        <td align="center">52.94</td>
        <td align="center">73.25</td>
        <td align="center">65.71</td>
        <td align="center">79.36</td>
        <td align="center">78.94</td>
      </tr>
      <tr>
        <td align="left">Qwen3-Omni-30B-A3B</td>
        <td align="center">47.36</td>
        <td align="center">46.85</td>
        <td align="center">47.88</td>
        <td align="center">77.46</td>
        <td align="center">70.75</td>
        <td align="center">78.25</td>
        <td align="center">73.50</td>
      </tr>
      <tr>
        <td align="left"><b>Logics-Parsing-Omni(Ours)</b></td>
        <td align="center"><u>59.07</u></td>
        <td align="center">53.77</td>
        <td align="center"><u>64.37</u></td>
        <td align="center"><b>88.66</b></td>
        <td align="center"><u>82.01</u></td>
        <td align="center"><b>92.12</b></td>
        <td align="center"><u>84.90</u></td>
      </tr>
    </tbody>
  </table>
  <br>

  <strong>Table 2: OmniParsingBench evaluation results on Audio and Video</strong>
  <br><br>
  <table>
    <thead>
      <tr>
        <th rowspan="2" align="left">Model</th>
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
        <th align="center">Ovr.</th>
        <th align="center">Perc.</th>
        <th align="center">Cog.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td align="left">Gemini-3-Pro</td>
        <td align="center"><u>79.40</u></td>
        <td align="center"><b>72.90</b></td>
        <td align="center">85.89</td>
        <td align="center"><b>63.40</b></td>
        <td align="center"><b>57.87</b></td>
        <td align="center"><b>68.92</b></td>
        <td align="center"><u>64.37</u></td>
        <td align="center"><b>58.54</b></td>
        <td align="center"><u>70.20</u></td>
      </tr>
      <tr>
        <td align="left">Qwen3-Omni-30B-A3B</td>
        <td align="center">75.17</td>
        <td align="center">62.13</td>
        <td align="center"><u>88.22</u></td>
        <td align="center">45.23</td>
        <td align="center">34.15</td>
        <td align="center">56.32</td>
        <td align="center">26.86</td>
        <td align="center">10.22</td>
        <td align="center">43.50</td>
      </tr>
      <tr>
        <td align="left"><b>Logics-Parsing-Omni(Ours)</b></td>
        <td align="center"><b>79.63</b></td>
        <td align="center"><u>69.27</u></td>
        <td align="center"><b>89.99</b></td>
        <td align="center"><u>61.12</u></td>
        <td align="center"><u>56.09</u></td>
        <td align="center"><u>66.15</u></td>
        <td align="center"><b>69.12</b></td>
        <td align="center"><u>57.39</u></td>
        <td align="center"><b>80.85</b></td>
      </tr>
    </tbody>
  </table>
  <p align="left"><em>Note: <b>Bold text</b> indicates the best result, and <u>underlined text</u> indicates the second-best result.</em></p>
</div>


As detailed in the tables above, **Logics-Parsing-Omni** demonstrates highly competitive capabilities across all diverse modalities:
- **Dominance in Complex Modalities:** Our model consistently surpasses all evaluated baselines—including the leading proprietary **Gemini-3-Pro**—in the *Graphics*, *Audio*, and *Text-Rich Video* domains.
- **Exceptional Cognitive Reasoning:** The superiority is particularly pronounced in the *Cognition* metric. Logics-Parsing-Omni exhibits exceptional logical reasoning and semantic understanding, achieving top-tier scores such as **92.12** in Graphics and **80.85** in Text-Rich Video.
These results firmly validate that our model successfully bridges fundamental signal detection with complex multi-modal interpreting.


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
