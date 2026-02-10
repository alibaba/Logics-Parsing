<div align="center">
  <img src="imgs/logo.jpg" width="80%" >
</div>


<p align="center">
    🤗 <a href="https://huggingface.co/Logics-MLLM/Logics-Parsing">Model</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/studios/Alibaba-DT/Logics-Parsing/summary">Demo</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2509.19760">Technical Report</a>
</p>

## Introduction
<div align="center">
  <img src="imgs/overview.png" alt="LogicsDocBench 概览" style="width: 800px; height: 250px;">
</div>

<div align="center">
  <table style="width: 800px;">
    <tr>
      <td align="center">
        <img src="imgs/report.gif" alt="研报示例">
      </td>
      <td align="center">
        <img src="imgs/chemistry.gif" alt="化学分子式示例">
      </td>
      <td align="center">
        <img src="imgs/paper.gif" alt="论文示例">
      </td>
      <td align="center">
        <img src="imgs/handwritten.gif" alt="手写示例">
      </td>
    </tr>
    <tr>
      <td align="center"><b>report</b></td>
      <td align="center"><b>chemistry</b></td>
      <td align="center"><b>paper</b></td>
      <td align="center"><b>handwritten</b></td>
    </tr>
  </table>
</div>



Logics-Parsing is a powerful, end-to-end document parsing model built upon a general Vision-Language Model (VLM) through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). It excels at accurately analyzing and structuring highly complex documents.

## Key Features

*   **Effortless End-to-End Processing**
    *   Our single-model architecture eliminates the need for complex, multi-stage pipelines. Deployment and inference are straightforward, going directly from a document image to structured output.
    *   It demonstrates exceptional performance on documents with challenging layouts.

*   **Advanced Content Recognition**
    *   It accurately recognizes and structures difficult content, including intricate scientific formulas.
    *   Chemical structures are intelligently identified and can be represented in the standard **SMILES** format.

*   **Rich, Structured HTML Output**
    *   The model generates a clean HTML representation of the document, preserving its logical structure.
    *   Each content block (e.g., paragraph, table, figure, formula) is tagged with its **category**, **bounding box coordinates**, and **OCR text**.
    *   It automatically identifies and filters out irrelevant elements like headers and footers, focusing only on the core content.

*   **State-of-the-Art Performance**
    * Logics-Parsing achieves the best performance on our in-house benchmark, which is specifically designed to comprehensively evaluate a model’s parsing capability on complex-layout documents and STEM content.





## Benchmark

Existing document-parsing benchmarks often provide limited coverage of complex layouts and STEM content. To address this, we constructed an in-house benchmark comprising 1,078 page-level images across nine major categories and over twenty sub-categories. Our model achieves the best performance on this benchmark.
<div align="center">
  <img src="imgs/BenchCls.png">
</div>
<table>
    <tr>
        <td rowspan="2">Model Type</td>
        <td rowspan="2">Methods</td>
        <td colspan="2">Overall <sup>Edit</sup> ↓</td>
        <td colspan="2">Text Edit <sup>Edit</sup> ↓</td>
        <td colspan="2">Formula <sup>Edit</sup> ↓</td>
        <td colspan="2">Table <sup>TEDS</sup> ↑</td>
        <td colspan="2">Table <sup>Edit</sup> ↓</td>
        <td colspan="2">ReadOrder<sup>Edit</sup> ↓</td>
        <td rowspan="1">Chemistry<sup>Edit</sup> ↓</td>
        <td rowspan="1">HandWriting<sup>Edit</sup> ↓</td>
    </tr>
    <tr>
        <td>EN</td>
        <td>ZH</td>
        <td>EN</td>
        <td>ZH</td>
        <td>EN</td>
        <td>ZH</td>
        <td>EN</td>
        <td>ZH</td>
        <td>EN</td>
        <td>ZH</td>
        <td>EN</td>
        <td>ZH</td>
        <td>ALL</td>
        <td>ALL</td>
    </tr>
    <tr>
        <td rowspan="7">Pipeline Tools</td>
        <td>doc2x</td>
        <td>0.209</td>
        <td>0.188</td>
        <td>0.128</td>
        <td>0.194</td>
        <td>0.377</td>
        <td>0.321</td>
        <td>81.1</td>
        <td>85.3</td>
        <td><ins>0.148</ins></td>
        <td><ins>0.115</ins></td>
        <td>0.146</td>
        <td>0.122</td>
        <td>1.0</td>
        <td>0.307</td>
    </tr>
    <tr>
        <td>Textin</td>
        <td>0.153</td>
        <td>0.158</td>
        <td>0.132</td>
        <td>0.190</td>
        <td>0.185</td>
        <td>0.223</td>
        <td>76.7</td>
        <td><ins>86.3</ins></td>
        <td>0.176</td>
        <td><b>0.113</b></td>
        <td><b>0.118</b></td>
        <td><b>0.104</b></td>
        <td>1.0</td>
        <td>0.344</td>
    </tr>
    <tr>
        <td>mathpix<sup>*</sup></td>
        <td><ins>0.128</ins></td>
        <td><ins>0.146</ins></td>
        <td>0.128</td>
        <td><ins>0.152</ins></td>
        <td><b>0.06</b></td>
        <td><b>0.142</b></td>
        <td><b>86.2</b></td>
        <td><b>86.6</b></td>
        <td><b>0.120</b></td>
        <td>0.127</td>
        <td>0.204</td>
        <td>0.164</td>
        <td>0.552</td>
        <td>0.263</td>
    </tr>
    <tr>
        <td>PP_StructureV3</td>
        <td>0.220</td>
        <td>0.226</td>
        <td>0.172</td>
        <td>0.29</td>
        <td>0.272</td>
        <td>0.276</td>
        <td>66</td>
        <td>71.5</td>
        <td>0.237</td>
        <td>0.193</td>
        <td>0.201</td>
        <td>0.143</td>
        <td>1.0</td>
        <td>0.382</td>
    </tr>
    <tr>
        <td>Mineru2</td>
        <td>0.212</td>
        <td>0.245</td>
        <td>0.134</td>
        <td>0.195</td>
        <td>0.280</td>
        <td>0.407</td>
        <td>67.5</td>
        <td>71.8</td>
        <td>0.228</td>
        <td>0.203</td>
        <td>0.205</td>
        <td>0.177</td>
        <td>1.0</td>
        <td>0.387</td>
    </tr>
    <tr>
        <td>Marker</td>
        <td>0.324</td>
        <td>0.409</td>
        <td>0.188</td>
        <td>0.289</td>
        <td>0.285</td>
        <td>0.383</td>
        <td>65.5</td>
        <td>50.4</td>
        <td>0.593</td>
        <td>0.702</td>
        <td>0.23</td>
        <td>0.262</td>
        <td>1.0</td>
        <td>0.50</td>
    </tr>
    <tr>
        <td>Pix2text</td>
        <td>0.447</td>
        <td>0.547</td>
        <td>0.485</td>
        <td>0.577</td>
        <td>0.312</td>
        <td>0.465</td>
        <td>64.7</td>
        <td>63.0</td>
        <td>0.566</td>
        <td>0.613</td>
        <td>0.424</td>
        <td>0.534</td>
        <td>1.0</td>
        <td>0.95</td>
    </tr>
    <tr>
        <td rowspan="8">Expert VLMs</td>
        <td>Dolphin</td>
        <td>0.208</td>
        <td>0.256</td>
        <td>0.149</td>
        <td>0.189</td>
        <td>0.334</td>
        <td>0.346</td>
        <td>72.9</td>
        <td>60.1</td>
        <td>0.192</td>
        <td>0.35</td>
        <td>0.160</td>
        <td>0.139</td>
        <td>0.984</td>
        <td>0.433</td>
    </tr>
    <tr>
        <td>dots.ocr</td>
        <td>0.186</td>
        <td>0.198</td>
        <td><ins>0.115</ins></td>
        <td>0.169</td>
        <td>0.291</td>
        <td>0.358</td>
        <td>79.5</td>
        <td>82.5</td>
        <td>0.172</td>
        <td>0.141</td>
        <td>0.165</td>
        <td>0.123</td>
        <td>1.0</td>
        <td><ins>0.255</ins></td>
    </tr>
    <tr>
        <td>MonkeyOcr</td>
        <td>0.193</td>
        <td>0.259</td>
        <td>0.127</td>
        <td>0.236</td>
        <td>0.262</td>
        <td>0.325</td>
        <td>78.4</td>
        <td>74.7</td>
        <td>0.186</td>
        <td>0.294</td>
        <td>0.197</td>
        <td>0.180</td>
        <td>1.0</td>
        <td>0.623</td>
    </tr>
    <tr>
        <td>OCRFlux</td>
        <td>0.252</td>
        <td>0.254</td>
        <td>0.134</td>
        <td>0.195</td>
        <td>0.326</td>
        <td>0.405</td>
        <td>58.3</td>
        <td>70.2</td>
        <td>0.358</td>
        <td>0.260</td>
        <td>0.191</td>
        <td>0.156</td>
        <td>1.0</td>
        <td>0.284</td>
    </tr>
    <tr>
        <td>Gotocr</td>
        <td>0.247</td>
        <td>0.249</td>
        <td>0.181</td>
        <td>0.213</td>
        <td>0.231</td>
        <td>0.318</td>
        <td>59.5</td>
        <td>74.7</td>
        <td>0.38</td>
        <td>0.299</td>
        <td>0.195</td>
        <td>0.164</td>
        <td>0.969</td>
        <td>0.446</td>
    </tr>
    <tr>
        <td>Olmocr</td>
        <td>0.341</td>
        <td>0.382</td>
        <td>0.125</td>
        <td>0.205</td>
        <td>0.719</td>
        <td>0.766</td>
        <td>57.1</td>
        <td>56.6</td>
        <td>0.327</td>
        <td>0.389</td>
        <td>0.191</td>
        <td>0.169</td>
        <td>1.0</td>
        <td>0.294</td>
    </tr>
    <tr>
        <td>SmolDocling</td>
        <td>0.657</td>
        <td>0.895</td>
        <td>0.486</td>
        <td>0.932</td>
        <td>0.859</td>
        <td>0.972</td>
        <td>18.5</td>
        <td>1.5</td>
        <td>0.86</td>
        <td>0.98</td>
        <td>0.413</td>
        <td>0.695</td>
        <td>1.0</td>
        <td>0.927</td>
    </tr>
    <tr>
        <td><b>Logics-Parsing</b></td>
        <td><b>0.124</b></td>
        <td><b>0.145</b></td>
        <td><b>0.089</b></td>
        <td><b>0.139</b></td>
        <td><ins>0.106</ins></td>
        <td><ins>0.165</ins></td>
        <td>76.6</td>
        <td>79.5</td>
        <td>0.165</td>
        <td>0.166</td>
        <td><ins>0.136</ins></td>
        <td><ins>0.113</ins></td>
        <td><b>0.519</b></td>
        <td><b>0.252</b></td>
    </tr>
    <tr>
        <td rowspan="5">General VLMs</td>
        <td>Qwen2VL-72B</td>
        <td>0.298</td>
        <td>0.342</td>
        <td>0.142</td>
        <td>0.244</td>
        <td>0.431</td>
        <td>0.363</td>
        <td>64.2</td>
        <td>55.5</td>
        <td>0.425</td>
        <td>0.581</td>
        <td>0.193</td>
        <td>0.182</td>
        <td>0.792</td>
        <td>0.359</td>
    </tr>
    <tr>
        <td>Qwen2.5VL-72B</td>
        <td>0.233</td>
        <td>0.263</td>
        <td>0.162</td>
        <td>0.24</td>
        <td>0.251</td>
        <td>0.257</td>
        <td>69.6</td>
        <td>67</td>
        <td>0.313</td>
        <td>0.353</td>
        <td>0.205</td>
        <td>0.204</td>
        <td>0.597</td>
        <td>0.349</td>
    </tr>
    <tr>
        <td>Doubao-1.6</td>
        <td>0.188</td>
        <td>0.248</td>
        <td>0.129</td>
        <td>0.219</td>
        <td>0.273</td>
        <td>0.336</td>
        <td>74.9</td>
        <td>69.7</td>
        <td>0.180</td>
        <td>0.288</td>
        <td>0.171</td>
        <td>0.148</td>
        <td>0.601</td>
        <td>0.317</td>
    </tr>
    <tr>
        <td>GPT-5</td>
        <td>0.242</td>
        <td>0.373</td>
        <td>0.119</td>
        <td>0.36</td>
        <td>0.398</td>
        <td>0.456</td>
        <td>67.9</td>
        <td>55.8</td>
        <td>0.26</td>
        <td>0.397</td>
        <td>0.191</td>
        <td>0.28</td>
        <td>0.88</td>
        <td>0.46</td>
    </tr>
    <tr>
        <td>Gemini2.5 pro</td>
        <td>0.185</td>
        <td>0.20</td>
        <td><ins>0.115</ins></td>
        <td>0.155</td>
        <td>0.288</td>
        <td>0.326</td>
        <td><ins>82.6</ins></td>
        <td>80.3</td>
        <td>0.154</td>
        <td>0.182</td>
        <td>0.181</td>
        <td>0.136</td>
        <td><ins>0.535</ins></td>
        <td>0.26</td>
    </tr>

</table>
<!-- 脚注说明 -->
<tr>
  <td colspan="5">
    <sup>*</sup> Tested on the v3/PDF Conversion API (August 2025 deployment).

  </td>
</tr>



## Quick Start
### 1. Installation
```shell
conda create -n logis-parsing python=3.10
conda activate logis-parsing

pip install -r requirement.txt

```
### 2. Download Model Weights

```
# Download our model from Modelscope.
pip install modelscope
python download_model.py -t modelscope

# Download our model from huggingface.
pip install huggingface_hub
python download_model.py -t huggingface
```

### 3. Inference
```shell
python3 inference.py --image_path PATH_TO_INPUT_IMG --output_path PATH_TO_OUTPUT --model_path PATH_TO_MODEL
```

## Acknowledgments


We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Mathpix](https://mathpix.com/)

