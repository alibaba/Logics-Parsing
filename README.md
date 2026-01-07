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
        <td>0.115</td>
        <td>0.146</td>
        <td>0.122</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>textin</td>
        <td>0.153</td>
        <td>0.158</td>
        <td>0.132</td>
        <td>0.190</td>
        <td>0.185</td>
        <td>0.223</td>
        <td>76.7</td>
        <td>86.3</td>
        <td>0.176</td>
        <td><ins>0.113</ins></td>
        <td><b>0.118</b></td>
        <td>0.104</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>Mathpix</td>
        <td>0.128</td>
        <td>0.146</td>
        <td>0.128</td>
        <td>0.152</td>
        <td><b>0.06</b></td>
        <td><ins>0.142</ins></td>
        <td><b>86.2</b></td>
        <td><ins>86.6</ins></td>
        <td><b>0.120</b></td>
        <td>0.127</td>
        <td>0.204</td>
        <td>0.164</td>
        <td>0.552</td>
    </tr>
    <tr>
        <td>pp_structure_v3</td>
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
    </tr>
    <tr>
        <td>mineru2</td>
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
    </tr>
    <tr>
        <td>marker</td>
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
    </tr>
    <tr>
        <td>pix2text</td>
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
    </tr>
    <tr>
        <td rowspan="8">Expert MLLMs</td>
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
    </tr>
    <tr>
        <td>dots.ocr</td>
        <td>0.186</td>
        <td>0.198</td>
        <td>0.115</td>
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
    </tr>
    <tr>
        <td>gotocr</td>
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
    </tr>
    <tr>
        <td>olmocr</td>
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
    </tr>
    <tr>
        <td>Logics-Parsing</td>
        <td><ins>0.124</ins></td>
        <td><ins>0.145</ins></td>
        <td><b>0.089</b></td>
        <td><ins>0.139</ins></td>
        <td>0.106</td>
        <td>0.165</td>
        <td>76.6</td>
        <td>79.5</td>
        <td>0.165</td>
        <td>0.166</td>
        <td><ins>0.136</ins></td>
        <td>0.113</td>
        <td>0.519</td>
    </tr>
    <tr>
        <td rowspan="9">General MLLMs</td>
        <td>Qwen2.5-VL-72B</td>
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
    </tr>
    <tr>
        <td>doubao-1.6</td>
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
    </tr>
    <tr>
        <td>Gemini2.5-Pro</td>
        <td>0.185</td>
        <td>0.20</td>
        <td>0.115</td>
        <td>0.155</td>
        <td>0.288</td>
        <td>0.326</td>
        <td><ins>82.6</ins></td>
        <td>80.3</td>
        <td>0.154</td>
        <td>0.182</td>
        <td>0.181</td>
        <td>0.136</td>
        <td>0.535</td>
    </tr>
    <tr>
        <td>Gemini3-Pro</td>
        <td>0.189</td>
        <td>0.179</td>
        <td>0.134</td>
        <td>0.163</td>
        <td>0.280</td>
        <td>0.267</td>
        <td><ins>82.6</ins></td>
        <td>83.9</td>
        <td>0.157</td>
        <td>0.155</td>
        <td>0.186</td>
        <td>0.131</td>
        <td><ins>0.512</ins></td>
    </tr>
    <tr>
        <td>Qwen3-VL-30B-A3B</td>
        <td>0.188</td>
        <td>0.195</td>
        <td>0.122</td>
        <td>0.202</td>
        <td>0.291</td>
        <td>0.297</td>
        <td>72.04</td>
        <td>73.59</td>
        <td>0.183</td>
        <td>0.165</td>
        <td>0.154</td>
        <td>0.116</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>Qwen3-VL-235B-A22B</td>
        <td>0.18</td>
        <td>0.156</td>
        <td>0.11</td>
        <td>0.146</td>
        <td>0.277</td>
        <td>0.214</td>
        <td>77.0</td>
        <td>82.4</td>
        <td>0.18</td>
        <td>0.176</td>
        <td>0.153</td>
        <td><b>0.090</b></td>
        <td>0.568</td>
    </tr>
    <tr>
        <td>Qwen3-Omni (Base)</td>
        <td>0.289</td>
        <td>0.254</td>
        <td>0.260</td>
        <td>0.272</td>
        <td>0.355</td>
        <td>0.287</td>
        <td>67.9</td>
        <td>69.41</td>
        <td>0.298</td>
        <td>0.290</td>
        <td>0.242</td>
        <td>0.166</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td><b>Logics-Parsing-Omni (Ours)</b></td>
        <td><b>0.122</b></td>
        <td><b>0.114</b></td>
        <td><ins>0.094</ins></td>
        <td><b>0.122</b></td>
        <td><ins>0.086</ins></td>
        <td><b>0.134</b></td>
        <td>79.55</td>
        <td><b>87.74</b></td>
        <td>0.162</td>
        <td><b>0.102</b></td>
        <td>0.146</td>
        <td><ins>0.096</ins></td>
        <td><b>0.458</b></td>
    </tr>
</table>


<table>
    <caption style="text-align:center; font-size:14px; font-weight:normal;">
        Table. Comparisons with State-of-the-art methods on OmniDocBench-v1.5
    </caption>
    <tr>
        <td>Model Type</td>
        <td>Methods</td>
        <td>Overall ↑</td>
        <td>Text Edit ↓</td>
        <td>Formula CDM ↑</td>
        <td>Table TEDS ↑</td>
        <td>Table TEDS-S ↑</td>
        <td>ReadOrder ↓</td>
    </tr>
    <!-- Pipeline Tools -->
    <tr>
        <td rowspan="3">Pipeline Tools</td>
        <td>Marker-1.8.2</td>
        <td>71.3</td>
        <td>0.206</td>
        <td>76.66</td>
        <td>57.88</td>
        <td>71.17</td>
        <td>0.250</td>
    </tr>
    <tr>
        <td>MinerU2-Pipe</td>
        <td>75.51</td>
        <td>0.209</td>
        <td>76.55</td>
        <td>70.90</td>
        <td>79.11</td>
        <td>0.225</td>
    </tr>
    <tr>
        <td>PP-StructureV3</td>
        <td>86.73</td>
        <td>0.073</td>
        <td>85.79</td>
        <td>81.68</td>
        <td>89.48</td>
        <td>0.073</td>
    </tr>
    <!-- General MLLMs -->
    <tr>
        <td rowspan="10">General MLLMs</td>
        <td>GPT-4o</td>
        <td>75.02</td>
        <td>0.217</td>
        <td>79.70</td>
        <td>67.07</td>
        <td>76.09</td>
        <td>0.148</td>
    </tr>
    <tr>
        <td>InternVL3</td>
        <td>80.33</td>
        <td>0.131</td>
        <td>83.42</td>
        <td>70.64</td>
        <td>77.74</td>
        <td>0.113</td>
    </tr>
    <tr>
        <td>InternVL3.5</td>
        <td>82.67</td>
        <td>0.142</td>
        <td>87.23</td>
        <td>75.00</td>
        <td>81.28</td>
        <td>0.125</td>
    </tr>
    <tr>
        <td>GPT-5</td>
        <td>86.23</td>
        <td>0.115</td>
        <td>88.6</td>
        <td>81.60</td>
        <td>86.42</td>
        <td>0.099</td>
    </tr>
    <tr>
        <td>Qwen2.5-VL-72B</td>
        <td>87.02</td>
        <td>0.094</td>
        <td>88.27</td>
        <td>82.15</td>
        <td>86.22</td>
        <td>0.102</td>
    </tr>
    <tr>
        <td>Gemini2.5-Pro</td>
        <td>88.03</td>
        <td>0.075</td>
        <td>85.82</td>
        <td>85.71</td>
        <td>90.29</td>
        <td>0.097</td>
    </tr>
    <tr>
        <td>Gemini3-Pro</td>
        <td>88.36</td>
        <td>0.062</td>
        <td>83.1</td>
        <td>88.19</td>
        <td>93.52</td>
        <td>0.072</td>
    </tr>
    <tr>
        <td>Qwen3-VL-30B-A3B</td>
        <td>83.61</td>
        <td>0.056</td>
        <td>79.4</td>
        <td>77.04</td>
        <td>81.94</td>
        <td>0.081</td>
    </tr>
    <tr>
        <td>Qwen3-VL-235B-A22B</td>
        <td>89.15</td>
        <td>0.069</td>
        <td>88.14</td>
        <td>86.21</td>
        <td>90.55</td>
        <td>0.068</td>
    </tr>
    <tr>
        <td>Qwen3-Omni (Base)</td>
        <td>73.18</td>
        <td>0.194</td>
        <td>64.7</td>
        <td>74.178</td>
        <td>77.68</td>
        <td>0.123</td>
    </tr>
    <!-- Specialized MLLMs -->
    <tr>
        <td rowspan="16">Specialized MLLMs</td>
        <td>Dolphin</td>
        <td>74.67</td>
        <td>0.125</td>
        <td>67.85</td>
        <td>68.70</td>
        <td>77.77</td>
        <td>0.124</td>
    </tr>
    <tr>
        <td>OCRFlux</td>
        <td>74.82</td>
        <td>0.193</td>
        <td>68.03</td>
        <td>75.75</td>
        <td>80.23</td>
        <td>0.202</td>
    </tr>
    <tr>
        <td>Mistral OCR</td>
        <td>78.83</td>
        <td>0.164</td>
        <td>82.84</td>
        <td>70.03</td>
        <td>78.04</td>
        <td>0.144</td>
    </tr>
    <tr>
        <td>POINTS-Reader</td>
        <td>80.98</td>
        <td>0.134</td>
        <td>79.20</td>
        <td>77.13</td>
        <td>81.66</td>
        <td>0.145</td>
    </tr>
    <tr>
        <td>Dolphin-1.5</td>
        <td>83.21</td>
        <td>0.092</td>
        <td>80.78</td>
        <td>78.06</td>
        <td>84.10</td>
        <td>0.080</td>
    </tr>
    <tr>
        <td>olmOCR</td>
        <td>81.79</td>
        <td>0.096</td>
        <td>86.04</td>
        <td>68.92</td>
        <td>74.77</td>
        <td>0.121</td>
    </tr>
    <tr>
        <td>MinerU2-VLM</td>
        <td>85.56</td>
        <td>0.078</td>
        <td>80.95</td>
        <td>83.54</td>
        <td>87.66</td>
        <td>0.086</td>
    </tr>
    <tr>
        <td>Nanonets-OCR-s</td>
        <td>85.59</td>
        <td>0.093</td>
        <td>85.90</td>
        <td>80.14</td>
        <td>85.57</td>
        <td>0.108</td>
    </tr>
    <tr>
        <td>MonkeyOCR-pro-1.2B</td>
        <td>86.96</td>
        <td>0.084</td>
        <td>85.02</td>
        <td>84.24</td>
        <td>89.02</td>
        <td>0.130</td>
    </tr>
    <tr>
        <td>Deepseek-OCR</td>
        <td>87.01</td>
        <td>0.073</td>
        <td>83.37</td>
        <td>84.97</td>
        <td>88.80</td>
        <td>0.086</td>
    </tr>
    <tr>
        <td>MonkeyOCR-3B</td>
        <td>87.13</td>
        <td>0.075</td>
        <td>87.45</td>
        <td>81.39</td>
        <td>85.92</td>
        <td>0.129</td>
    </tr>
    <tr>
        <td>dots.ocr</td>
        <td>88.41</td>
        <td>0.048</td>
        <td>83.22</td>
        <td>86.78</td>
        <td>90.62</td>
        <td>0.053</td>
    </tr>
    <tr>
        <td>OCRVerse</td>
        <td>88.56</td>
        <td>0.058</td>
        <td>86.91</td>
        <td>84.55</td>
        <td>88.45</td>
        <td>0.071</td>
    </tr>
    <tr>
        <td>MonkeyOCR-pro-3B</td>
        <td>88.85</td>
        <td>0.075</td>
        <td>87.25</td>
        <td>86.78</td>
        <td>90.63</td>
        <td>0.128</td>
    </tr>
    <tr>
        <td>MinerU2.5</td>
        <td>90.67</td>
        <td><ins>0.047</ins></td>
        <td>88.46</td>
        <td>88.22</td>
        <td>92.38</td>
        <td><ins>0.044</ins></td>
    </tr>
    <tr>
        <td>PaddleOCR-VL</td>
        <td><b>92.86</b></td>
        <td><b>0.035</b></td>
        <td><ins>91.22</ins></td>
        <td><b>90.89</b></td>
        <td><b>94.76</b></td>
        <td><b>0.043</b></td>
    </tr>
    <!-- Ours -->
    <tr>
        <td>General MLLMs</td>
        <td><b>Logics-Parsing-Omni (Ours)</b></td>
        <td><ins>92.42</ins></td>
        <td>0.052</td>
        <td><b>92.9</b></td>
        <td><ins>89.5</ins></td>
        <td><ins>92.9</ins></td>
        <td>0.052</td>
    </tr>
</table>


## Quick Start
### 1. Installation
```shell
conda create -n logics-parsing-omni python=3.10
conda activate logics-parsing-omni

pip install -r requirements.txt
```

### 2. Inference
```shell
python inference.py --image_url PATH_TO_INPUT_IMG --audio_url PATH_TO_INPUT_AUDIO --text_prompt "What can you see and hear? Answer in one short sentence."
```

## Acknowledgments


We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)

