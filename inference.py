import torch
import os 
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image,ImageFont,ImageDraw
import json 
import re
import math 
import cv2 
import argparse
import time 


def inference(img_url, prompt="QwenVL HTML"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_url,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt")
    inputs = inputs.to(model.device)
    #3.模型推理
    generated_ids = model.generate(**inputs, max_new_tokens=16384, temperature=0.1, top_p=0.5, repetition_penalty=1.05)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    #输出为html格式
    html_output = output_text[0]
    return html_output


def remove_lines_starting_with(text):
    lines = text.splitlines(keepends=True)
    
    filtered = []
    prefixes_to_remove = ('Z:')
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped.strip():
            continue
        if stripped.startswith(prefixes_to_remove):
            continue

        filtered.append(line)  
    return "".join(filtered)

def process_code_content(content: str) -> str:
    content = content.replace('```', '')
    
    content = re.sub(r'^\s*<pre[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</pre>\s*$', '', content, flags=re.IGNORECASE)
    
    content = re.sub(r'^\s*<code[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</code>\s*$', '', content, flags=re.IGNORECASE)
    
    # 包裹在 ``` 中
    return f"```code\n{content.strip()}\n```"


def process_pseudocode_content(content: str) -> str:
    """Process pseudocode content, preserving indentation and not breaking LaTeX formulas"""

    content = content.replace('```', '')
    content = re.sub(r'^\s*<(pre|code)[^>]*>', '', content, flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r'</(pre|code)>\s*$', '', content, flags=re.IGNORECASE | re.MULTILINE)

    # Extract and protect LaTeX formulas
    math_blocks = []
    def save_math(match):
        placeholder = f"___MATH_ID_{len(math_blocks)}___"
        math_blocks.append(match.group(0))
        return placeholder

    # Regex: prioritize matching double dollar signs, then single dollar signs
    math_pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    protected_content = re.sub(math_pattern, save_math, content, flags=re.DOTALL)

    protected_content = protected_content.replace(' ', '&nbsp;')
    protected_content = protected_content.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    protected_content = protected_content.replace('\n', '<br>')

    # Restore LaTeX formulas
    final_content = protected_content
    for i, original_math in enumerate(math_blocks):
        placeholder = f"___MATH_ID_{i}___"
        final_content = final_content.replace(placeholder, original_math)

    return f"___\n<br>{final_content.strip()}<br>\n___"


def qwenvl_cast_html_tag(input_text: str) -> str:
    output = input_text
    IMG_RE = re.compile(
        r'<img\b[^>]*\bdata-bbox\s*=\s*"?\d+,\d+,\d+,\d+"?[^>]*\/?>',
        flags=re.IGNORECASE,
    )
    output = IMG_RE.sub('', output)

    # code
    def replace_code(match):
        content = match.group(1)
        processed_content = process_code_content(content)
        return f"\n{processed_content}\n"
    
    code_pattern = re.compile(
        r'<div\b[^>]*class="code"[^>]*>(.*?)</div>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    output = code_pattern.sub(replace_code, output)
    
    # pseudocode
    def replace_pseudocode(match):
        content = match.group(1)
        processed_content = process_pseudocode_content(content)
        return f"\n{processed_content}\n"
    
    pseudocode_pattern = re.compile(
        r'<div\b[^>]*class="pseudocode"[^>]*>(.*?)</div>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    output = pseudocode_pattern.sub(replace_pseudocode, output)

    # <div>
    def strip_div(class_name: str, txt: str) -> str:
        if class_name in ['code', 'pseudocode']:
            return txt

        def replace_func(match):
            content = match.group(1)
            
            # 仅针对匹配到的 div 内部内容进行清洗
            if class_name == 'chart':
                content = re.sub(r'^\s*(click\s+|style\s+|linkStyle\s+|stroke|classDef\s+|class\s+)\b.*\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
                content = re.sub(r'^\s*(?:%%|::icon).*\n?', '', content, flags=re.MULTILINE)

                content = content.strip()
                if content.startswith('mermaid'):
                    content = '```' + content
                elif re.match(r'^```\s*mermaid', content):
                    pass
                else:
                    content = '```mermaid\n' + content
                if not content.endswith('```'):
                    content += '\n```'

            if class_name == 'music':
                content = remove_lines_starting_with(content)

                content = content.strip()
                if content.startswith('abc'):
                    content = '```' + content
                elif re.match(r'^```\s*abc', content):
                    pass
                else:
                    content = '```abc\n' + content
                if not content.endswith('```'):
                    content += '\n```'
                
            return f"\n{content}\n"

        pattern = re.compile(
            rf'\s*<div\b[^>]*class="{class_name}"[^>]*>(.*?)</div>\s*',
            flags=re.DOTALL | re.IGNORECASE,
        )
        return pattern.sub(replace_func, txt)

    other_classes = ['image', 'chemistry', 'table', 'formula', 'image caption', 'table caption']
    # other_classes = ['image', 'chemistry', 'table', 'formula', 'image caption', 'table caption', 'music', 'chart']
    for cls in other_classes:
        output = strip_div(cls, output)
    
    # <p>
    output = re.sub(
        r'<p\b[^>]*>(.*?)</p>',
        r'\1\n',
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    
    output = output.replace(" </td>", "</td>")
    return output


def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 3136, max_pixels: int = 7200*32*32
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def plot_bbox(img_path, pred, output_path):
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    # scale = (img_width / input_width, img_height / input_height)
    bboxes = []

    pattern = re.compile(r'data-bbox="(\d+),(\d+),(\d+),(\d+)"')

    # scale_x, scale_y = scale  

    def replace_bbox(match):
        x1, y1, x2, y2 = map(int, match)
        x1 = int(x1/1000 * img_width)
        y1 = int(y1/1000 * img_height)
        x2 = int(x2/1000 * img_width)
        y2 = int(y2/1000 * img_height)

        bboxes.append([x1,y1,x2,y2])

    
    matches = re.findall(pattern, pred)
    if matches:
        for match in matches:
            # print(match)
            replace_bbox(match)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)

    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logics-Parsing for document parsing and visualize the output.")

    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the directory containing the pre-trained model and processor.")
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to the input image file for parsing.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the prediction.")
    parser.add_argument("--prompt", type=str, default="QwenVL HTML", 
                        help="The prompt to send to the model. (default: %(default)s)")


    args = parser.parse_args()
    

    model_path = args.model_path
    image_path = args.image_path
    prompt = args.prompt
    output_path =  args.output_path



    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    

    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.max_pixels = 7200 * 32 * 32
    processor.image_processor.min_pixels = 3136

    raw_output  = inference(image_path, prompt)
    output_img_path = output_path + "_vis.png"
    output_raw_path = output_path + "_raw.mmd"
    output_mmd_path = output_path + ".mmd"
    plot_bbox(image_path, raw_output, output_img_path)
    
    with open(output_raw_path, 'w') as f:
        f.write(raw_output)
    
    markdown_output = qwenvl_cast_html_tag(raw_output)
    with open(output_mmd_path, 'w') as f:
        f.write(markdown_output)

    print(raw_output)
    


