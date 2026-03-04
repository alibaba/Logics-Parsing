import argparse
import torch
from typing import List, Optional
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# =====================================================================
# Pre-defined Task Examples
# =====================================================================
TASK_EXAMPLES = {
    # ------------------ Single Image Modality ------------------
    "document_structure_parsing": {
        "image_paths": ["assets/document.jpg"],
        "text_prompt": {
            "en": "Output the parsing results of this document in JSON format.",
            "ch": "请以JSON格式输出该文档的结构化解析结果。"
        }
    },
    "document_structure_and_semantic_parsing": {
        "image_paths": ["assets/document_semantic.jpg"],
        "text_prompt": {
            "en": "Output the parsing results of this document in JSON format. Include descriptions for illustrations, structurally parse natural images and graphics, and add a global overview at the end. Use the same language as the document text.",
            "ch": "请以JSON格式输出该文档的解析结果。包含对插图的描述，对自然图像和图形进行结构化解析，并在最后添加全局概述。请使用与文档文本相同的语言。"
        }
    },
    "natural_image_parsing": {
        "image_paths": ["assets/natural_image.jpg"],
        "text_prompt": {
            "en": "Please detect text and entities in the image, extract structured information such as bounding boxes, labels, attributes, and detailed descriptions, and provide a global image description. Output the results in JSON format.",
            "ch": "请检测图中的文本与实体，提取边界框、标签、属性及详细描述等结构化信息，并给出全局图像描述。结果以JSON格式输出。"
        }
    },
    "chart_image_parsing": {
        "image_paths": ["assets/chart_image.jpg"],
        "text_prompt": {
            "en": "Perform an in-depth parsing of the image, locate text and charts, extract their bounding boxes, labels, parsing results, and descriptions, and provide a global image description. Please present the results in JSON format.",
            "ch": "对图片进行深度解析，定位文本和图表，提取其边界框、标签、解析结果与描述，并给出全局图像描述，请用JSON格式呈现。"
        }
    },
    "geometric_image_parsing": {
        "image_paths": ["assets/geometric_image.jpg"],
        "text_prompt": {
            "en": "Please detect the text and geometric shapes in the image, extract bounding boxes, labels, parsing results, and detailed descriptions, and provide a global image description. Output the results in JSON format.",
            "ch": "请检测图中的文本和几何形状，提取边界框、标签、解析结果及详细描述，并提供全局图像描述。结果以JSON格式输出。"
        }
    },

    # ------------------ Audio Modality ------------------
    "audio_parsing": {
        "audio_path": "https://example.com/assets/sample_audio.wav",
        "text_prompt": {
            "en": "Divide the audio into continuous segments primarily based on speaker and VAD (split non-speech parts by audio classification); segments should include timestamps, classification labels, ASR, and speaker IDs, with a global description added at the end, output in JSON format.",
            "ch": "主要基于说话人和VAD将音频划分为连续的片段（通过音频分类分割非语音部分）；片段应包含时间戳、分类标签、ASR和说话人ID，最后添加全局描述，以JSON格式输出。"
        }
    },

    # ------------------ Video Modality ------------------
    "natural_video_parsing": {
        "video_path": "https://example.com/assets/natural_video.mp4",
        "use_audio_in_video": True,
        "text_prompt": {
            "en": "Split the video into continuous time segments based on visual semantic changes; for each segment, extract timestamps, internal audio split points and classification labels (following the principle of prioritizing human voice VAD, and classifying non-vocal parts by audio type) and video attributes. Finally, integrate a global audio-visual description, ASR (including speaker distinction), and language information. Please output in JSON format.",
            "ch": "基于视觉语义变化将视频划分为连续的时间片段；为每个片段提取时间戳、内部音频分割点和分类标签（遵循优先识别人声VAD的原则，并按音频类型对非人声部分进行分类）以及视频属性。最后，整合全局视听描述、ASR（包括说话人区分）和语言信息。请以JSON格式输出。"
        }
    },
    "camera_aware_video_parsing": {
        "video_path": "https://example.com/assets/camera_video.mp4",
        "use_audio_in_video": True,
        "text_prompt": {
            "en": "Describe the video content and its camera movement characteristics, while extracting the timestamps and camera movement labels for the visual segments, outputting in JSON format.",
            "ch": "描述视频内容并说明其运镜特点，同时提取视觉片段的时间戳与运镜标签，以JSON格式输出。"
        }
    },
    "text_rich_video_parsing": {
        "video_path": "https://example.com/assets/text_rich_video.mp4",
        "use_audio_in_video": True,
        "text_prompt": {
            "en": "Please analyze the video using OCR information stability as the basis for segmentation, extract the timestamp, OCR, and ASR content of each segment in chronological order, add a global audio-video description at the end, and output the result in JSON format.",
            "ch": "请以OCR信息的稳定性作为分割依据对视频进行分析，按时间顺序提取每个片段的时间戳、OCR和ASR内容，在最后添加全局音视频描述，并以JSON格式输出结果。"
        }
    },
    "text_rich_video_in_depth_caption": {
        "video_path": "https://example.com/assets/course_video.mp4",
        "use_audio_in_video": True,
        "text_prompt": {
            "en": "Based on the input course video, generate a course description report that is clearly structured, detailed in content, and easy for learners to read.",
            "ch": "根据输入的课程视频，生成一份结构清晰、内容详尽、易于学习者阅读的课程描述报告。"
        }
    },

    # ------------------ Multi-Image Modality ------------------
    "natural_image_diff_parsing": {
        "image_paths": ["assets/natural_img_before.jpg", "assets/natural_img_after.jpg"],
        "text_prompt": {
            "en": "Generate the structured parsing result of the editing from the first image to the second. List all changed elements item by item, providing their corresponding bounding boxes, labels, attributes, and descriptions; finally, provide a global editing description summarizing the overall changes. Output in JSON format.",
            "ch": "生成从第一张图编辑到第二张图的结构化解析结果。逐项列出所有变化元素，并给出对应的边界框、标签、属性及描述等信息；最后给出全局编辑描述总结整体变化。以JSON格式输出。"
        }
    },
    "geometric_diff_parsing": {
        "image_paths": ["assets/geo_img_before.jpg", "assets/geo_img_after.jpg"],
        "text_prompt": {
            "en": "Generate the geometric editing parsing result from the first image to the second. The content must include a structured parsing of all changed geometric elements, their geometric and quantitative relationships, and provide a global editing instruction summarizing the overall changes. Output in JSON format.",
            "ch": "生成从第一张图到第二张图的几何编辑解析结果。内容需包含所有变化几何元素的结构化解析、几何与定量关系，并给出总结整体变化的全局编辑指令。以JSON格式输出。"
        }
    }
}


def load_model(model_path: str):
    print(f"========== Loading Model: {model_path} ==========")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor

def run_inference(
    model, 
    processor,
    video_path: Optional[str] = None,          
    image_paths: Optional[List[str]] = None,   
    audio_path: Optional[str] = None,          
    text_prompt: Optional[str] = None, 
    use_audio_in_video: bool = False            
):
    content_list = []
    
    if video_path:
        content_list.append({"type": "video", "video": video_path})
    if image_paths:
        for img in image_paths:
            content_list.append({"type": "image", "image": img})
    if audio_path:
        content_list.append({"type": "audio", "audio": audio_path})
    if text_prompt:
        content_list.append({"type": "text", "text": text_prompt})

    if not content_list:
        raise ValueError("You must provide at least one input (video, image, audio, or text).")

    conversation = [{"role": "user", "content": content_list}]
    conversations = [conversation]

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video 
    )
    
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids, _ = model.generate(
            **inputs, 
            use_audio_in_video=use_audio_in_video, 
            thinker_return_dict_in_generate=True,
            return_audio=False,                    
            thinker_max_new_tokens=8192
        )

    text_result = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return text_result[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Omni Inference Script")
    parser.add_argument("--model_path", type=str, default="Logics-MLLM/Logics-Parsing-Omni", help="Path to model")
    
    task_choices = list(TASK_EXAMPLES.keys()) + ["custom"]
    parser.add_argument("--task", type=str, required=True, choices=task_choices, 
                        help="Choose a pre-defined task to run, or 'custom' to use CLI args.")
    
    parser.add_argument("--language", type=str, choices=["en", "ch"], default="en",
                        help="Choose the language for the predefined task prompt (en for English, ch for Chinese).")

    parser.add_argument("--video_path", type=str, help="Custom single video path")
    parser.add_argument("--image_paths", type=str, nargs='*', help="Custom image paths")
    parser.add_argument("--audio_path", type=str, help="Custom audio path")
    parser.add_argument("--text_prompt", type=str, help="Custom text prompt")
    parser.add_argument("--use_audio_in_video", action="store_true", help="Enable audio in video")
    
    args = parser.parse_args()
    
    model, processor = load_model(args.model_path)
    
    if args.task == "custom":
        print("\n========== Running Custom CLI Task ==========")
        result = run_inference(
            model=model, processor=processor,
            video_path=args.video_path,
            image_paths=args.image_paths,
            audio_path=args.audio_path,
            text_prompt=args.text_prompt, 
            use_audio_in_video=args.use_audio_in_video
        )
        print(f"Output:\n{result}\n=============================================")
        
    else:
        task_data = TASK_EXAMPLES[args.task]
        task_args = task_data.copy()
        selected_prompt = task_data["text_prompt"][args.language]
        task_args["text_prompt"] = selected_prompt
        
        print(f"\n" + "="*50)
        print(f"Running Task: {args.task} (Language: {args.language})")
        print(f"Inputs: {task_args}")
        print("="*50)
        
        try:
            result = run_inference(model=model, processor=processor, **task_args)
            print(f"\nOutput:\n{result}\n")
        except Exception as e:
            print(f"\nError running {args.task}: {e}\n(Ensure the assets/ files exist or update the paths)")
