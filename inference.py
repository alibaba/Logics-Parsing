import argparse
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

def run_inference(model_path, image_url=None, audio_url=None, text_prompt=None, use_audio_in_video=True, output_audio_path="output.wav"):
    # Load model
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    # Compose conversation based on passed parameters
    content_list = []
    if image_url:
        content_list.append({"type": "image", "image": image_url})
    if audio_url:
        content_list.append({"type": "audio", "audio": audio_url})
    if text_prompt:
        content_list.append({"type": "text", "text": text_prompt})

    conversation = [{"role": "user", "content": content_list}]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)

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

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(
        **inputs, 
        speaker="Ethan", 
        thinker_return_dict_in_generate=True,
        use_audio_in_video=use_audio_in_video
    )

    text_result = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print("Generated text:", text_result)

    if audio is not None:
        sf.write(
            output_audio_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Audio saved to {output_audio_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OURMODEL multimodal inference script.")
    
    parser.add_argument("--model_path", type=str, default="Logics-MLLM/Logics-Parsing-Caption", help="Path or name of the pretrained model.")
    parser.add_argument("--image_url", type=str, help="Image URL or local path.")
    parser.add_argument("--audio_url", type=str, help="Audio URL or local path.")
    parser.add_argument("--text_prompt", type=str, help="Text prompt for the model.")
    parser.add_argument("--use_audio_in_video", action="store_true", help="Enable audio usage in video inputs.")
    parser.add_argument("--output_audio_path", type=str, default="output.wav", help="Path to save generated audio.")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        image_url=args.image_url,
        audio_url=args.audio_url,
        text_prompt=args.text_prompt,
        use_audio_in_video=args.use_audio_in_video,
        output_audio_path=args.output_audio_path
    )
