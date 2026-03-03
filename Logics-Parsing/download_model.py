from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="modelscope")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_dir = os.path.join(script_dir, "weights/Logics-Parsing-2.0")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.type == "huggingface":
        from huggingface_hub import snapshot_download
        model_name = "Logics-MLLM/Logics-Parsing-2.0"
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    elif args.type == "modelscope":
        from modelscope import snapshot_download
        model_name = "Alibaba-DT/Logics-Parsing-v2.0"
        snapshot_download(repo_id=model_name, local_dir=model_dir)
    else:
        raise ValueError(f"Invalid type: {args.type}")
    
    print(f"model downloaded to {model_dir}")


    