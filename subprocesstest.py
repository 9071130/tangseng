import subprocess
# epochs='5'
subprocess.Popen([
    "python",
    "ModelTrainingServe.py",
    "--train_data_path", "output_text_data/formatted_data.json",
    "--book_name", "见春天",
    "--role_name", "魏清越",
    "--spliced_data_save_path", "output_text_data/spliced_data.json",
    "--pretrained_model_path", "Qwen/Qwen1.5-0.5B",
    "--upload_path_safetensors", "LiuShisan123/testlocal_safetensors",
    "--upload_path_gguf", "LiuShisan123/testlocal_gguf"
])