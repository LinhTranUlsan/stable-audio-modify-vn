****Tải các thư viện này trước (Run it to download the lib)  
pip install stable-audio-tools  
pip install .

download the model_config.json and model.safetensors at https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main  
after that, go to the "run_inference_midi10_15_4_branching3_backup.py" and change the path:  
config_path = r'add your model_config path'  
ckpt_path = r'add your model.safetensors path'  

**Kiểm tra xem có GPU hay không:  
Open Command Prompt (or PowerShell) and run:  
nvidia-smi

Hoặc kiểm tra bằng (Or write the syntax in Python):  
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print(torch.version.cuda)"  
***Nếu lỗi thì search chatgpt để fix cho nhanh.



Còn không lỗi thì chạy lệnh này để xem kết quả:  
python run_inference_midi10_15_4_branching3_backup.py

