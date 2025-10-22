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



# Run the code by following the instruction below:
# Generate all (branched + standalone + base)  
python run_inference.py  
# Skip standalone baseline generation  (optional)
python run_inference.py --no-standalone  
# Skip base reference (optional)
python run_inference.py --no-base  
# Use cached prefix  (optional)
python run_inference.py --use-cache  

# Evaluate and compare  
python evaluate_branching_results.py  

1. Fréchet Audio Distance (FAD) - Kilgour et al. 2019 [1]  
2. CLAP Score - Huang et al. 2023 [2]  
3. Mel Cepstral Distortion (MCD) - Kubichek 1993 [3]  
4. Multi-Scale Spectral Loss (MSS) - Défossez et al. 2023 [4]  
5. SNR and SI-SNR  
