# import sys
# import os
# import torch
# import torchaudio
# from datetime import datetime
# from stable_audio_tools.models.factory import create_model_from_config
# from stable_audio_tools.inference.generation import generate_diffusion_cond
# from einops import rearrange
# import json
# from safetensors.torch import load_file as safetensors_load
# import traceback

# # Set random seed for reproducibility
# torch.manual_seed(321)

# def generate_audio(model, sample_rate, sample_size, base_prompt, branch_prompt, output_path, num_steps=50, text_cfg_scale=4.0, duration=18):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Starting with device={device}, sample_rate={sample_rate}, sample_size={sample_size}, num_steps={num_steps}, cfg_scale={text_cfg_scale}, base_prompt={base_prompt}, branch_prompt={branch_prompt}")
#     try:
#         # Generate base audio for comparison
#         base_output_path = "output_base.wav"
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Generating base audio: {base_prompt}")
#         with torch.amp.autocast('cuda', dtype=torch.float16):
#             base_audio = generate_diffusion_cond(
#                 model,
#                 steps=num_steps,
#                 cfg_scale=text_cfg_scale,
#                 conditioning=[{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}],
#                 sample_size=sample_size,
#                 sigma_min=0.2,
#                 sigma_max=100,
#                 sampler_type="dpmpp-3m-sde",
#                 device=device,
#                 seed=32341278
#             )
#             max_abs = torch.max(torch.abs(base_audio))
#             if max_abs > 0 and not torch.isinf(max_abs):
#                 base_audio = base_audio / (max_abs + 1e-8)
#             base_audio = rearrange(base_audio, "b d n -> d (b n)").to(torch.float32)
#             base_waveform = base_audio.detach().cpu().numpy()
#             base_audio = torch.tensor(base_waveform, dtype=torch.float32) * 32767
#             base_audio = torch.clamp(base_audio, -32767, 32767).to(torch.int16)
#             torchaudio.save(base_output_path, base_audio, sample_rate)
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Saved base audio: {base_output_path}")


#         # Step 1: Generate prefix latent at step 25
#         prefix_steps = 150
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Generating prefix latent with {prefix_steps} steps")
#         conditioning = [{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}]
#         try:
#             cond_test = model.conditioner(conditioning, device=device)
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Prefix cond keys={list(cond_test.keys())}")
#             for key in cond_test:
#                 print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: {key} shape={[t.shape if isinstance(t, torch.Tensor) else None for t in cond_test[key]]}")
#         except Exception as e:
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [WARNING]: Prefix conditioning failed: {e}")
#             raise
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Calling generate_diffusion_cond with return_latents=True")
#         with torch.amp.autocast('cuda', dtype=torch.float16):
#             prefix_latent, final_sigma = generate_diffusion_cond(
#                 model,
#                 steps=prefix_steps,
#                 cfg_scale=text_cfg_scale,
#                 conditioning=conditioning,
#                 sample_size=sample_size,
#                 sigma_min=0.2,
#                 sigma_max=100,
#                 sampler_type="dpmpp-3m-sde",
#                 device=device,
#                 seed=32341278,
#                 return_latents=True
#             )
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Prefix latent shape={prefix_latent.shape}, mean={prefix_latent.mean().item():.6f}, std={prefix_latent.std().item():.6f}, final_sigma={final_sigma:.6f}")

#         # Step 2: Continue diffusion with new prompt
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Generating branch: {branch_prompt}")
#         branch_conditioning = [{"prompt": branch_prompt, "seconds_start": 0, "seconds_total": duration}]
#         try:
#             cond = model.conditioner(branch_conditioning, device=device)
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Branch cond keys={list(cond.keys())}")
#             for key in cond:
#                 print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: {key} shape={[t.shape if isinstance(t, torch.Tensor) else None for t in cond[key]]}")
#         except Exception as e:
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [WARNING]: Branch conditioning failed: {e}")
#             raise
#         with torch.amp.autocast('cuda', dtype=torch.float16):
#             audio = generate_diffusion_cond(
#                 model,
#                 steps=num_steps,
#                 cfg_scale=20,
#                 conditioning=branch_conditioning,
#                 sample_size=sample_size,
#                 sigma_min=0.2,
#                 sigma_max=final_sigma,
#                 sampler_type="dpmpp-3m-sde",
#                 device=device,
#                 seed=32341278,
#                 init_noise=prefix_latent,
#                 init_sigma=final_sigma,
#                 partial_steps=num_steps - prefix_steps
#             )
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Decoded branch shape={audio.shape}, mean={audio.mean().item():.6f}, std={audio.std().item():.6f}")

#         # Normalize and save
#         max_abs = torch.max(torch.abs(audio))
#         if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [WARNING]: Audio may be noisy: max_abs={max_abs:.6f}, has_nan={torch.isnan(audio).any()}, has_inf={torch.isinf(audio).any()}")
#         if max_abs > 0 and not torch.isinf(max_abs):
#             audio = audio / (max_abs + 1e-8)
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Normalized branch: min={audio.min().item():.6f}, max={audio.max().item():.6f}")

#         audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
#         waveform = audio.detach().cpu().numpy()
#         audio = torch.tensor(waveform, dtype=torch.float32) * 32767
#         audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
#         torchaudio.save(output_path, audio, sample_rate)
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Saved {output_path}")



#         return audio, sample_rate

#     except Exception as e:
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [ERROR]: Error: {e}")
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Stack trace: {traceback.format_exc()}")
#         raise

# if __name__ == "__main__":
#     # config_path = r'D:\stable-audio-tools\models\stable-audio-open-1.0\model_config.json'
#     # ckpt_path = r'D:\stable-audio-tools\models\stable-audio-open-1.0\model.safetensors'
#     config_path = r'stableaudiocode\model_config.json'
#     ckpt_path = r'stableaudiocode\model.safetensors'
#     try:
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Loading model config from {config_path}")
#         with open(config_path, 'r') as f:
#             model_config = json.load(f)
#         model = create_model_from_config(model_config)
#         model.model_config = model_config
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Loading checkpoint from {ckpt_path}")
#         state_dict = safetensors_load(ckpt_path, device='cpu')
#         model.load_state_dict(state_dict, strict=True)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model = model.to(device).half()
#         sample_rate = model_config.get("sample_rate", 44100)
#         sample_size = 44100*30
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Using sample_size={sample_size}")

#         base_prompt = "Format: Band | Subgenre: Post Rock | Instruments: echoing electric guitars with chorus, well recorded drum-kit, electric bass, occasional soaring harmonies | Moods: moving, epic, climactic | 125 BPM"
#         branch_prompt = "Genre: Rock | Subgenre: Pop Rock, Indie Rock | Instruments: piano | Moods: soft| Tempo: Medium"
#         output_path = "output_verse.wav"
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating song section")
#         output, sr = generate_audio(
#             model, sample_rate, sample_size, base_prompt, branch_prompt, output_path,
#             num_steps=200, text_cfg_scale=10.0, duration=30
#         )

#     except Exception as e:
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Main execution error: {e}")
#         print(f"[{datetime.now().strftime('%H:%M:%S')}] main [DEBUG]: Stack trace: {traceback.format_exc()}")

import sys
import os
import torch
import torchaudio
from datetime import datetime
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from einops import rearrange
import json
from safetensors.torch import load_file as safetensors_load
import traceback

# Set random seed for reproducibility
torch.manual_seed(321)

def generate_audio(model, sample_rate, sample_size, base_prompt, branch_prompt, output_path, num_steps=50, text_cfg_scale=4.0, duration=18):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Starting with device={device}, sample_rate={sample_rate}, sample_size={sample_size}, num_steps={num_steps}, cfg_scale={text_cfg_scale}, base_prompt={base_prompt}, branch_prompt={branch_prompt}")
    
    try:
        # STEP 0: Generate base audio for comparison
        # This is the full generation using only base_prompt
        base_output_path = "output_base.wav"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Generating base audio: {base_prompt}")
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            base_audio = generate_diffusion_cond(
                model,
                steps=num_steps,
                cfg_scale=text_cfg_scale,
                conditioning=[{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}],
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=100,
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=32341278
            )
            
            # Normalize audio
            max_abs = torch.max(torch.abs(base_audio))
            if max_abs > 0 and not torch.isinf(max_abs):
                base_audio = base_audio / (max_abs + 1e-8)
            
            # Rearrange and convert to int16 for saving
            base_audio = rearrange(base_audio, "b d n -> d (b n)").to(torch.float32)
            base_waveform = base_audio.detach().cpu().numpy()
            base_audio = torch.tensor(base_waveform, dtype=torch.float32) * 32767
            base_audio = torch.clamp(base_audio, -32767, 32767).to(torch.int16)
            torchaudio.save(base_output_path, base_audio, sample_rate)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Saved base audio: {base_output_path}")

        # STEP 1: Generate prefix latent
        # Run partial diffusion with base_prompt and save the latent
        prefix_steps = 150
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Generating prefix latent with {prefix_steps} steps")
        
        conditioning = [{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        # Debug: Test conditioning
        try:
            cond_test = model.conditioner(conditioning, device=device)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Prefix cond keys={list(cond_test.keys())}")
            for key in cond_test:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: {key} shape={[t.shape if isinstance(t, torch.Tensor) else None for t in cond_test[key]]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [WARNING]: Prefix conditioning failed: {e}")
            raise
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Calling generate_diffusion_cond with return_latents=True")
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # CRITICAL: return_latents=True makes the function return (latent, sigma) instead of audio
            prefix_latent, final_sigma = generate_diffusion_cond(
                model,
                steps=prefix_steps,
                cfg_scale=text_cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=100,
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=32341278,
                return_latents=True  # Returns (latent, sigma) tuple
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Prefix latent shape={prefix_latent.shape}, mean={prefix_latent.mean().item():.6f}, std={prefix_latent.std().item():.6f}, final_sigma={final_sigma:.6f}")

        # STEP 2: Continue diffusion with branch prompt
        # Use the prefix_latent as starting point with new conditioning
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Generating branch: {branch_prompt}")
        
        branch_conditioning = [{"prompt": branch_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        # Debug: Test branch conditioning
        try:
            cond = model.conditioner(branch_conditioning, device=device)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Branch cond keys={list(cond.keys())}")
            for key in cond:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: {key} shape={[t.shape if isinstance(t, torch.Tensor) else None for t in cond[key]]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [WARNING]: Branch conditioning failed: {e}")
            raise
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # CRITICAL: 
            # - steps=prefix_steps: Run the SAME number of steps as prefix (not remaining steps)
            # - init_noise=prefix_latent: Start from the saved latent
            # - init_sigma=final_sigma: Use the sigma value from where prefix stopped
            # - sigma_max=final_sigma: Override sigma_max to continue from saved point
            audio = generate_diffusion_cond(
                model,
                steps=prefix_steps,
                cfg_scale=20,
                conditioning=branch_conditioning,
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=final_sigma,
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=32341278,
                init_noise=prefix_latent, 
                init_sigma=final_sigma
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Decoded branch shape={audio.shape}, mean={audio.mean().item():.6f}, std={audio.std().item():.6f}")

        # STEP 3: Normalize and save branch audio
        
        # Check for audio issues
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [WARNING]: Audio may be noisy: max_abs={max_abs:.6f}, has_nan={torch.isnan(audio).any()}, has_inf={torch.isinf(audio).any()}")
        
        # Normalize
        if max_abs > 0 and not torch.isinf(max_abs):
            audio = audio / (max_abs + 1e-8)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Normalized branch: min={audio.min().item():.6f}, max={audio.max().item():.6f}")

        # Rearrange and convert to int16 for saving
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        waveform = audio.detach().cpu().numpy()
        audio = torch.tensor(waveform, dtype=torch.float32) * 32767
        audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
        torchaudio.save(output_path, audio, sample_rate)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [INFO]: Saved {output_path}")

        return audio, sample_rate

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_audio [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # config_path = r'D:\stable-audio-tools\models\stable-audio-open-1.0\model_config.json'
    # ckpt_path = r'D:\stable-audio-tools\models\stable-audio-open-1.0\model.safetensors'
    config_path = r'stableaudiocode\model_config.json'
    ckpt_path = r'stableaudiocode\model.safetensors'
    
    try:
        # Load model
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Loading model config from {config_path}")
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        model = create_model_from_config(model_config)
        model.model_config = model_config
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Loading checkpoint from {ckpt_path}")
        state_dict = safetensors_load(ckpt_path, device='cpu')
        model.load_state_dict(state_dict, strict=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).half()
        
        sample_rate = model_config.get("sample_rate", 44100)
        sample_size = 44100*30  # 30 seconds
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Using sample_size={sample_size}")

        # Base prompt: Defines the musical structure (rhythm, harmony, tempo)
        base_prompt = "Format: Band | Subgenre: Post Rock | Instruments: echoing electric guitars with chorus, well recorded drum-kit, electric bass, occasional soaring harmonies | Moods: moving, epic, climactic | 125 BPM"
        
        # Branch prompt: Defines the variation (different instruments, mood)
        branch_prompt = "Genre: Indie | Subgenre: Pop Rock, Indie Rock | Instruments: piano, drum machine, organ | Moods: soft, happy| Tempo: Medium"
        
        output_path = "output_verse.wav"
        
        # Generate audio
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating song section")
        
        output, sr = generate_audio(
            model, 
            sample_rate, 
            sample_size, 
            base_prompt, 
            branch_prompt, 
            output_path,
            num_steps=200,  # Total steps for full generation
            text_cfg_scale=10.0,  # CFG scale
            duration=30  # Duration in seconds
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generation complete!")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Output files:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]:   - output_base.wav (full generation with base prompt)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]:   - output_verse.wav (branched generation)")

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Main execution error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [DEBUG]: Stack trace: {traceback.format_exc()}")