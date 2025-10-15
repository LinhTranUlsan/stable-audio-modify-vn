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

def generate_base_audio(model, sample_rate, sample_size, base_prompt, output_path, num_steps=200, text_cfg_scale=10.0, duration=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [INFO]: Generating base audio with {num_steps} steps")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [INFO]: Prompt: {base_prompt}")
    
    try:
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
            torchaudio.save(output_path, base_audio, sample_rate)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [INFO]: Saved base audio: {output_path}")
        return base_audio, sample_rate
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


def method1_remaining_steps(model, sample_rate, sample_size, base_prompt, branch_prompt, output_path, 
                            prefix_steps=150, remaining_steps=50, text_cfg_scale=10.0, branch_cfg_scale=20.0, duration=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: METHOD 1: Remaining Steps ({prefix_steps} + {remaining_steps})")

    
    try:
        # Step 1: Generate prefix latent
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Step 1: Generating prefix latent ({prefix_steps} steps)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Prefix prompt: {base_prompt}")
        
        conditioning = [{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
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
                return_latents=True
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Prefix latent generated")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Latent shape={prefix_latent.shape}, final_sigma={final_sigma:.6f}")
        
        # Calculate actual final sigma for remaining steps
        # For 150/200 steps with Karras schedule, sigma should be around 1.5
        # We'll use a calculated value instead of the returned 100
        steps_ratio = prefix_steps / 200.0
        calculated_sigma = 0.2 + (100 - 0.2) * ((1 - steps_ratio) ** 3)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Calculated sigma for continuation: {calculated_sigma:.6f}")
        
        # Step 2: Continue with remaining steps using new prompt
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Step 2: Continuing with remaining {remaining_steps} steps")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Branch prompt: {branch_prompt}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Sigma range: {calculated_sigma:.2f} → 0.2")
        
        branch_conditioning = [{"prompt": branch_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            audio = generate_diffusion_cond(
                model,
                steps=remaining_steps,  # Only remaining steps (50)
                cfg_scale=branch_cfg_scale,
                conditioning=branch_conditioning,
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=calculated_sigma,  # Start from where prefix stopped
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=32341278,
                init_noise=prefix_latent,
                init_sigma=calculated_sigma
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Branch audio decoded, shape={audio.shape}")
        
        # Normalize and save
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 0 and not torch.isinf(max_abs):
            audio = audio / (max_abs + 1e-8)
        
        if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [WARNING]: Audio may have issues: max_abs={max_abs:.6f}")
        
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        waveform = audio.detach().cpu().numpy()
        audio = torch.tensor(waveform, dtype=torch.float32) * 32767
        audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
        torchaudio.save(output_path, audio, sample_rate)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Saved: {output_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [INFO]: Total steps used: {prefix_steps + remaining_steps}")
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method1_remaining_steps [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


def method2_restart_steps(model, sample_rate, sample_size, base_prompt, branch_prompt, output_path, 
                          prefix_steps=150, branch_steps=150, text_cfg_scale=10.0, branch_cfg_scale=20.0, duration=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: METHOD 2: Restart Steps ({prefix_steps} + {branch_steps})")
    
    try:
        # Step 1: Generate prefix latent
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Step 1: Generating prefix latent ({prefix_steps} steps)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Prefix prompt: {base_prompt}")
        
        conditioning = [{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
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
                return_latents=True
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Prefix latent generated")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Latent shape={prefix_latent.shape}, final_sigma={final_sigma:.6f}")
        
        # Step 2: Restart diffusion with new prompt (FULL branch_steps)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Step 2: Restarting with {branch_steps} steps")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Branch prompt: {branch_prompt}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Sigma range: 100 → 0.2 (RESTART from high sigma)")
        
        branch_conditioning = [{"prompt": branch_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            audio = generate_diffusion_cond(
                model,
                steps=branch_steps,  # Full steps (150), not remaining
                cfg_scale=branch_cfg_scale,
                conditioning=branch_conditioning,
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=final_sigma,  # Use final_sigma from prefix
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=32341278,
                init_noise=prefix_latent,  # But initialize from prefix
                init_sigma=final_sigma
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Branch audio decoded, shape={audio.shape}")
        
        # Normalize and save
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 0 and not torch.isinf(max_abs):
            audio = audio / (max_abs + 1e-8)
        
        if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [WARNING]: Audio may have issues: max_abs={max_abs:.6f}")
        
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        waveform = audio.detach().cpu().numpy()
        audio = torch.tensor(waveform, dtype=torch.float32) * 32767
        audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
        torchaudio.save(output_path, audio, sample_rate)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Saved: {output_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [INFO]: Total steps used: {prefix_steps + branch_steps}")
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method2_restart_steps [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


def method3_hybrid_steps(model, sample_rate, sample_size, base_prompt, branch_prompt, output_path, 
                         prefix_steps=150, remaining_steps=100, text_cfg_scale=10.0, branch_cfg_scale=20.0, duration=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: METHOD 3: Hybrid Approach ({prefix_steps} + {remaining_steps})")
    
    try:
        # Step 1: Generate prefix latent
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Step 1: Generating prefix latent ({prefix_steps} steps)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Prefix prompt: {base_prompt}")
        
        conditioning = [{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
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
                return_latents=True
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Prefix latent generated")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Latent shape={prefix_latent.shape}, final_sigma={final_sigma:.6f}")
        
        # Calculate sigma for hybrid approach
        steps_ratio = prefix_steps / 200.0
        calculated_sigma = 0.2 + (100 - 0.2) * ((1 - steps_ratio) ** 3)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Calculated sigma for continuation: {calculated_sigma:.6f}")
        
        # Step 2: Continue with moderate remaining steps
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Step 2: Continuing with {remaining_steps} steps (MODERATE)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Branch prompt: {branch_prompt}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Sigma range: {calculated_sigma:.2f} → 0.2")
        
        branch_conditioning = [{"prompt": branch_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            audio = generate_diffusion_cond(
                model,
                steps=remaining_steps,  # Moderate steps (100)
                cfg_scale=branch_cfg_scale,
                conditioning=branch_conditioning,
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=calculated_sigma,
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=32341278,
                init_noise=prefix_latent,
                init_sigma=calculated_sigma
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Branch audio decoded, shape={audio.shape}")
        
        # Normalize and save
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 0 and not torch.isinf(max_abs):
            audio = audio / (max_abs + 1e-8)
        
        if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [WARNING]: Audio may have issues: max_abs={max_abs:.6f}")
        
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        waveform = audio.detach().cpu().numpy()
        audio = torch.tensor(waveform, dtype=torch.float32) * 32767
        audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
        torchaudio.save(output_path, audio, sample_rate)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Saved: {output_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [INFO]: Total steps used: {prefix_steps + remaining_steps}")
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] method3_hybrid_steps [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # Configuration paths
    config_path = r'stableaudiocode\model_config.json'
    ckpt_path = r'stableaudiocode\model.safetensors'
    
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: BRANCHING METHODS COMPARISON")
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
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Model loaded successfully")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Sample rate: {sample_rate}, Sample size: {sample_size}")

        # Base prompt: Post rock with electric guitars
        base_prompt = "Format: Band | Subgenre: Post Rock | Instruments: echoing electric guitars with chorus, well recorded drum-kit, electric bass, occasional soaring harmonies | Moods: moving, epic, climactic | 125 BPM"
        
        # Branch prompt: Indie pop with piano (VERY DIFFERENT from base)
        branch_prompt = "Genre: Indie | Subgenre: Pop Rock, Indie Rock | Instruments: piano, drum machine, organ | Moods: soft, happy | Tempo: Medium"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: PROMPTS")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Base: {base_prompt}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Branch: {branch_prompt}")
        
        # COMPARISON 1: Generate base audio (reference)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: GENERATING BASE AUDIO (Reference)")
        
        generate_base_audio(
            model,
            sample_rate,
            sample_size,
            base_prompt,
            "comparison_base.wav",
            num_steps=200,
            text_cfg_scale=10.0,
            duration=30
        )
        
        # COMPARISON 2: Method 1 - Remaining 50 steps (150 + 50)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: TESTING METHOD 1: Remaining 50 Steps")
        
        method1_remaining_steps(
            model,
            sample_rate,
            sample_size,
            base_prompt,
            branch_prompt,
            "comparison_method1_remaining50.wav",
            prefix_steps=150,
            remaining_steps=50,
            text_cfg_scale=10.0,
            branch_cfg_scale=20.0,
            duration=30
        )
        
        # COMPARISON 3: Method 2 - Restart 150 steps (150 + 150)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: TESTING METHOD 2: Restart 150 Steps")
        
        method2_restart_steps(
            model,
            sample_rate,
            sample_size,
            base_prompt,
            branch_prompt,
            "comparison_method2_restart150.wav",
            prefix_steps=150,
            branch_steps=150,
            text_cfg_scale=10.0,
            branch_cfg_scale=20.0,
            duration=30
        )
        
        # COMPARISON 4: Method 3 - Hybrid 100 steps (150 + 100)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: TESTING METHOD 3: Hybrid 100 Steps")
        
        method3_hybrid_steps(
            model,
            sample_rate,
            sample_size,
            base_prompt,
            branch_prompt,
            "comparison_method3_hybrid100.wav",
            prefix_steps=150,
            remaining_steps=100,
            text_cfg_scale=10.0,
            branch_cfg_scale=20.0,
            duration=30
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: COMPARISON COMPLETE!")
       
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Main execution error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [DEBUG]: Stack trace: {traceback.format_exc()}")
