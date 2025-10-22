
import sys
import os
import torch
import torchaudio
from datetime import datetime
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from einops import rearrange
import json
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
import traceback
from pathlib import Path
import argparse

torch.manual_seed(321)

class PrefixLatentCache:
    def __init__(self, cache_dir="latent_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def save_prefix(self, prefix_latent, final_sigma, prompt, steps, cfg_scale, seed, duration, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"prefix_{timestamp}.safetensors"
        
        filepath = self.cache_dir / filename
        
        metadata = {
            'final_sigma': str(final_sigma),
            'prompt': prompt,
            'steps': str(steps),
            'cfg_scale': str(cfg_scale),
            'seed': str(seed),
            'duration': str(duration),
            'shape': str(list(prefix_latent.shape)),
            'dtype': str(prefix_latent.dtype),
            'timestamp': datetime.now().isoformat()
        }
        
        tensors = {'prefix_latent': prefix_latent.cpu()}
        safetensors_save(tensors, filepath, metadata=metadata)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Saved prefix to {filepath}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Metadata: {metadata}")
        
        return str(filepath)
    
    def load_prefix(self, filename):
        filepath = self.cache_dir / filename if not Path(filename).is_absolute() else Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Prefix file not found: {filepath}")
        
        from safetensors import safe_open
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            prefix_latent = f.get_tensor('prefix_latent')
            metadata = f.metadata()
        
        final_sigma = float(metadata['final_sigma'])
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Loaded prefix from {filepath}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Shape={prefix_latent.shape}, final_sigma={final_sigma}")
        
        return prefix_latent, final_sigma, metadata
    
    def list_cached_prefixes(self):
        files = list(self.cache_dir.glob("prefix_*.safetensors"))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Found {len(files)} cached prefixes:")
        
        for filepath in files:
            from safetensors import safe_open
            with safe_open(filepath, framework="pt", device="cpu") as f:
                metadata = f.metadata()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: {filepath.name}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Prompt: {metadata.get('prompt', 'N/A')[:80]}...")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Steps: {metadata.get('steps')}, Sigma: {metadata.get('final_sigma')}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Duration: {metadata.get('duration', 'N/A')}s")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PrefixCache [INFO]: Created: {metadata.get('timestamp', 'N/A')}")
        
        return files


def generate_base_audio(model, sample_rate, sample_size, base_prompt, output_path, num_steps=50, text_cfg_scale=4.0, duration=18, device='cuda'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [INFO]: Starting generation")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [INFO]: Device={device}, sample_rate={sample_rate}, sample_size={sample_size}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_base_audio [INFO]: num_steps={num_steps}, cfg_scale={text_cfg_scale}")
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
            
            max_abs = torch.max(torch.abs(base_audio))
            if max_abs > 0 and not torch.isinf(max_abs):
                base_audio = base_audio / (max_abs + 1e-8)
            
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


def generate_standalone_audio(model, sample_rate, sample_size, prompt, output_path, num_steps=150, text_cfg_scale=10.0, duration=30, seed=32341278, device='cuda'):
    """
    Generate standalone audio directly from prompt without branching diffusion.
    Used as baseline comparison for branching approach.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [INFO]: Starting standalone generation")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [INFO]: Device={device}, sample_rate={sample_rate}, sample_size={sample_size}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [INFO]: num_steps={num_steps}, cfg_scale={text_cfg_scale}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [INFO]: Prompt: {prompt}")
    
    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            audio = generate_diffusion_cond(
                model,
                steps=num_steps,
                cfg_scale=text_cfg_scale,
                conditioning=[{"prompt": prompt, "seconds_start": 0, "seconds_total": duration}],
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=100,
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=seed
            )
            
            max_abs = torch.max(torch.abs(audio))
            if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [WARNING]: Audio may be noisy: max_abs={max_abs:.6f}, has_nan={torch.isnan(audio).any()}, has_inf={torch.isinf(audio).any()}")
            
            if max_abs > 0 and not torch.isinf(max_abs):
                audio = audio / (max_abs + 1e-8)
            
            audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
            waveform = audio.detach().cpu().numpy()
            audio = torch.tensor(waveform, dtype=torch.float32) * 32767
            audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
            torchaudio.save(output_path, audio, sample_rate)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [INFO]: Saved standalone audio: {output_path}")
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_standalone_audio [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


def generate_and_cache_prefix(model, sample_rate, sample_size, base_prompt, prefix_steps=150, text_cfg_scale=4.0, duration=18, seed=32341278, cache_filename=None, device='cuda'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [INFO]: Starting prefix generation")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [INFO]: prefix_steps={prefix_steps}, cfg_scale={text_cfg_scale}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [INFO]: Prompt: {base_prompt}")
    
    try:
        conditioning = [{"prompt": base_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        try:
            cond_test = model.conditioner(conditioning, device=device)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [DEBUG]: Prefix cond keys={list(cond_test.keys())}")
            for key in cond_test:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [DEBUG]: {key} shape={[t.shape if isinstance(t, torch.Tensor) else None for t in cond_test[key]]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [WARNING]: Prefix conditioning failed: {e}")
            raise
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [DEBUG]: Calling generate_diffusion_cond with return_latents=True")
        
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
                seed=seed,
                return_latents=True
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [INFO]: Prefix latent shape={prefix_latent.shape}, mean={prefix_latent.mean().item():.6f}, std={prefix_latent.std().item():.6f}, final_sigma={final_sigma:.6f}")
        
        cache = PrefixLatentCache()
        cache_filepath = cache.save_prefix(
            prefix_latent, 
            final_sigma, 
            base_prompt, 
            prefix_steps, 
            text_cfg_scale, 
            seed,
            duration,
            filename=cache_filename
        )
        
        return prefix_latent, final_sigma, cache_filepath
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_and_cache_prefix [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


def generate_branch_from_prefix(model, sample_rate, sample_size, prefix_latent, final_sigma, branch_prompt, output_path, branch_steps=200, branch_cfg_scale=20.0, duration=18, seed=32341278, device='cuda'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [INFO]: Starting branch generation")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [INFO]: branch_steps={branch_steps}, branch_cfg_scale={branch_cfg_scale}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [INFO]: Branch prompt: {branch_prompt}")
    
    try:
        branch_conditioning = [{"prompt": branch_prompt, "seconds_start": 0, "seconds_total": duration}]
        
        try:
            cond = model.conditioner(branch_conditioning, device=device)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [DEBUG]: Branch cond keys={list(cond.keys())}")
            for key in cond:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [DEBUG]: {key} shape={[t.shape if isinstance(t, torch.Tensor) else None for t in cond[key]]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [WARNING]: Branch conditioning failed: {e}")
            raise
        
        if prefix_latent.device != torch.device(device):
            prefix_latent = prefix_latent.to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            audio = generate_diffusion_cond(
                model,
                steps=branch_steps,
                cfg_scale=branch_cfg_scale,
                conditioning=branch_conditioning,
                sample_size=sample_size,
                sigma_min=0.2,
                sigma_max=final_sigma,
                sampler_type="dpmpp-3m-sde",
                device=device,
                seed=seed,
                init_noise=prefix_latent,
                init_sigma=final_sigma
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [INFO]: Decoded branch shape={audio.shape}, mean={audio.mean().item():.6f}, std={audio.std().item():.6f}")
        
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 1.5 or torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [WARNING]: Audio may be noisy: max_abs={max_abs:.6f}, has_nan={torch.isnan(audio).any()}, has_inf={torch.isinf(audio).any()}")
        
        if max_abs > 0 and not torch.isinf(max_abs):
            audio = audio / (max_abs + 1e-8)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [INFO]: Normalized branch: min={audio.min().item():.6f}, max={audio.max().item():.6f}")
        
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        waveform = audio.detach().cpu().numpy()
        audio = torch.tensor(waveform, dtype=torch.float32) * 32767
        audio = torch.clamp(audio, -32767, 32767).to(torch.int16)
        torchaudio.save(output_path, audio, sample_rate)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [INFO]: Saved {output_path}")
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [ERROR]: Error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] generate_branch_from_prefix [DEBUG]: Stack trace: {traceback.format_exc()}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Branching Diffusion Audio Generation with Baseline Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Load prefix from cache instead of generating new one'
    )
    
    parser.add_argument(
        '--cache-file',
        type=str,
        default='my_prefix_v1.safetensors',
        help='Cache file name to save/load (default: my_prefix_v1.safetensors)'
    )
    
    parser.add_argument(
        '--list-cache',
        action='store_true',
        help='List all cached prefixes and exit'
    )
    
    parser.add_argument(
        '--no-base',
        action='store_true',
        help='Skip generating base audio for comparison'
    )
    
    parser.add_argument(
        '--no-standalone',
        action='store_true',
        help='Skip generating standalone baseline audio for comparison'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=r'stableaudiocode\model_config.json',
        help='Path to model config file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=r'stableaudiocode\model.safetensors',
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--prefix-steps',
        type=int,
        default=150,
        help='Number of steps for prefix generation (default: 150)'
    )
    
    parser.add_argument(
        '--branch-steps',
        type=int,
        default=150,
        help='Number of steps for branch generation (default: 150)'
    )
    
    parser.add_argument(
        '--standalone-steps',
        type=int,
        default=200,
        help='Number of steps for standalone baseline generation (default: 200)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Audio duration in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=32341278,
        help='Random seed for reproducibility (default: 32341278)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.list_cache:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Listing cached prefixes")
        cache = PrefixLatentCache()
        cache.list_cached_prefixes()
        sys.exit(0)
    
    config_path = args.config
    ckpt_path = args.checkpoint
    
    try:
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
        sample_size = 44100 * args.duration
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Configuration:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Sample rate: {sample_rate}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Sample size: {sample_size}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Duration: {args.duration}s")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Device: {device}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Use cached prefix: {args.use_cache}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Cache file: {args.cache_file}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Prefix steps: {args.prefix_steps}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Branch steps: {args.branch_steps}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Standalone steps: {args.standalone_steps}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Seed: {args.seed}")
        
        base_prompt = "Format: Band | Genre: Rock | Subgenre: Indie Rock | Instruments: echoing electric guitars with chorus, \
                        well recorded drum-kit, Electric Bass, occasional soaring harmonies | Moving, Epic, Climactic| BPM: 125"
        
        branches = [
            {
                "name": "piano",
                "prompt": "Genre: Rock | Subgenre: Pop Rock, Indie Rock | Instruments: soft piano | Moods: soft| Tempo: Medium",
                "output_branch": "output_soft_piano_branch.wav",
                "output_standalone": "output_verse_piano_standalone.wav",
                "cfg_scale": 10.0
            },
            {
                "name": "Cine_Orchestral",
                "prompt": "Ensemble | Subgenre: Modern Orchestral | Instruments: string ensemble, soft piano, ambient percussion, brass swells |\
                        Moods: inspiring, epic, heartfelt, subtle | 125 BPM",
                "output_branch": "output_Cinematic_Orchestral_branch.wav",
                "output_standalone": "output_cinematic_orchestral_standalone.wav",
                "cfg_scale": 10.0
            },
            {
                "name": "Alter_Pop",
                "prompt": "Format: Band | Subgenre: Dream Pop | Instruments: reverb guitars, synth layers, deep bass, gentle drums, airy vocals |\
                        Moods: emotional, nostalgic, soaring | 125 BPM",
                "output_branch": "output_Alternative_Pop_branch.wav",
                "output_standalone": "output_alternative_pop_standalone.wav",
                "cfg_scale": 10.0
            }
        ]
            
            
            
            # {
            #     "name": "verse_piano",
            #     "prompt": "Genre: Indie | Subgenre: Pop Rock, Indie Rock | Instruments: piano, drum machine, organ | Moods: soft, happy | Tempo: Medium",
            #     "output_branch": "output_verse_piano_branched.wav",
            #     "output_standalone": "output_verse_piano_standalone.wav",
            #     "cfg_scale": 18.0
            # },
            # {
            #     "name": "verse_acoustic",
            #     "prompt": "Genre: Folk | Subgenre: Indie Folk | Instruments: acoustic guitar, soft vocals, light percussion | Moods: intimate, warm | Tempo: Medium",
            #     "output_branch": "output_verse_acoustic_branched.wav",
            #     "output_standalone": "output_verse_acoustic_standalone.wav",
            #     "cfg_scale": 18.0
            # },
            # {
            #     "name": "verse_electronic",
            #     "prompt": "Genre: Electronic | Subgenre: Synth Pop | Instruments: synthesizer, electronic drums, bass synth | Moods: energetic, modern | Tempo: Fast",
            #     "output_branch": "output_verse_electronic_branched.wav",
            #     "output_standalone": "output_verse_electronic_standalone.wav",
            #     "cfg_scale": 18.0
            # }
        # ]
        
        # Generate base audio (from base prompt)
        if not args.no_base:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating base audio for comparison")
            generate_base_audio(
                model, sample_rate, sample_size, base_prompt, "output_base.wav",
                num_steps=200, text_cfg_scale=10.0, duration=args.duration
            )
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Skipping base audio generation")
        
        # Generate or load prefix latent
        if not args.use_cache:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating and caching prefix latent")
            prefix_latent, final_sigma, cache_filepath = generate_and_cache_prefix(
                model, sample_rate, sample_size, base_prompt,
                prefix_steps=args.prefix_steps,
                text_cfg_scale=10.0,
                duration=args.duration,
                seed=args.seed,
                cache_filename=args.cache_file
            )
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Loading cached prefix latent")
            cache = PrefixLatentCache()
            
            try:
                prefix_latent, final_sigma, metadata = cache.load_prefix(args.cache_file)
                prefix_latent = prefix_latent.to(device)
                
                # Validate duration matches
                cached_duration = int(metadata.get('duration', 30))
                if cached_duration != args.duration:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Duration mismatch!")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Cached: {cached_duration}s, Requested: {args.duration}s")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Please use --duration {cached_duration} or regenerate cache")
                    sys.exit(1)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Using cached prefix:")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Prompt: {metadata['prompt'][:80]}...")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Steps: {metadata['steps']}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - CFG scale: {metadata['cfg_scale']}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Seed: {metadata['seed']}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Duration: {metadata['duration']}s")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - Created: {metadata['timestamp']}")
            except FileNotFoundError:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Cache file not found: {args.cache_file}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Available cached prefixes:")
                cache.list_cached_prefixes()
                sys.exit(1)
        
        # Generate branched audio and standalone baseline for each branch
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating {len(branches)} branches (branched + standalone)")
        
        for i, branch in enumerate(branches, 1):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: ========== Branch {i}/{len(branches)}: {branch['name']} ==========")
            
            # Generate branched version (using prefix latent)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating BRANCHED version")
            generate_branch_from_prefix(
                model, sample_rate, sample_size,
                prefix_latent, final_sigma,
                branch_prompt=branch['prompt'],
                output_path=branch['output_branch'],
                branch_steps=args.branch_steps,
                branch_cfg_scale=branch['cfg_scale'],
                duration=args.duration,
                seed=args.seed
            )
            
            # Generate standalone baseline (direct generation without prefix)
            if not args.no_standalone:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Generating STANDALONE baseline version")
                generate_standalone_audio(
                    model, sample_rate, sample_size,
                    prompt=branch['prompt'],
                    output_path=branch['output_standalone'],
                    num_steps=args.standalone_steps,
                    text_cfg_scale=branch['cfg_scale'],
                    duration=args.duration,
                    seed=args.seed
                )
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Skipping standalone baseline generation")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: All generations completed successfully")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Output files:")
        if not args.no_base:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - output_base.wav (base reference from base prompt)")
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Branched outputs (using prefix latent):")
        for branch in branches:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - {branch['output_branch']} ({branch['name']})")
        
        if not args.no_standalone:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Standalone baseline outputs (direct generation):")
            for branch in branches:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: - {branch['output_standalone']} ({branch['name']})")

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Main execution error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [DEBUG]: Stack trace: {traceback.format_exc()}")
        sys.exit(1)