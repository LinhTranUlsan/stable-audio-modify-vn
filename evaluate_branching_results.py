
import sys
import os
import torch
import torchaudio
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import json
from typing import List, Dict, Tuple
import traceback

# Try to import optional dependencies
try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    print("[WARNING] scipy not available, FAD metric will be disabled")
    SCIPY_AVAILABLE = False

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    print("[WARNING] laion_clap not available, CLAP score will be disabled")
    CLAP_AVAILABLE = False


class AudioMetrics:
    """Comprehensive audio quality metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize CLAP model if available
        if CLAP_AVAILABLE:
            try:
                # Set torch.load to use weights_only=False for CLAP compatibility
                import torch.serialization
                old_load = torch.load
                torch.load = lambda *args, **kwargs: old_load(*args, **{**kwargs, 'weights_only': False})
                
                self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
                self.clap_model.load_ckpt()
                
                # Restore original torch.load
                torch.load = old_load
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] AudioMetrics [INFO]: CLAP model loaded successfully")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] AudioMetrics [WARNING]: Failed to load CLAP model: {e}")
                self.clap_model = None
        else:
            self.clap_model = None
    
    def load_audio(self, filepath: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample if necessary"""
        waveform, sr = torchaudio.load(filepath)
        
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        
        return waveform, sr
    
    def compute_mfcc(self, waveform: torch.Tensor, sr: int, n_mfcc: int = 13) -> torch.Tensor:
        """Compute Mel-Frequency Cepstral Coefficients"""
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}
        )
        mfcc = mfcc_transform(waveform)
        return mfcc
    
    def mel_cepstral_distortion(self, ref_audio: torch.Tensor, gen_audio: torch.Tensor, sr: int) -> float:
        """Compute Mel Cepstral Distortion (MCD)"""
        # Ensure same length
        min_len = min(ref_audio.shape[-1], gen_audio.shape[-1])
        ref_audio = ref_audio[..., :min_len]
        gen_audio = gen_audio[..., :min_len]
        
        # Compute MFCCs
        ref_mfcc = self.compute_mfcc(ref_audio, sr)
        gen_mfcc = self.compute_mfcc(gen_audio, sr)
        
        # Align temporal dimension
        min_frames = min(ref_mfcc.shape[-1], gen_mfcc.shape[-1])
        ref_mfcc = ref_mfcc[..., :min_frames]
        gen_mfcc = gen_mfcc[..., :min_frames]
        
        # Exclude c0 and compute distance
        ref_mfcc = ref_mfcc[:, 1:, :]
        gen_mfcc = gen_mfcc[:, 1:, :]
        
        diff = ref_mfcc - gen_mfcc
        frame_dist = torch.sqrt(torch.sum(diff ** 2, dim=1))
        mcd = (10.0 / np.log(10)) * torch.sqrt(2 * torch.mean(frame_dist ** 2))
        
        return mcd.item()
    
    def compute_spectrogram(self, waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """Compute magnitude spectrogram"""
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1
        )
        return spec_transform(waveform)
    
    def multi_scale_spectral_loss(self, ref_audio: torch.Tensor, gen_audio: torch.Tensor) -> Dict[str, float]:
        """Compute Multi-Scale Spectral Loss"""
        min_len = min(ref_audio.shape[-1], gen_audio.shape[-1])
        ref_audio = ref_audio[..., :min_len]
        gen_audio = gen_audio[..., :min_len]
        
        scales = [512, 1024, 2048, 4096]
        losses = {}
        total_loss = 0.0
        
        for n_fft in scales:
            hop_length = n_fft // 4
            
            ref_spec = self.compute_spectrogram(ref_audio, n_fft, hop_length)
            gen_spec = self.compute_spectrogram(gen_audio, n_fft, hop_length)
            
            min_frames = min(ref_spec.shape[-1], gen_spec.shape[-1])
            ref_spec = ref_spec[..., :min_frames]
            gen_spec = gen_spec[..., :min_frames]
            
            loss = torch.mean(torch.abs(ref_spec - gen_spec))
            losses[f'scale_{n_fft}'] = loss.item()
            total_loss += loss.item()
        
        losses['total'] = total_loss / len(scales)
        return losses
    
    def snr_metrics(self, ref_audio: torch.Tensor, gen_audio: torch.Tensor) -> Dict[str, float]:
        """Compute SNR and SI-SNR"""
        min_len = min(ref_audio.shape[-1], gen_audio.shape[-1])
        ref_audio = ref_audio[..., :min_len]
        gen_audio = gen_audio[..., :min_len]
        
        # Standard SNR
        noise = ref_audio - gen_audio
        signal_power = torch.mean(ref_audio ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        
        # Scale-Invariant SNR
        dot_product = torch.sum(ref_audio * gen_audio)
        ref_energy = torch.sum(ref_audio ** 2)
        scaling = dot_product / (ref_energy + 1e-8)
        projection = scaling * ref_audio
        
        noise_si = gen_audio - projection
        si_snr = 10 * torch.log10(
            torch.sum(projection ** 2) / (torch.sum(noise_si ** 2) + 1e-8)
        )
        
        return {
            'snr': snr.item(),
            'si_snr': si_snr.item()
        }
    
    def compute_vggish_embeddings(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Compute embeddings for FAD calculation"""
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        mel_spec = torch.log(mel_spec + 1e-8)
        pooled = torch.nn.functional.adaptive_avg_pool2d(mel_spec.unsqueeze(0), (128, 96))
        
        return pooled.squeeze().cpu().numpy().flatten()
    
    def frechet_audio_distance(self, ref_embeddings: np.ndarray, gen_embeddings: np.ndarray) -> float:
        """Compute Fréchet Audio Distance (FAD)"""
        if not SCIPY_AVAILABLE:
            return -1.0
        
        # Ensure embeddings are 2D
        if ref_embeddings.ndim == 1:
            ref_embeddings = ref_embeddings.reshape(1, -1)
        if gen_embeddings.ndim == 1:
            gen_embeddings = gen_embeddings.reshape(1, -1)
        
        # Need at least 2 samples
        if ref_embeddings.shape[0] < 2 or gen_embeddings.shape[0] < 2:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] AudioMetrics [WARNING]: Not enough samples for FAD, skipping")
            return -1.0
        
        try:
            mu_ref = np.mean(ref_embeddings, axis=0)
            mu_gen = np.mean(gen_embeddings, axis=0)
            
            sigma_ref = np.cov(ref_embeddings, rowvar=False)
            sigma_gen = np.cov(gen_embeddings, rowvar=False)
            
            # Ensure 2D
            if sigma_ref.ndim == 0:
                sigma_ref = sigma_ref.reshape(1, 1)
            if sigma_gen.ndim == 0:
                sigma_gen = sigma_gen.reshape(1, 1)
            
            diff = mu_ref - mu_gen
            covmean, _ = linalg.sqrtm(sigma_ref @ sigma_gen, disp=False)
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fad = diff.dot(diff) + np.trace(sigma_ref + sigma_gen - 2 * covmean)
            
            return float(fad)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] AudioMetrics [WARNING]: FAD computation failed: {e}")
            return -1.0
    
    def clap_score(self, audio_path: str, text_prompt: str) -> float:
        """Compute CLAP Score"""
        if self.clap_model is None:
            return -1.0
        
        try:
            audio_embed = self.clap_model.get_audio_embedding_from_filelist(
                x=[audio_path], use_tensor=True
            )
            
            text_embed = self.clap_model.get_text_embedding([text_prompt], use_tensor=True)
            
            audio_embed = audio_embed / torch.norm(audio_embed, dim=-1, keepdim=True)
            text_embed = text_embed / torch.norm(text_embed, dim=-1, keepdim=True)
            
            similarity = torch.sum(audio_embed * text_embed, dim=-1)
            
            return similarity.item()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] AudioMetrics [ERROR]: CLAP score failed: {e}")
            return -1.0


def evaluate_pair(audio1_path: str, audio2_path: str, prompt: str, metrics: AudioMetrics, comparison_name: str) -> Dict:
    """Evaluate a pair of audio files"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Evaluating {comparison_name}")
    
    results = {
        'comparison': comparison_name,
        'audio1': Path(audio1_path).name,
        'audio2': Path(audio2_path).name,
        'prompt': prompt,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Load audio files
        audio1, sr = metrics.load_audio(audio1_path)
        audio2, _ = metrics.load_audio(audio2_path, target_sr=sr)
        
        # Ensure stereo
        if audio1.shape[0] == 1:
            audio1 = audio1.repeat(2, 1)
        if audio2.shape[0] == 1:
            audio2 = audio2.repeat(2, 1)
        
        # Compute metrics
        print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Computing MCD...")
        results['mcd'] = metrics.mel_cepstral_distortion(audio1, audio2, sr)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Computing Multi-Scale Spectral Loss...")
        mss_losses = metrics.multi_scale_spectral_loss(audio1, audio2)
        results['spectral_loss'] = mss_losses
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Computing SNR metrics...")
        snr_results = metrics.snr_metrics(audio1, audio2)
        results.update(snr_results)
        
        # CLAP scores
        if prompt and metrics.clap_model is not None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Computing CLAP scores...")
            results['clap_score_audio1'] = metrics.clap_score(audio1_path, prompt)
            results['clap_score_audio2'] = metrics.clap_score(audio2_path, prompt)
            results['clap_score_diff'] = abs(results['clap_score_audio1'] - results['clap_score_audio2'])
        
        # FAD - FIXED VERSION
        if SCIPY_AVAILABLE:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Computing FAD...")
            audio1_embed = metrics.compute_vggish_embeddings(audio1, sr)
            audio2_embed = metrics.compute_vggish_embeddings(audio2, sr)
            
            # Split into equal-sized chunks
            n_chunks = 10
            chunk_size = len(audio1_embed) // n_chunks
            
            # Trim to be divisible by n_chunks
            audio1_embed_trimmed = audio1_embed[:chunk_size * n_chunks]
            audio2_embed_trimmed = audio2_embed[:chunk_size * n_chunks]
            
            # Reshape into equal chunks
            audio1_chunks = audio1_embed_trimmed.reshape(n_chunks, chunk_size)
            audio2_chunks = audio2_embed_trimmed.reshape(n_chunks, chunk_size)
            
            results['fad'] = metrics.frechet_audio_distance(audio1_chunks, audio2_chunks)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [INFO]: Evaluation complete for {comparison_name}")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [ERROR]: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluate_pair [DEBUG]: {traceback.format_exc()}")
        results['error'] = str(e)
    
    return results


def print_comparison_table(results_list: List[Dict], title: str):
    """Print comparison results in formatted table"""
    print("\n" + "="*120)
    print(f"{title}")
    print("="*120)
    
    print(f"\n{'Branch':<25} {'MCD':<12} {'SNR':<12} {'SI-SNR':<12} {'Spectral':<12} {'CLAP Δ':<12} {'FAD':<12}")
    print("-"*120)
    
    for result in results_list:
        if 'error' in result:
            print(f"{result['comparison']:<25} ERROR: {result['error']}")
            continue
        
        comp_name = result['comparison']
        mcd = result.get('mcd', -1)
        snr = result.get('snr', -1)
        si_snr = result.get('si_snr', -1)
        spectral = result.get('spectral_loss', {}).get('total', -1)
        clap_diff = result.get('clap_score_diff', -1)
        fad = result.get('fad', -1)
        
        print(f"{comp_name:<25} {mcd:>11.4f} {snr:>11.4f} {si_snr:>11.4f} {spectral:>11.4f} {clap_diff:>11.4f} {fad:>11.4f}")
    
    print("="*120)


def print_summary_statistics(all_results: Dict):
    """Print summary statistics"""
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    branched_vs_standalone = all_results.get('branched_vs_standalone', [])
    
    if branched_vs_standalone:
        mcds = [r.get('mcd', -1) for r in branched_vs_standalone if 'error' not in r and r.get('mcd', -1) > 0]
        snrs = [r.get('snr', -1) for r in branched_vs_standalone if 'error' not in r and r.get('snr', -1) > -100]
        si_snrs = [r.get('si_snr', -1) for r in branched_vs_standalone if 'error' not in r and r.get('si_snr', -1) > -100]
        
        print(f"\nBranched vs Standalone Comparison:")
        if mcds:
            print(f"  MCD: Mean={np.mean(mcds):.4f}, Std={np.std(mcds):.4f}, Min={np.min(mcds):.4f}, Max={np.max(mcds):.4f}")
        if snrs:
            print(f"  SNR: Mean={np.mean(snrs):.4f}, Std={np.std(snrs):.4f}, Min={np.min(snrs):.4f}, Max={np.max(snrs):.4f}")
        if si_snrs:
            print(f"  SI-SNR: Mean={np.mean(si_snrs):.4f}, Std={np.std(si_snrs):.4f}, Min={np.min(si_snrs):.4f}, Max={np.max(si_snrs):.4f}")
    
    print("\n" + "="*120)
    print("INTERPRETATION GUIDE:")
    print("="*120)
    print("Branched vs Standalone:")
    print("  - Lower MCD = More similar")
    print("  - Higher SNR/SI-SNR = Better quality")
    print("  - Lower Spectral Loss = More similar frequency content")
    print("  - Lower CLAP Δ = Similar text alignment")
    print("  - Lower FAD = More similar distribution")
    print("="*120 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate branching diffusion vs standalone generation')
    
    parser.add_argument('--base', type=str, default='output_base.wav', help='Base reference audio')
    parser.add_argument('--no-base-comparison', action='store_true', help='Skip base comparison')
    parser.add_argument('--output-json', type=str, default='evaluation_comparison_results.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Starting comprehensive evaluation")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Device: {args.device}")
    
    branches = [
        # {
        #     "name": "verse_piano",
        #     "prompt": "Genre: Indie | Subgenre: Pop Rock, Indie Rock | Instruments: piano, drum machine, organ | Moods: soft, happy | Tempo: Medium",
        #     "branched": "output_verse_piano_branched.wav",
        #     "standalone": "output_verse_piano_standalone.wav"
        # },
        # {
        #     "name": "verse_acoustic",
        #     "prompt": "Genre: Folk | Subgenre: Indie Folk | Instruments: acoustic guitar, soft vocals, light percussion | Moods: intimate, warm | Tempo: Medium",
        #     "branched": "output_verse_acoustic_branched.wav",
        #     "standalone": "output_verse_acoustic_standalone.wav"
        # },
        # {
        #     "name": "verse_electronic",
        #     "prompt": "Genre: Electronic | Subgenre: Synth Pop | Instruments: synthesizer, electronic drums, bass synth | Moods: energetic, modern | Tempo: Fast",
        #     "branched": "output_verse_electronic_branched.wav",
        #     "standalone": "output_verse_electronic_standalone.wav"
        # },
        
        {
            "name": "piano",
            "prompt": "Genre: Rock | Subgenre: Pop Rock, Indie Rock | Instruments: soft piano | Moods: soft| Tempo: Medium",
            "branched": "output_soft_piano_branch.wav",
            "standalone": "output_verse_piano_standalone.wav"
        },
        {
            "name": "Cine_Orchestral",
            "prompt": "Ensemble | Subgenre: Modern Orchestral | Instruments: string ensemble, soft piano, ambient percussion, brass swells | Moods: inspiring, epic, heartfelt, subtle | 125 BPM",
            "branched": "output_Cinematic_Orchestral_branch.wav",
            "standalone": "output_cinematic_orchestral_standalone.wav"
        },
        {
            "name": "Alter_Pop",
            "prompt": "Format: Band | Subgenre: Dream Pop | Instruments: reverb guitars, synth layers, deep bass, gentle drums, airy vocals | Moods: emotional, nostalgic, soaring | 125 BPM",
            "branched": "output_Alternative_Pop_branch.wav",
            "standalone": "output_alternative_pop_standalone.wav"
        }
    ]
    
    # Check files exist
    missing_files = []
    for branch in branches:
        if not Path(branch['branched']).exists():
            missing_files.append(branch['branched'])
        if not Path(branch['standalone']).exists():
            missing_files.append(branch['standalone'])
    
    if not args.no_base_comparison and not Path(args.base).exists():
        missing_files.append(args.base)
    
    if missing_files:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: Missing files:")
        for f in missing_files:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] main [ERROR]: - {f}")
        sys.exit(1)
    
    # Initialize metrics
    metrics = AudioMetrics(device=args.device)
    
    all_results = {
        'branched_vs_standalone': [],
        'branched_vs_base': [],
        'standalone_vs_base': []
    }
    
    # Comparison 1: Branched vs Standalone
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: ========== COMPARISON 1: Branched vs Standalone ==========")
    for branch in branches:
        result = evaluate_pair(branch['branched'], branch['standalone'], branch['prompt'], metrics, branch['name'])
        all_results['branched_vs_standalone'].append(result)
    
    # Comparison 2 & 3: vs Base
    if not args.no_base_comparison:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: ========== COMPARISON 2: Branched vs Base ==========")
        for branch in branches:
            result = evaluate_pair(branch['branched'], args.base, branch['prompt'], metrics, f"{branch['name']}_branched")
            all_results['branched_vs_base'].append(result)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: ========== COMPARISON 3: Standalone vs Base ==========")
        for branch in branches:
            result = evaluate_pair(branch['standalone'], args.base, branch['prompt'], metrics, f"{branch['name']}_standalone")
            all_results['standalone_vs_base'].append(result)
    
    # Print results
    print_comparison_table(all_results['branched_vs_standalone'], "BRANCHED vs STANDALONE (Same Prompt Comparison)")
    
    if not args.no_base_comparison:
        print_comparison_table(all_results['branched_vs_base'], "BRANCHED vs BASE REFERENCE")
        print_comparison_table(all_results['standalone_vs_base'], "STANDALONE vs BASE REFERENCE")
    
    print_summary_statistics(all_results)
    
    # Save JSON
    with open(args.output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Results saved to {args.output_json}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] main [INFO]: Evaluation completed successfully")