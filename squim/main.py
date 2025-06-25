import argparse
import os
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio.functional as F
import csv
from tqdm import tqdm
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model once, on correct device
model = SQUIM_OBJECTIVE.get_model().to(device)
model.eval()


def process_audio_file(filepath):
    try:
        audio, fs = torchaudio.load(filepath)
        if fs != 16000:
            audio = F.resample(audio, fs, 16000)
        audio = audio.to(device)
        stoi, pesq, sisdr = model(audio[0:1, :])
        return stoi.item(), pesq.item(), sisdr.item()
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None


def main(input_dir):
    results = []

    # Collect all .wav file paths first
    wav_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files if file.endswith(".wav")
    ]

    for filepath in tqdm(wav_files, desc="Processing audio files"):
        stoi, pesq, sisdr = process_audio_file(filepath)
        if None not in (stoi, pesq, sisdr):
            results.append({
                "id": os.path.basename(filepath),
                "STOI": stoi,
                "PESQ": pesq,
                "SI-SDR": sisdr
            })

    output_csv = "results.csv"
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "STOI", "PESQ", "SI-SDR"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wav files and compute STOI, PESQ, SI-SDR")
    parser.add_argument("input_dir", type=str, help="Input directory containing .wav files")
    args = parser.parse_args()

    main(args.input_dir)
