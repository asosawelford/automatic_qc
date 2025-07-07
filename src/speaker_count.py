
import torch
import numpy as np
from typing import Dict, Optional, List
from speechbrain.inference import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering

# --- Re-using our VAD utility ---
from src.vad_utils import get_speech_timestamps

# --- Model Caching for performance ---
_SPEAKER_MODEL = None

def _load_speaker_model():
    """Loads the SpeechBrain x-vector model, caching it globally."""
    global _SPEAKER_MODEL
    if _SPEAKER_MODEL is None:
        print("Initializing speaker embedding model (SpeechBrain x-vector)...")
        # This will download the model on the first run
        _SPEAKER_MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )
    return _SPEAKER_MODEL

def _create_speech_chunks(
    audio_array: np.ndarray,
    speech_timestamps: List[Dict[str, int]],
    chunk_size: int = 24000, # 1.5 seconds * 16000 Hz
    hop_size: int = 8000     # 0.5 seconds * 16000 Hz
) -> List[np.ndarray]:
    """
    Creates overlapping chunks of audio from speech segments.
    This ensures each embedding is calculated on a reasonably long segment.
    """
    chunks = []
    for segment in speech_timestamps:
        start, end = segment['start'], segment['end']
        for i in range(start, end - chunk_size + 1, hop_size):
            chunks.append(audio_array[i : i + chunk_size])
    
    # If there are no chunks but there was speech, it means speech was too short.
    # We take the longest speech segment as a single chunk.
    if not chunks and speech_timestamps:
        longest_segment = max(speech_timestamps, key=lambda s: s['end'] - s['start'])
        chunks.append(audio_array[longest_segment['start']:longest_segment['end']])

    return chunks

def estimate_speaker_count(
    audio_array: np.ndarray,
    sample_rate: int,
    distance_threshold: float = 0.15
) -> Optional[Dict[str, int]]:
    """
    Estimates the number of speakers using x-vector embeddings and clustering.

    Args:
        audio_array (np.ndarray): Mono float32 audio.
        sample_rate (int): The sample rate (must be 16000).
        distance_threshold (float): The clustering threshold. A key parameter to tune.
                                    Represents the max cosine distance for two embeddings
                                    to be considered from the same speaker.
                                    Good values are typically between 0.4 and 0.7.

    Returns:
        A dictionary with the estimated speaker count, or None on error.
    """
    # 1. Load model and get speech segments
    model = _load_speaker_model()
    speech_timestamps = get_speech_timestamps(audio_array, sample_rate)

    if not speech_timestamps:
        return {'estimated_speakers': 0}

    # 2. Create overlapping chunks of speech audio
    speech_chunks = _create_speech_chunks(audio_array, speech_timestamps)

    if not speech_chunks:
        # This case happens if speech exists but is shorter than the minimum chunk size
        return {'estimated_speakers': 1}

    # 3. Extract x-vector embeddings for each chunk
    embeddings = []
    for chunk in speech_chunks:
        # Model expects a batch, so we add a dimension
        tensor_chunk = torch.from_numpy(chunk).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_batch(tensor_chunk)
        # Normalize the embedding (good practice) and remove batch dimension
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=2)
        embeddings.append(embedding.squeeze().numpy())
    
    if len(embeddings) == 1:
        # Only one chunk of speech could be analyzed
        return {'estimated_speakers': 1}

    embeddings_array = np.array(embeddings)

    # 4. Cluster the embeddings to find the number of unique speakers
    # We use Agglomerative Clustering with a distance threshold.
    # This avoids having to guess the number of speakers beforehand.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine', # Use 'cosine' for comparing embeddings
        linkage='complete'
    )
    clustering.fit(embeddings_array)

    num_speakers = clustering.n_clusters_
    return {'estimated_speakers': num_speakers}