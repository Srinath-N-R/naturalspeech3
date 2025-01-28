from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np
import librosa
import torch
import json
from torch.utils.data import Dataset
import logging
from phonemizer import phonemize
from phonemizer.separator import Separator
from utils.tokenizer import Tokenizer
import re

logger = logging.getLogger(__name__)



class PhonemizeText:
    def __init__(self):
        pass
    
    def phonemize(self, text, language='en-us', separator=' h# '):
        return phonemize_text(text, language, separator)


def phonemize_text(text, language='en-us', separator=' h# '):
    try:
        return  f"{separator}" + phonemize(
            text,
            language=language,
            backend='espeak',
            separator=Separator(word=separator, phone=' '),
            strip=True,
            preserve_punctuation=True
        ) + f"{separator}"
    except RuntimeError as e:
        print(f"Phonemization error: {str(e)}")
        return None


def clean_text(text):
    text = re.sub(r'[^\w\s\'\-]', '', text)  # Remove punctuation except apostrophes and hyphens
    text = re.sub(r'\s+', ' ', text)  # Collapse spaces
    return text.lower().strip()  # Normalize case and trim spaces


tokenizer = Tokenizer(phonemizer=PhonemizeText(), text_cleaner=clean_text)

# def custom_collate_fn(batch):
#     target_audio = [torch.tensor(item['target_audio']) for item in batch]
#     prompt_audio = [torch.tensor(item['prompt_audio']) for item in batch]
#     phoneme_sequence = tokenizer.phoneme_to_tensor_ids([item['phoneme_sequence'].split() for item in batch], padding_value=69)
#     phoneme_durations = [torch.tensor([np.ceil(it["duration"] * (1000/12.5)) for it in item['phoneme_durations'].values()] + [0]) for item in batch]

#     target_audio = pad_sequence(target_audio, batch_first=True).unsqueeze(1)
#     prompt_audio = pad_sequence(prompt_audio, batch_first=True)
#     phoneme_sequence = pad_sequence(phoneme_sequence, batch_first=True)
#     phoneme_durations = pad_sequence(phoneme_durations, batch_first=True)

#     return {
#         'target_audio': target_audio,
#         'prompt_audio': prompt_audio,
#         'phoneme_sequence': phoneme_sequence,
#         'phoneme_durations': phoneme_durations,
#     }


def custom_collate_fn(batch):
    target_audio = [torch.tensor(item['target_audio'], dtype=torch.float32) for item in batch]
    prompt_audio = [torch.tensor(item['prompt_audio'], dtype=torch.float32) for item in batch]

    # The rest is unchanged:
    phoneme_sequence = tokenizer.phoneme_to_tensor_ids(
        [item['phoneme_sequence'].split() for item in batch],
        padding_value=69
    )

    phoneme_durations = [
        torch.tensor(
            [np.ceil(it["duration"] * (1000 / 12.5)) for it in item['phoneme_durations'].values()] + [0],
            dtype=torch.float32
        )
        for item in batch
    ]

    target_audio = pad_sequence(target_audio, batch_first=True).unsqueeze(1)   # [B, 1, T]
    prompt_audio = pad_sequence(prompt_audio, batch_first=True).unsqueeze(1)   # [B, 1, T]

    phoneme_sequence = pad_sequence(phoneme_sequence, batch_first=True)
    phoneme_durations = pad_sequence(phoneme_durations, batch_first=True)

    return {
        'target_audio': target_audio,
        'prompt_audio': prompt_audio,
        'phoneme_sequence': phoneme_sequence,
        'phoneme_durations': phoneme_durations,
    }


class CustomDataset(Dataset):
    def __init__(self, dataset_folder, max_items=None, sampling_rate=16000):
            self.dataset_folder = dataset_folder
            self.sampling_rate = sampling_rate
            # Collect files
            self.audio_files = sorted(
                [path for path in (Path(dataset_folder) / 'wav').rglob('*.wav') if not path.name.startswith('._')]
            )
            self.phoneme_files = sorted(
                [path for path in (Path(dataset_folder) / 'phonemized').rglob('*.txt') if not path.name.startswith('._')]
            )
            self.segment_files = sorted(
                [path for path in (Path(dataset_folder) / 'segments').rglob('*.json') if not path.name.startswith('._')]
            )

            # Get the base file names (without extensions) for matching
            audio_basenames = {path.stem for path in self.audio_files}
            phoneme_basenames = {path.stem for path in self.phoneme_files}
            segment_basenames = {path.stem for path in self.segment_files}

            common_basenames = audio_basenames & phoneme_basenames & segment_basenames

            # Filter files to only include common base names
            self.audio_files = [path for path in self.audio_files if path.stem in common_basenames]
            self.phoneme_files = [path for path in self.phoneme_files if path.stem in common_basenames]
            self.segment_files = [path for path in self.segment_files if path.stem in common_basenames]


            if max_items is not None:
                self.audio_files = self.audio_files[:max_items]
                self.phoneme_files = self.phoneme_files[:max_items]
                self.segment_files = self.segment_files[:max_items]

            logger.info(f"Dataset initialized with {len(self.audio_files)} audio files, "
                        f"{len(self.phoneme_files)} phoneme files, "
                        f"{len(self.segment_files)} segments, "
            )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]
            phoneme_path = self.phoneme_files[idx]
            segment_path = self.segment_files[idx]

            audio = librosa.load(str(audio_path), sr=16000)[0]
                
            with open(phoneme_path, 'r') as f:
                phoneme = f.read()
            
            with open(segment_path, 'r') as file:
                segment = json.load(file)

            speaker_folder = self.audio_files[idx].parent
            prompt_audio = self.accumulate_prompt_audio(speaker_folder)

            return {
                'target_audio': audio,
                'prompt_audio': prompt_audio,
                'phoneme_sequence': phoneme,
                'phoneme_durations': segment
            }
        except IndexError as e:
            logger.error(f"IndexError: {e} - Index {idx} is out of range.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error at index {idx}: {e}")
            raise


    def accumulate_prompt_audio(self, speaker_folder):
        """
        Accumulate audio for a speaker up to 30 seconds and pad if necessary.
        """
        max_duration = 15  # Maximum duration in seconds
        accumulated_audio = []
        total_frames = 0
        sampling_rate = self.sampling_rate

        for audio_path in speaker_folder.rglob('*.wav'):
            try:
                audio = librosa.load(str(audio_path), sr=16000)[0]

                frames = audio.shape[-1]
                if total_frames + frames > max_duration * sampling_rate:
                    remaining_frames = max_duration * sampling_rate - total_frames
                    accumulated_audio.append(torch.tensor(audio[:, :remaining_frames]))
                    total_frames += remaining_frames
                    break
                else:
                    accumulated_audio.append(torch.tensor(audio))
                    total_frames += frames

            except Exception as e:
                logger.warning(f"Error loading prompt audio {audio_path}: {e}")
                continue

        # Combine all accumulated audio
        if accumulated_audio:
            accumulated_audio = torch.cat(accumulated_audio, dim=-1)
        else:
            accumulated_audio = torch.zeros(1, 0)

        # Pad to 30 seconds if necessary
        if accumulated_audio.shape[0] < max_duration * sampling_rate:
            padding = max_duration * sampling_rate - accumulated_audio.shape[0]
            accumulated_audio = torch.nn.functional.pad(accumulated_audio, (0, padding), mode='constant', value=0)

        return accumulated_audio

