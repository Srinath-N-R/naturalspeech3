# from .naturalspeech3 import NaturalSpeech3
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from naturalspeech3 import NaturalSpeech3

import torch
import librosa
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.eval()
fa_decoder.eval()

# 1) Instantiate your model
model = NaturalSpeech3(
    facodec_encoder=fa_encoder,
    facodec_decoder=fa_decoder,
)
model.eval()

# 2) Prepare a single-sample test input
sample_phoneme_seq = [[10, 11, 12, 13], [10, 11, 12, 13], [10, 11, 12, 13], [10, 11, 12, 13]]     # e.g., some arbitrary IDs
sample_durations = [[2.0, 3.0, 2.0, 4.0], [2.0, 3.0, 2.0, 4.0], [2.0, 3.0, 2.0, 4.0], [2.0, 3.0, 2.0, 4.0]]     # a guess for durations


sample_prompt_path_1 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_001.wav"
sample_prompt_path_2 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_002.wav"
sample_prompt_path_3 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_003.wav"
sample_prompt_path_4 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_004.wav"

sample_target_path_1 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_001.wav"
sample_target_path_2 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_001.wav"
sample_target_path_3 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_001.wav"
sample_target_path_4 = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_001.wav"

def read_audio(sample_prompt_path):
    sample_prompt_audio = librosa.load(sample_prompt_path, sr=16000)[0]
    sample_prompt_audio = torch.from_numpy(sample_prompt_audio).float()
    return sample_prompt_audio.unsqueeze(0)


sample_prompt_audio = [read_audio(sample_prompt_path_1).squeeze(0), read_audio(sample_prompt_path_2).squeeze(0), read_audio(sample_prompt_path_3).squeeze(0), read_audio(sample_prompt_path_4).squeeze(0)]
sample_target_audio = [read_audio(sample_target_path_1).squeeze(0), read_audio(sample_target_path_2).squeeze(0), read_audio(sample_target_path_3).squeeze(0), read_audio(sample_target_path_4).squeeze(0)]

sample_prompt_audio = pad_sequence(sample_prompt_audio, batch_first=True).unsqueeze(1)
sample_target_audio = pad_sequence(sample_target_audio, batch_first=True).unsqueeze(1)


# 3) Run forward pass
with torch.no_grad():
    outputs = model(
        phoneme_sequence=torch.tensor(sample_phoneme_seq, dtype=torch.long),
        phoneme_durations=torch.tensor(sample_durations, dtype=torch.float),
        prompt_audio=sample_prompt_audio,
        target_audio=sample_target_audio
    )


# 4) Inspect outputs
print("Outputs keys:", outputs.keys())
print("durations_pred:", outputs["durations_pred"])
print("prosody_pred:", outputs["prosody_pred"])
print("c_ph shape:", outputs["c_ph"].shape)
