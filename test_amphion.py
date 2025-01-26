from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download

import torch
import librosa
import soundfile as sf

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



test_wav_path = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/wav/p225/p225_003.wav"
test_wav = librosa.load(test_wav_path, sr=16000)[0]
test_wav = torch.from_numpy(test_wav).float()
test_wav = test_wav.unsqueeze(0).unsqueeze(0)

with torch.no_grad():

    # encode
    enc_out = fa_encoder(test_wav)
    print(enc_out.shape)

    # quantize
    vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
    
    # latent after quantization
    print(vq_post_emb.shape)
    
    # codes
    print("vq id shape:", vq_id.shape)
    
    # get prosody code
    prosody_code = vq_id[:1]
    print(prosody_code)
    print("prosody code shape:", prosody_code.shape)
    
    # get content code
    cotent_code = vq_id[1:3]
    print("content code shape:", cotent_code.shape)
    
    # get residual code (acoustic detail codes)
    residual_code = vq_id[3:]
    print("residual code shape:", residual_code.shape)
    
    # speaker embedding
    print("speaker embedding shape:", spk_embs.shape)

    # decode (recommand)
    recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
    print(recon_wav.shape)
    sf.write("recon.wav", recon_wav[0][0].cpu().numpy(), 16000)
