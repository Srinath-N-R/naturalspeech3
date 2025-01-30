from naturalspeech3 import NaturalSpeech3
from dataloader import custom_collate_fn, CustomDataset
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import torch
from transformers import Trainer, TrainingArguments
import torch.multiprocessing as mp


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = NaturalSpeech3(
        facodec_encoder=fa_encoder,
        facodec_decoder=fa_decoder,
    ).to(device)


    # Dataset
    train_dataset_folder = "/workspace/datasets/LibriTTS_R-360-Train-new/"
    train_dataset = CustomDataset(dataset_folder=train_dataset_folder, max_items=32)


    val_dataset_folder = "/workspace/datasets/VCTK_val/"
    val_dataset = CustomDataset(dataset_folder=val_dataset_folder, max_items=16)


    training_args = TrainingArguments(
        output_dir="/workspace/results", 
        evaluation_strategy="steps",
        eval_steps=100,
        logging_dir="/workspace/logs",
        logging_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=1,
        deepspeed="/workspace/codebase/naturalspeech3/dp_config.json"
    )


    # Trainer
    trainer = Trainer(
        model=diffusion,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collate_fn
    )
    # Train
    trainer.train()

if __name__ == '__main__':
    main()
