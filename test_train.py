from naturalspeech3 import NaturalSpeech3
from dataloader import custom_collate_fn, CustomDataset
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import torch
from transformers import Trainer, TrainingArguments



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

device = torch.device("cpu")

def main():
    # Initialize NaturalSpeech2
    diffusion = NaturalSpeech3(
        facodec_encoder=fa_encoder,
        facodec_decoder=fa_decoder,
    )

    diffusion = diffusion.to(device)

    # Dataset
    train_dataset_folder = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/"
    train_dataset = CustomDataset(dataset_folder=train_dataset_folder, max_items=10)


    val_dataset_folder = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_val/"
    val_dataset = CustomDataset(dataset_folder=val_dataset_folder, max_items=5)


    # Training Arguments (CPU enforced)
    training_args = TrainingArguments(
        output_dir="./results",       # Directory to save checkpoints
        evaluation_strategy="steps", # Evaluation strategy
        eval_steps=100,              # Evaluate every 100 steps
        logging_dir="./logs",        # Log directory
        logging_steps=10,
        num_train_epochs=3,          # Number of epochs
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=500,              # Save checkpoints every 500 steps
        save_total_limit=2,          # Keep only the last 2 checkpoints
        no_cuda=True,                # Enforce CPU usage
        per_gpu_train_batch_size=2,
        dataloader_num_workers=1,
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
