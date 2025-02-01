import os
import tempfile

from naturalspeech3 import NaturalSpeech3
from dataloader import custom_collate_fn, CustomDataset
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import torch
from transformers import Trainer, TrainingArguments

os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["TEMP"] = "/workspace/tmp"
os.environ["TMP"] = "/workspace/tmp"
tempfile.tempdir = "/workspace/tmp"
tempfile.gettempdir()

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

for p in fa_encoder.parameters():
    p.requires_grad = False
for p in fa_decoder.parameters():
    p.requires_grad = False


def compute_metrics(eval_pred, compute_result=True):
    print("DEBUG: compute_metrics function called!")  # Debugging log
    
    if eval_pred is None:
        print("Warning: eval_pred is None")
        return {}

    predictions, _ = eval_pred

    metrics = {
        "duration_loss": predictions[0].mean(),
        "prosody_beta_loss": predictions[1].mean(),
        "prosody_loss": predictions[2].mean(),
        "content_loss": predictions[3].mean(),
        "acoustic_loss": predictions[4].mean(),
        "speech_loss": predictions[5].mean()
    }

    if compute_result:
        return metrics
    else:
        return {}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert model weights to bfloat16
    fa_encoder.to(device, dtype=torch.bfloat16)
    fa_decoder.to(device, dtype=torch.bfloat16)

    diffusion = NaturalSpeech3(
        facodec_encoder=fa_encoder,
        facodec_decoder=fa_decoder,
    ).to(device, dtype=torch.bfloat16)

    # diffusion_ckpt = "/workspace/results/checkpoint-1000"
    # diffusion.load_state_dict(torch.load(diffusion_ckpt))

    # Dataset
    train_dataset_folder = "/workspace/datasets/LibriTTS_R-360-Train-new/"
    train_dataset = CustomDataset(dataset_folder=train_dataset_folder)


    val_dataset_folder = "/workspace/datasets/VCTK_train/"
    val_dataset = CustomDataset(dataset_folder=val_dataset_folder)


    training_args = TrainingArguments(
        output_dir="/workspace/results",
        logging_dir="/workspace/logs",
        eval_strategy="steps",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        bf16=True,
        fp16=False,
        num_train_epochs=100,
        save_steps=871,
        eval_steps=871,
        logging_steps=10,
        dataloader_num_workers=8,
        learning_rate=float(1e-6),
        warmup_steps=1000,
        save_total_limit=3,
        optim="adamw_torch",
        deepspeed="/workspace/codebase/naturalspeech3/dp_config.json",
        lr_scheduler_type="cosine_with_restarts",
        per_device_train_batch_size=2,
        batch_eval_metrics=True,
        report_to="wandb",
        run_name="natspeech3_test_30"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=diffusion,
        data_collator=custom_collate_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train(resume_from_checkpoint="/workspace/results/checkpoint-4356")

if __name__ == '__main__':
    main()
