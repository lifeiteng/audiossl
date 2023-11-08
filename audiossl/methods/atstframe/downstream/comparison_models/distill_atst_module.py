import torch
import pytorch_lightning as pl
from audiossl.methods.atst.downstream.utils_dcase.comparison_models.frame_atst import FrameATSTLightningModule
from audiossl.models.atst.audio_transformer import AST_base
from audiossl.methods.atst.downstream.utils import load_pretrained_weights
from audiossl.methods.atst.downstream.transform import FreezingTransform
from audiossl.methods.atst.downstream.utils_dcase.model_distill_utils import DistillTeacherModule

class DistillATSTEncoder(pl.LightningModule):
    def __init__(self, distill_mode):
        super().__init__()
        self.n_blocks = 1
        self.embed_dim = 768
        self.distill_mode = distill_mode
        assert distill_mode in ["clip->frame", "frame->clip"], "wrong mode"
        print("Distill mode: ", distill_mode)
        if distill_mode == "clip->frame":
            clip_path = "/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/utils_dcase/comparison_models/ckpts/clip_atst.ckpt"
            frame_path = "/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/frameatst_lr_5e-1_max_epohcs_100_lr_scale_0.75_finetune/checkpoint-epoch=00028.ckpt"
            self.teacher_module = DistillTeacherModule("frame")
            s = torch.load(frame_path, map_location="cpu")
            self.teacher_module.load_state_dict(s["state_dict"])
            self.clip_encoder = AST_base(use_cls=True)
            load_pretrained_weights(self.clip_encoder, clip_path, checkpoint_key="teacher")

        elif distill_mode == "frame->clip":
            clip_path = "/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/dcase_logs/as_strong_407/clipatst_lr_5e-1_max_epohcs_100_lr_scale_0.75_finetune/checkpoint-epoch=00099.ckpt"
            frame_path = "/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/utils_dcase/comparison_models/ckpts/frame_atst.ckpt"
            self.teacher_module = DistillTeacherModule("clip")
            s = torch.load(clip_path, map_location="cpu")
            self.teacher_module.load_state_dict(s["state_dict"])
            self.frame_encoder = get_frame_atst(frame_path)


    def forward(self, x, length):
        x = x.unsqueeze(1)
        if self.distill_mode == "clip->frame":
            x = self.clip_encoder.get_intermediate_layers(
            x,
            length,
            1
            )
        
            x = [item[:, 1:, :] for item in x]
            x = torch.concat(x, dim=-1)

        elif self.distill_mode == "frame->clip":
            x = self.frame_encoder.get_intermediate_layers(
                x,
                length,
                self.n_blocks,
                scene=False
            )
        return x

class DistillATSTPredModule(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self, distill_mode):
        super().__init__()
        self.transform = FreezingTransform(max_len=10)
        self.encoder = DistillATSTEncoder(distill_mode)
        self.embed_dim = self.encoder.embed_dim
        self.distill_mode = distill_mode

    def finetune_mode(self):
        self.freeze()
        # Unfreeze last tfm block
        if self.distill_mode == "frame->clip":
            for n, p in self.encoder.frame_encoder.named_parameters():
                if "mask_embed" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for p in self.encoder.teacher_module.parameters():
                p.requires_grad = False
        else:
            for n, p in self.encoder.clip_encoder.named_parameters():
                if "mask_embed" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for p in self.encoder.teacher_module.parameters():
                p.requires_grad = False


    def finetune_mannual_train(self):
        if self.distill_mode == "frame->clip":
            self.encoder.frame_encoder.train()
            self.encoder.teacher_module.eval()

        elif self.distill_mode == "clip->frame":
            self.encoder.clip_encoder.train()
            self.encoder.teacher_module.eval()
        

    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder(x, length)
        return x, y

def get_frame_atst(pretrained_ckpt_path):
    # get pretrained encoder
    print("Loading frame-atst model:")
    s = torch.load(pretrained_ckpt_path)
    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        pretrained_ckpt_path)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    return pretrained_encoder
