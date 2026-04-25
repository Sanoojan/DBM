import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from pathlib import Path


class FrameTemporalModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ---- 2D backbone (frame-wise) ----

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # change input from 3 → 2 channels

        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():

            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        backbone.conv1 = new_conv

        # remove FC

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B*T, 512, 1, 1)

        self.feat_dim = 512

        # ---- Temporal Conv ----

        self.temporal = nn.Sequential(

            nn.Conv1d(self.feat_dim, 256, kernel_size=3, padding=1),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),

            nn.ReLU(),

        )

        # ---- Frame-wise classifier ----

        self.classifier = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x):

        # x: (B, 2, T, H, W)

        B, C, T, H, W = x.shape

        # ---- Frame-wise feature extraction ----

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # (B*T, 2, H, W)

        feats = self.backbone(x)                              # (B*T, 512, 1, 1)

        feats = feats.view(B, T, self.feat_dim)               # (B, T, 512)

        # ---- Temporal modeling ----

        feats = feats.permute(0, 2, 1)                        # (B, 512, T)

        temp_out = self.temporal(feats)                       # (B, 128, T)

        logits = self.classifier(temp_out).squeeze(1)         # (B, T)

        return logits  # frame-wise logits
    
    
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1)).view(-1)
    
    
def adapt_vit(model):

    old_proj = model.conv_proj

    new_proj = nn.Conv2d(

        2,

        old_proj.out_channels,

        kernel_size=old_proj.kernel_size,

        stride=old_proj.stride,

        padding=old_proj.padding

    )

    with torch.no_grad():

        new_proj.weight[:] = old_proj.weight.mean(dim=1, keepdim=True)

    model.conv_proj = new_proj

    return model

def adapt_3d_conv(model):

    old_conv = model.stem[0]

    new_conv = nn.Conv3d(

        2,

        old_conv.out_channels,

        kernel_size=old_conv.kernel_size,

        stride=old_conv.stride,

        padding=old_conv.padding,

        bias=old_conv.bias is not None

    )

    with torch.no_grad():

        mean_w = old_conv.weight.mean(dim=1, keepdim=True)

        new_conv.weight.copy_(mean_w.expand_as(new_conv.weight))

    model.stem[0] = new_conv

    return model

def adapt_2d_conv(model):

    old_conv = model.conv1

    new_conv = nn.Conv2d(

        2,

        old_conv.out_channels,

        kernel_size=old_conv.kernel_size,

        stride=old_conv.stride,

        padding=old_conv.padding,

        bias=old_conv.bias is not None

    )

    with torch.no_grad():

        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    model.conv1 = new_conv

    return model

class FrameWrapper(nn.Module):

    def __init__(self, model, feat_dim):

        super().__init__()

        self.model = model

        self.feat_dim = feat_dim

    def forward(self, x):

        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        feats = self.model(x)

        feats = feats.view(B, T, self.feat_dim)

        return feats  # simple temporal pooling
    
class ViTWrapper(nn.Module):

    def __init__(self, model, feat_dim):

        super().__init__()

        self.model = model

        self.feat_dim = feat_dim

    def forward(self, x):

        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        feats = self.model._process_input(x)

        cls = self.model.class_token.expand(feats.shape[0], -1, -1)

        feats = torch.cat([cls, feats], dim=1)

        feats = self.model.encoder(feats)

        feats = feats[:, 0]  # CLS

        feats = feats.view(B, T, self.feat_dim)

        return feats.mean(dim=1)

def build_backbone(name):

    import torchvision.models as models

    import torchvision.models.video as video_models

    # =========================

    # 🔵 3D CNNs (BEST for flow)

    # =========================

    if name == "r3d18":

        model = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)

        feat_dim = model.fc.in_features

        model = adapt_3d_conv(model)

        model.fc = nn.Identity()

        return model, feat_dim

    if name == "mc3_18":

        model = video_models.mc3_18(weights=video_models.MC3_18_Weights.KINETICS400_V1)

        feat_dim = model.fc.in_features

        model = adapt_3d_conv(model)

        model.fc = nn.Identity()

        return model, feat_dim

    if name == "r2plus1d_18":

        model = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.KINETICS400_V1)

        feat_dim = model.fc.in_features

        model = adapt_3d_conv(model)

        model.fc = nn.Identity()

        return model, feat_dim

    # =========================

    # 🟢 2D CNN (frame-based)

    # =========================

    if name == "resnet18":

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        feat_dim = model.fc.in_features

        model = adapt_2d_conv(model)

        model.fc = nn.Identity()

        return FrameWrapper(model, feat_dim), feat_dim

    if name == "resnet50":

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        feat_dim = model.fc.in_features

        model = adapt_2d_conv(model)

        model.fc = nn.Identity()

        return FrameWrapper(model, feat_dim), feat_dim

    # =========================

    # 🟣 Transformer (image)

    # =========================

    if name == "vit_b_16":

        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        feat_dim = model.hidden_dim

        model = adapt_vit(model)

        return ViTWrapper(model, feat_dim), feat_dim

    raise ValueError(f"Unknown backbone: {name}")

class VideoModel(nn.Module):

    def __init__(self, backbone_name="r3d18", num_classes=1, crop=True, resize=None):

        super().__init__()

        self.backbone_name = backbone_name

        self.crop = crop

        self.resize = resize  # e.g., (224,224)

        self.backbone, feat_dim = build_backbone(backbone_name)

        self.classifier = nn.Linear(feat_dim, num_classes)

    def center_crop(self, x):

        B, C, T, H, W = x.shape

        size = min(H, W)

        start_h = (H - size) // 2

        start_w = (W - size) // 2

        return x[:, :, :, start_h:start_h+size, start_w:start_w+size]

    def forward(self, x):

        # x: (B, C, T, H, W)

        if self.crop:

            x = self.center_crop(x)

        if self.resize is not None:

            B, C, T, H, W = x.shape

            x = x.view(B * T, C, H, W)

            x = F.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)

            x = x.view(B, C, T, self.resize[0], self.resize[1])

        B, C, T, H, W = x.shape

        # ---- Try feeding directly (3D case) ----

        try:

            out = self.backbone(x)

        except:

            # fallback to 2D case

            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

            out = self.backbone(x)
        
        
        if self.backbone_name== "resnet50" or self.backbone_name== "resnet18":
            out = self.classifier(out.view(B*T, -1)) # GAP for 2D CNN features
            out = out.view(B, T)
            return out
        
        else:
            out = self.classifier(out.view(B, -1))

        # =========================================================

        # 🔥 HANDLE ALL CASES BASED ON OUTPUT SHAPE

        # =========================================================

        # Case 1: already (B,) or (B,1)
        # breakpoint()
        if out.shape[0] == B:

            return out.view(B)

        # Case 2: (B*T, 1) → frame logits

        if out.shape[0] == B * T and out.shape[-1] == 1:

            out = out.view(B, T)

            return out.mean(dim=1)

        # Case 3: (B*T, D) → features

        if out.shape[0] == B * T:
            if out.ndim == 4:
                out = out.mean(dim=[2, 3])  # GAP
            out = out.view(B, T, -1)       # (B, T, D)
            out = out.mean(dim=1)          # (B, D)
            out = self.head(out)           # (B,1)
            return out.view(B)

        # 🚨 fallback (should not happen)

        raise RuntimeError(f"Unexpected output shape: {out.shape}")  

class CustomR3D18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone= build_r3d18()

    def forward(self, x,return_features=False):
        B, C, T, H, W = x.shape
        crop_size = H  # 128
        start_w = (W - crop_size) // 2
        end_w = start_w + crop_size
        x = x[:, :, :,:, start_w:end_w]

        return self.backbone(x)

def build_r3d18():
    from torchvision.models.video import r3d_18, R3D_18_Weights
    backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    old_conv = backbone.stem[0]
    new_conv = nn.Conv3d(2, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                         padding=old_conv.padding, bias=old_conv.bias is not None)
    with torch.no_grad():
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(mean_w.expand_as(new_conv.weight))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    backbone.stem[0] = new_conv
    backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(backbone.fc.in_features, 1))
    return backbone


class MILAttention(nn.Module):

    def __init__(self, in_dim, hidden_dim=128):

        super().__init__()

        self.attn = nn.Sequential(

            nn.Linear(in_dim, hidden_dim),

            nn.Tanh(),

            nn.Linear(hidden_dim, 1)

        )

    def forward(self, x):

        # x: (B, T, D)

        attn_scores = self.attn(x)          # (B, T, 1)

        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)

        bag_feature = (attn_weights * x).sum(dim=1)       # (B, D)

        return bag_feature, attn_weights.squeeze(-1)
    
    
class MILFrameModel(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # fix 2-channel input

        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():

            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        backbone.conv1 = new_conv

        # remove FC

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 512

        # optional temporal smoothing (light)

        self.temporal = nn.Conv1d(512, 512, kernel_size=5, padding=1)
        self.temporal2 = nn.Conv1d(512, 512, kernel_size=5, padding=1)
        self.temporal3 = nn.Conv1d(512, 512, kernel_size=5, padding=1)
        # MIL attention

        self.mil = MILAttention(512)

        # final classifier

        self.classifier = nn.Linear(512, 1)

        # frame-level classifier (for auxiliary supervision)

        self.frame_classifier = nn.Linear(512, 1)

    def forward(self, x, return_attn=False):

        # x: (B, 2, T, H, W)

        B, C, T, H, W = x.shape

        # ---- frame features ----

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        crop_size = H  # 128

        start_w = (W - crop_size) // 2

        end_w = start_w + crop_size
        x = x[:, :, :, start_w:end_w] 
        feats = self.backbone(x)              # (B*T, 512, 1, 1)

        feats = feats.view(B, T, self.feat_dim)  # (B, T, 512)

        # ---- temporal smoothing ----

        feats_t = self.temporal(feats.permute(0, 2, 1))
        feats_t = self.temporal2(feats_t)
        feats_t = self.temporal3(feats_t).permute(0, 2, 1)  # (B, T, 512)

        # ---- MIL pooling ----

        bag_feat, attn = self.mil(feats_t)    # (B, 512), (B, T)

        # ---- outputs ----

        video_logit = self.classifier(bag_feat).view(-1)      # (B,)

        frame_logits = self.frame_classifier(feats_t).squeeze(-1)  # (B, T)

        if return_attn:

            return video_logit, frame_logits, attn

        return video_logit
    
    
class MILFrameTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # patch embedding projection

        old_proj = vit.conv_proj  # (3 → embed_dim)

        new_proj = nn.Conv2d(

            2,

            old_proj.out_channels,

            kernel_size=old_proj.kernel_size,

            stride=old_proj.stride,

            padding=old_proj.padding,

            bias=old_proj.bias is not None

        )

        # init weights (average RGB → 2ch)

        with torch.no_grad():

            new_proj.weight[:] = old_proj.weight.mean(dim=1, keepdim=True)

        vit.conv_proj = new_proj

        # remove classification head

        self.backbone = vit

        self.feat_dim = vit.hidden_dim  # 768

        self.temporal = nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=5, padding=2)

        self.temporal2 = nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=5, padding=2)

        self.mil = MILAttention(self.feat_dim)

        self.classifier = nn.Linear(self.feat_dim, 1)

        self.frame_classifier = nn.Linear(self.feat_dim, 1)

    def forward(self, x, return_attn=False):

        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # ViT forward → take CLS token

        feats = self.backbone._process_input(x)

        n = feats.shape[0]

        batch_class_token = self.backbone.class_token.expand(n, -1, -1)

        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = self.backbone.encoder(feats)

        feats = feats[:, 0]  # CLS token → (B*T, 768)

        feats = feats.view(B, T, self.feat_dim)

        # temporal smoothing

        feats_t = self.temporal(feats.permute(0, 2, 1))

        feats_t = self.temporal2(feats_t).permute(0, 2, 1)

        bag_feat, attn = self.mil(feats_t)

        video_logit = self.classifier(bag_feat).view(-1)

        frame_logits = self.frame_classifier(feats_t).squeeze(-1)

        if return_attn:

            return video_logit, frame_logits, attn

        return video_logit