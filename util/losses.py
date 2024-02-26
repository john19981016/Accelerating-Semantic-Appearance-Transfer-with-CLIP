from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F
import clip
from models.extractor import VitExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossG(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)
        self.clip, self.clip_preprocess = clip.load("ViT-B/32")
        
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)
        self.clip_resize_transform = Resize([224,224])
        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])

        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_global_ssim=0,
            lambda_entire_ssim=0,
            lambda_entire_cls=0,
            lambda_global_identity=0
        )

    def update_lambda_config(self, step):
        if step == self.cfg['cls_warmup']:
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']

        if step % self.cfg['entire_A_every'] == 0:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_entire_cls'] = self.cfg['lambda_entire_cls']
            self.lambdas['lambda_entire_clip'] = self.cfg['lambda_entire_clip']
            self.lambdas['lambda_local_clip'] = self.cfg['lambda_local_clip']
        else:
            self.lambdas['lambda_entire_ssim'] = 0
            self.lambdas['lambda_entire_cls'] = 0
            self.lambdas['lambda_entire_clip'] = 0
            self.lambdas['lambda_local_clip'] = self.cfg['lambda_local_clip']

    def forward(self, outputs, inputs, text, local_text):
        self.update_lambda_config(inputs['step'])
        clip_local_threshold = min(0+inputs['step']*0.001, 0)
        losses = {}
        loss_G = 0
        if self.lambdas['lambda_entire_clip'] > 0:
            inputs['text_entire'] = clip.tokenize(['This is the '+text]).cuda()
            with torch.no_grad():
                text_features = self.clip.encode_text(inputs['text_entire'])
            imageA_features = self.clip.encode_image(self.clip_resize_transform(outputs['x_entire']))
            imageA_features = imageA_features / imageA_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            losses['loss_entire_clip'] = -F.cosine_similarity(imageA_features, text_features).mean()
            loss_G = loss_G + self.lambdas['lambda_entire_clip'] * losses['loss_entire_clip']
        
        if self.lambdas['lambda_local_clip'] > 0:
            inputs['text_local'] = clip.tokenize(['Part of the '+local_text]).cuda()
            with torch.no_grad():
                text_features = self.clip.encode_text(inputs['text_local'])
            imageA_features = self.clip.encode_image(outputs['x_local'])
            imageA_features = imageA_features / imageA_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            losses['loss_local_clip'] = -F.cosine_similarity(imageA_features, text_features).mean()
            if losses['loss_local_clip']<-clip_local_threshold:
                loss_G = loss_G + self.lambdas['lambda_local_clip'] * losses['loss_local_clip']
            
        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
            loss_G = loss_G + losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

        if self.lambdas['lambda_entire_ssim'] > 0:
            losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs['x_entire'], inputs['A'])
            loss_G = loss_G + losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

        if self.lambdas['lambda_entire_cls'] > 0:

            losses['loss_entire_cls'] = self.calculate_crop_cls_loss(outputs['x_entire'], inputs['B_global'])
            loss_G = loss_G + losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']



        if self.lambdas['lambda_global_cls'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
            loss_G = loss_G + losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        if self.lambdas['lambda_global_identity'] > 0:
            losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
            loss_G = loss_G + losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss

