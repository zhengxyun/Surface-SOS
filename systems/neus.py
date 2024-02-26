import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy

from PIL import Image
import numpy as np
import cv2
import os

@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation 
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)

        rgb_full = rgb
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'rgb_full': rgb_full
        })      
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh

        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [  # -{'%.3f' % (psnr)}
            {'type': 'rgb', 'img': batch['rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': batch['fg_mask'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}  # white_bgr
            # {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ])

        # out_np = out['comp_rgb_full'].view(H, W, 3).cpu().numpy()
        # rgb_bg_out_np = out['comp_rgb_bg'].view(H, W, 3).numpy()
        # out_bbk_np = out['comp_rgb'].view(H, W, 3).cpu().numpy()
        # opa_np = out['opacity'].view(H, W).cpu().numpy()
        # dep_np = out['depth'].view(H, W).cpu().numpy()
        # nrm_np = out['comp_normal'].view(H, W, 3).cpu().numpy()
        # rgb_np = batch['rgb'].view(H, W, 3).cpu().numpy()
        # rgb_full_np = batch['rgb_full'].view(H, W, 3).cpu().numpy()
        # com_np = opa_np[:, :, np.newaxis] * rgb_np  # + bgr_np * (1-opa_np[:,:,np.newaxis])

        # self.save_rgb_image(f"it{self.global_step}-val/out/{batch['index'][0].item()}.png", out_np, 'HWC', (0, 1))
        # self.save_rgb_image(f"it{self.global_step}-val/out_bbk/{batch['index'][0].item()}.png", out_bbk_np, 'HWC',(0, 1))
        # self.save_grayscale_image(f"it{self.global_step}-val/opa/{batch['index'][0].item()}.png", opa_np, (0, 1), None)
        # self.save_grayscale_image(f"it{self.global_step}-val/dep/{batch['index'][0].item()}.png", dep_np, (0, 1), None)
        # self.save_rgb_image(f"it{self.global_step}-val/nrm/{batch['index'][0].item()}.png", nrm_np, 'HWC', (-1, 1))
        # self.save_rgb_image(f"it{self.global_step}-val/com/{batch['index'][0].item()}.png", com_np, 'HWC', (0, 1))
        # self.save_rgb_image(f"it{self.global_step}-val/bg/{batch['index'][0].item()}.png", rgb_bg_out_np, 'HWC', (0, 1))

        # # save_res_image
        # self.save_res_image(f"res_neus/{self.global_step}/alpha/{batch['index'][0].item()}.png", opa_np, (0, 1), None)
        # img_rgba = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2RGBA)
        # opa = opa_np[:,:, np.newaxis]
        # # print("opa:", opa.shape)
        #
        # fgr_rgba = img_rgba * opa
        # fgr_rgba_pil = Image.fromarray(np.clip(fgr_rgba * 255, 0, 255).astype(np.uint8))
        # out_fgr_pth = os.path.join(self.config.res_dir, f"res_neus/{self.global_step}/com_rgba")
        # os.makedirs(out_fgr_pth, exist_ok=True)
        # fgr_rgba_pil.save(os.path.join(out_fgr_pth,f"{batch['index'][0].item()}.png"))
        #
        # comp_rgba = cv2.cvtColor(out_bbk_np, cv2.COLOR_RGB2RGBA)
        # comp_pth = os.path.join(self.config.res_dir, f"res_neus/{self.global_step}/comp")
        # os.makedirs(comp_pth, exist_ok=True)
        # comp_pil = Image.fromarray(np.clip(comp_rgba * 255, 0, 255).astype(np.uint8))
        # comp_pil.save(os.path.join(comp_pth, f"{batch['index'][0].item()}.png"))
        #
        # # save bgr
        # out_bgr = cv2.cvtColor(rgb_bg_out_np, cv2.COLOR_RGB2RGBA)
        # bgr_pth = os.path.join(self.config.res_dir, f"res_neus/{self.global_step}/bgr")
        # os.makedirs(bgr_pth, exist_ok=True)
        # bgr_pil = Image.fromarray(np.clip(out_bgr * 255, 0, 255).astype(np.uint8))
        # bgr_pil.save(os.path.join(bgr_pth, f"{batch['index'][0].item()}.png"))


        return {
            'psnr': psnr,
            'index': batch['index']
        }

    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)
        
        self.export()

    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        
