"""
 ------------------------------------------------------------------------
 Modified from ControlNet (https://github.com/lllyasviel/ControlNet)
 ------------------------------------------------------------------------
"""

import einops
import torch
import torch as th
import torch.nn as nn

from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config, exists, default
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from ldm.modules.pmsa_vit import PMSAViT
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepBlock, ResBlock, Downsample, AttentionBlock
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F


class StablePose(LatentDiffusion):
    def __init__(self, gaussian_kernels, control_config=None, alpha=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stable_pose_adapter = instantiate_from_config(control_config)
        self.gaussian_kernels = gaussian_kernels
        self.alpha = alpha # coefficient in the proposed pose-mask guided loss

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if len(batch['hint']) != 0:
            pose_condition = batch['hint']
            if bs is not None:
                pose_condition = pose_condition[:bs]
            pose_condition = pose_condition.to(self.device)
            pose_condition = einops.rearrange(pose_condition, 'b h w c -> b c h w')
            pose_condition = pose_condition.to(memory_format=torch.contiguous_format).float()
        else:
            pose_condition = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()
        
        # pose mask
        masks = None
        for k in self.gaussian_kernels:
            pose_image = torch.cat([pose_condition], 1)
            tmp_masks = torch.zeros((batch['hint'].shape[0], 64,64))
            blur = GaussianBlur(kernel_size=(k, k), sigma=3)
            pose_image_blured = blur(pose_image)
            for i, pose in enumerate(pose_image_blured):
                _, h_idx, w_idx = torch.where(pose>-1)
                h_idx, w_idx = h_idx//8, w_idx//8
                tmp_masks[i][h_idx, w_idx] = 1
            if masks is None:
                masks = tmp_masks
            else:
                masks = torch.cat((masks, tmp_masks), 0)
        masks.requires_grad = False
        
        return x, dict(c_crossattn=[c], pose_condition=[pose_condition], pose_mask=masks)

    def apply_model(self, x_noisy, t, cond, global_strength=1, *args, **kwargs):
        assert isinstance(cond, dict)
        if 'pose_mask' not in cond.keys():
            cond['pose_mask'] = None

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        assert cond['pose_condition'][0] != None
        pose_condition = torch.cat(cond['pose_condition'], 1)
        pose_control = self.stable_pose_adapter(x=x_noisy, timesteps=t, context=cond_txt, pose_condition=pose_condition, 
                                            mask=cond['pose_mask'])
        
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, pose_control=pose_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    # reweighted loss for pose region
    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, **kwargs)
        
        masks = cond['pose_mask']
        if masks.shape[0] != noise.shape[0]:
            masks = torch.split(masks, noise.shape[0])[-1] # finer mask for loss
        masks = masks.unsqueeze(1).repeat(1,4,1,1).to(noise.device)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False)
        # reweighted pose region
        loss_simple = (torch.mul(loss_simple, masks)*self.alpha + torch.mul(loss_simple, 1-masks)).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat = c["pose_condition"][0][:N]
        c = c["c_crossattn"][0][:N]
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["pose_image"] = rearrange(batch['hint'], 'b h w c -> b c h w')
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond={"pose_condition": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"pose_condition": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"pose_condition": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        
        # add pose to samples
        log_add = {}
        for key,image in log.items():
            if key.startswith("samples"):
                image_draw = torch.nn.functional.interpolate(
                    image,
                    size=log["pose_image"].shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
                draw_skeleton_mask=(torch.sum(log["pose_image"],1)<=(-1.+1e-3)).unsqueeze(1).repeat((1,3,1,1)).float()
                posed_image=(1-draw_skeleton_mask)*log["pose_image"]+draw_skeleton_mask*image_draw
                log_add[key.replace("samples","pose_samples")] = posed_image
        
        for key,item in log_add.items():
            log[key]=item

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        _, _, h, w = cond["pose_condition"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params += list(self.stable_pose_adapter.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.stable_pose_adapter = self.stable_pose_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.stable_pose_adapter = self.stable_pose_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class AdaptedTimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, pose_features=None, mask=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, AttnResBlock):
                x = layer(x, emb, pose_features, mask)
            else:
                x = layer(x)
        return x


class AttnResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        cond_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        

        self.attn_layer = PMSAViT(image_size=64, patch_size=2, dim=512, depth=2, heads=8, mlp_dim=512, 
                                    channels=self.out_channels, dim_head=self.out_channels//8)
        
        # added
        self.norm = nn.GroupNorm(32, self.out_channels, affine=False)
        self.cond_conv = conv_nd(dims, cond_channels, self.out_channels, 3, padding=1)
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, pose_condition, mask):
        return checkpoint(
            self._forward, (x, emb, pose_condition, mask), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, pose_condition, mask):
        h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.attn_layer(h, mask)
        
        h = self.norm(h)
        h += self.cond_conv(pose_condition)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class FeatureExtractor(nn.Module):
    def __init__(self, cond_channels, inject_channels, dims=2):
        super().__init__()
        self.pre_extractor = AdaptedTimestepEmbedSequential(
            conv_nd(dims, cond_channels, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.extractors = nn.ModuleList([
            AdaptedTimestepEmbedSequential(
                conv_nd(dims, 128, inject_channels[0], 3, padding=1, stride=2),
                nn.SiLU()
            ),
        ])

        self.zero_convs = nn.ModuleList([
            zero_module(conv_nd(dims, inject_channels[0], inject_channels[0], 3, padding=1)),
        ])

    
    def forward(self, pose_condition):
        pose_features = self.pre_extractor(pose_condition, None)
        assert len(self.extractors) == len(self.zero_convs)
        
        output_features = []
        for idx in range(len(self.extractors)):
            pose_features = self.extractors[idx](pose_features, None)
            output_features.append(self.zero_convs[idx](pose_features))
        return output_features


class StablePoseAdapter(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            cond_channels,
            inject_channels,
            inject_layers,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.inject_layers = inject_layers
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.feature_extractor = FeatureExtractor(cond_channels, inject_channels)
        self.input_blocks = nn.ModuleList(
            [
                AdaptedTimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                if (1 + 3*level + nr) in self.inject_layers:
                    layers = [
                        AttnResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            cond_channels=inject_channels[level]
                        )
                    ]
                else:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(AdaptedTimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    AdaptedTimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = AdaptedTimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return AdaptedTimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, timesteps, context, pose_condition, mask=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        pose_features = self.feature_extractor(pose_condition.cuda())

        outs = []
        h = x.type(self.dtype)
        for layer_idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if layer_idx in self.inject_layers:
                h = module(h, emb, context, pose_features[self.inject_layers.index(layer_idx)], mask)
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlUNetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, pose_control=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        h += pose_control.pop()

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop() + pose_control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)
