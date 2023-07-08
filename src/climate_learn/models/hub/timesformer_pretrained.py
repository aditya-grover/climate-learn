#Local application
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

#Third Party
import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from transformers import TimesformerConfig, TimesformerModel
from transformers.models.timesformer.modeling_timesformer import TimesformerModel as TrueClassTimesformerModel
from transformers.models.timesformer.modeling_timesformer import BaseModelOutput

@register('timesformer_pretrained')
class TimeSformerPretrained(nn.Module):

    def __init__(self, 
        in_img_size,
        in_channels, 
        out_channels,
        history,
        pretrained_model,
        ckpt_path,
        use_pretrained_weights=False,
        use_pretrained_embeddings=False,
        use_n_blocks=None,
        freeze_backbone=False, 
        freeze_embeddings=False,
        patch_size=2, 
        decoder_depth=1,
        mlp_embed_depth=1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_img_size = in_img_size
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.use_pretrained_weights = use_pretrained_weights
        self.use_n_blocks = use_n_blocks
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.pretrained_model_name = pretrained_model
        self.ckpt_path = ckpt_path

        # initialize timesformer architecture
        config = TimesformerConfig.from_pretrained(pretrained_model)
        config.patch_size = patch_size
        config.num_frames = history
        config.image_size = in_img_size
        config.num_channels = in_channels
        self.timesformer: TrueClassTimesformerModel = TimesformerModel(config)

        if use_pretrained_weights:
            self.load_pretrained_model()

        embed_dim = config.hidden_size
        
        if not use_pretrained_embeddings:
            self.patch_embed = PatchEmbed(in_img_size, patch_size, in_channels, embed_dim)
            # self.pos_embed = nn.Parameter(
            #     torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb,
            # )

            self.mlp_embed = nn.ModuleList()
            for _ in range(mlp_embed_depth):
                self.mlp_embed.append(nn.GELU())
                self.mlp_embed.append(nn.Linear(embed_dim, embed_dim))
            self.mlp_embed = nn.Sequential(*self.mlp_embed)

            self.pos_drop = nn.Dropout(p=0.0)
            self.time_drop = nn.Dropout(p=0.0)

            print('Using new embeddings')

        if self.freeze_embeddings:
            self.patch_embed.requires_grad_(False)
            # self.pos_embed.requires_grad_(False)
            self.mlp_embed.requires_grad_(False)
        
        if self.freeze_backbone:
            for name, param in self.timesformer.encoder.named_parameters():
                if 'layernorm' in name:
                    continue
                param.requires_grad = False
        
        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels*self.patch_size**2))
        self.head = nn.Sequential(*self.head)

        # self.initialize_weights()

    def load_pretrained_model(self):
        # load and process checkpoint
        state_dict = torch.load(self.ckpt_path)
        state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        state_dict = {k[12:]: v for k, v in state_dict.items()}

        # resizing the positional embeddings
        if self.timesformer.embeddings.position_embeddings.size(1) != state_dict['embeddings.position_embeddings'].size(1):
            ckpt_position_embeddings = state_dict['embeddings.position_embeddings']
            cls_pos_embed = ckpt_position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = ckpt_position_embeddings[0, 1:, :].unsqueeze(0).transpose(1, 2)
            patch_num = int(other_pos_embed.size(2) ** 0.5)
            other_pos_embed = other_pos_embed.reshape(1, self.timesformer.embeddings.position_embeddings.size(2), patch_num, patch_num)
            model_pos_shape = (self.in_img_size[0] // self.patch_size, self.in_img_size[1] // self.patch_size)
            print (f"Interpolating positional embeddings from ({patch_num}, {patch_num}) to {model_pos_shape}")
            new_pos_embed = nn.functional.interpolate(other_pos_embed, size=model_pos_shape, mode="bicubic")
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            state_dict['embeddings.position_embeddings'] = new_pos_embed

        # Time Embeddings
        # Resizing time embeddings in case they don't match
        if self.timesformer.config.num_frames != state_dict['embeddings.time_embeddings'].size(1):
            ckpt_time_embeddings = state_dict['embeddings.time_embeddings'].transpose(1, 2)
            orig_len = ckpt_time_embeddings.size(2)
            print (f"Interpolating time embeddings from ({orig_len}) to ({self.timesformer.config.num_frames})")
            new_time_embeddings = nn.functional.interpolate(ckpt_time_embeddings, size=(self.timesformer.config.num_frames), mode="linear")
            new_time_embeddings = new_time_embeddings.transpose(1, 2)
            state_dict['embeddings.time_embeddings'] = new_time_embeddings

        for k in list(state_dict.keys()):
            if k not in self.timesformer.state_dict().keys() or state_dict[k].shape != self.timesformer.state_dict()[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del state_dict[k]
            
        msg = self.timesformer.load_state_dict(state_dict, strict=False)
        print (msg)

    # def initialize_weights(self):
    #     if not self.use_pretrained_embeddings:
    #         pos_embed = get_2d_sincos_pos_embed(
    #             self.pos_embed.shape[-1],
    #             self.in_img_size[0] // self.patch_size,
    #             self.in_img_size[1] // self.patch_size,
    #             cls_token=False,
    #         )
    #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    #     if not self.use_pretrained_weights:
    #         self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = self.out_channels
        h = self.in_img_size[0] // self.patch_size if h is None else h // p
        w = self.in_img_size[1] // self.patch_size if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, self.patch_size, self.patch_size, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * self.patch_size, w * self.patch_size))
        return imgs

    def forward_embedding(self, x):
        # x: B, T, C, H, W
        b, t, c, h, w = x.shape

        embeddings = self.patch_embed(x.flatten(0, 1))
        embeddings = self.mlp_embed(embeddings)

        num_frames = t
        patch_width = w // self.patch_size

        cls_tokens = self.timesformer.embeddings.cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add pos embedding
        embeddings = embeddings + self.timesformer.embeddings.position_embeddings
        embeddings = self.pos_drop(embeddings)

        # add time embedding
        cls_tokens = embeddings[:b, 0, :].unsqueeze(1)
        embeddings = embeddings[:, 1:]
        _, patch_height, patch_width = embeddings.shape
        embeddings = (
            embeddings.reshape(b, num_frames, patch_height, patch_width) # b, t, l, d
            .permute(0, 2, 1, 3) # b, l, t, d
            .reshape(b * patch_height, num_frames, patch_width) # b*l, t, d
        )
        embeddings = embeddings + self.timesformer.embeddings.time_embeddings
        embeddings = self.time_drop(embeddings)
        embeddings = embeddings.view(b, patch_height, num_frames, patch_width).reshape(
            b, patch_height * num_frames, patch_width
        )
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

    def forward_encoder(self, x):
        output_attentions = self.timesformer.config.output_attentions
        output_hidden_states = self.timesformer.config.output_hidden_states
        return_dict = self.timesformer.config.use_return_dict

        embedding_output = self.forward_embedding(x)
        encoder_outputs = self.timesformer.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.timesformer.layernorm is not None:
            sequence_output = self.timesformer.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(self, x):
        # x.shape = [B,T,in_channels,H,W]
        # x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x)['last_hidden_state']
        x = x[:, 1:].reshape(x.shape[0], self.num_patches, self.timesformer.config.num_frames, x.shape[-1])
        x = x.mean(2)
        # x.shape = [B, num_patches, embed_dim]
        x = self.head(x)
        # x.shape = [B, num_patches, out_channels*patch_size**2]
        x = self.unpatchify(x)
        # x.shape = [B, out_channels, H, W]
        return x
