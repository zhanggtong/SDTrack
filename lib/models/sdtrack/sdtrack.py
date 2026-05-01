"""
Basic SDTrack model.
"""
import math
import os
import re
from typing import List
import torch.nn.functional as F
import torch
from pyasn1.codec.ber.encoder import NullEncoder
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.sdtrack.hivit import hivit_small, hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.prompt_dec import build_prompt_dec
from lib.models.layers.position_encoding import build_position_encoding
from lib.utils.misc import NestedTensor

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from lib.models.language_model import build_bert, build_tiny_bert

from functools import reduce
from operator import mul

# from mamba_ssm import Mamba
from .cross_attension import CrossAttention, SelfAttention
import torch




class SDTrack(nn.Module):
    """ This is the base class for SDTrack """

    # def __init__(self, transformer, box_head, transformer_dec, prompt_dec, position_encoding, aux_loss=False,
    #              head_type="CORNER", tokenizer=None, text_encoder=None):
    def __init__(self, transformer, box_head, transformer_dec, prompt_dec, position_encoding, aux_loss=False, head_type="CORNER", tokenizer=None, text_encoder=None ,student_text_encoder=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.position_encoding = position_encoding
        self.query_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)

        # text encoder
        self.language_backbone = text_encoder
        self.student_language_backbone = student_text_encoder

        self.text_proj = nn.Linear(768, 512)
        # self.student_text_proj = nn.Linear(312, 512)


        # initiate prompt:
        val = math.sqrt(6. / float(1 * reduce(mul, [1, 1], 1) + 512))  # noqa
        self.prompt_q = nn.Parameter(torch.zeros(
            1, 1, 512))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_q.data, -val, val)
        self.prompt_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)
        self.prompt_proj = nn.Linear(512, 768)
        self.cross_att = CrossAttention(dim=512)
        self.self_att1 = SelfAttention(512, 512, 512)

        self.last_nlp_text = None



    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text=None,
                training=True,  # True
                reverse_pre=None,
                tgt_pre=None,
                nlp = None,
                path = None,
                search_attn_mask=None,
                search_anno=None,
                search_segmask_vertices=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):


        b0, num_search = template[0].shape[0], len(search)

        if not training:
            b0, num_search = template.shape[0], len(search)

        if training:
            search = torch.cat(search, dim=0)
            template = template[0].repeat(num_search, 1, 1, 1)

        # get visual feature
        z_patch, x_patch = self.backbone.patch(z=template, x=search)

        teacher_layer_mid = None
        teacher_attention_mid = None
        teacher_text = None
        att_score = None

        vis = z_patch
        if training:
            text_fea, all_encoder_layers, layer_atts = self.language_backbone(text)
        student_text_fea, student_all_encoder_layers, student_layer_atts = self.student_language_backbone(text)

        # text_tokens = nlp.split()
        # text_tokens = ['[CLS]'] + text_tokens + ['[SEP]']


        if training:
            # text_fea = text_fea[:, 1:]# 1,40,768
            text_fea = text_fea.repeat(num_search, 1, 1)
            text_fea = self.text_proj(text_fea)  # 1 40 512
            teacher_text= text_fea
            concatenated_tensor = torch.cat((text_fea, vis), dim=1)
            text_vis, text_vis_atts = self.self_att1(concatenated_tensor)
            text_self_attn = text_vis_atts[:, :40, :40]
            text_vis_attn = text_vis_atts[:, 40:, :40]
            text_self_attn = text_self_attn.transpose(1, 2)
            text_vis_attn = text_vis_attn.transpose(1, 2)
            adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=40)
            avg_pooled = adaptive_avg_pool(text_vis_attn)
            deltas = avg_pooled + text_self_attn

            min_val = deltas.min()
            max_val = deltas.max()
            if max_val == min_val:
                deltas = deltas
            else:
                deltas = (deltas - min_val) / (max_val - min_val)
            adaptive_avg_pool1 = nn.AdaptiveAvgPool1d(output_size=1)
            pooled_score = adaptive_avg_pool1(deltas)
            att_score= pooled_score


        # student_text_fea = student_text_fea[:, 1:]
        student_text_fea = self.text_proj(student_text_fea)  # 1 40 512

        # text_fea = self.text_proj(text_fea)

        student_text = student_text_fea.repeat(num_search, 1, 1)




        if training:
            teacher_layer_mid = all_encoder_layers
            teacher_attention_mid = layer_atts

        student_layer_mid = student_all_encoder_layers
        student_attention_mid = student_layer_atts



        x, aux_dict = self.backbone(z=z_patch, x=x_patch, text=student_text,
                                    return_last_attn=return_last_attn, )  # x=[B,N,C]

        feat_last = x

        # Forward head
        out = self.forward_head(feat_last, out_dec=None, gt_score_map=None)
        out.update(aux_dict)
        out['tgt'] = tgt_pre
        out['reverse'] = reverse_pre

        out['teacher_layer_mid'] = teacher_layer_mid
        out['student_layer_mid'] = student_layer_mid
        out['teacher_attention_mid'] = teacher_attention_mid
        out['student_attention_mid'] = student_attention_mid
        out['teacher_text'] = teacher_text
        out['student_text'] = student_text
        out['att_score'] = att_score

        return out

    def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # STM
        enc_opt = cat_feature[:, -self.feat_len_s:]
        # dec_opt = out_dec.transpose(0,1).transpose(1,2)
        # att = torch.matmul(enc_opt, dec_opt)
        # opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        # Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_sdtrack(cfg, training=True):
    # Build Text Encoder
    tokenizer, text_encoder = None, None
    if cfg.MODEL.TEXT_ENCODER == 'roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained(
            'pretrained_models/roberta-base')  # load pretrained RoBERTa Tokenizer
        text_encoder = RobertaModel.from_pretrained('pretrained_models/roberta-base')  # load pretrained RoBERTa model
    elif cfg.MODEL.TEXT_ENCODER == 'bert-base':
        tokenizer = None
        if training:
            text_encoder= build_bert()
        # else:
        #     text_encoder = None
        student_text_encoder = build_tiny_bert()
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('SDTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    transformer_dec = build_transformer_dec(cfg, hidden_dim)
    prompt_dec = build_prompt_dec(cfg, hidden_dim)
    position_encoding = build_position_encoding(cfg, sz=1)

    box_head = build_box_head(cfg, hidden_dim)
    model = SDTrack(
        backbone,
        box_head,
        transformer_dec,
        prompt_dec,
        position_encoding,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        student_text_encoder =student_text_encoder
    )

    if 'SDTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
