from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from torch.nn import MSELoss
import torch.nn.functional as F
class SDTrackActor(BaseActor):
    """ Actor for training SDTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        with_bbox = self.cfg.TRAIN.BBOX_TASK
        with_mask = getattr(self.cfg.TRAIN, 'MASK_TASK', False)
        with_language = getattr(self.cfg.TRAIN, 'LANGUAGE_TASK', False)

        # forward pass
        out_dict = self.forward_pass(data, with_bbox=with_bbox, with_mask=with_mask, with_language=with_language)


        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data, with_bbox=False, with_mask=False, with_language=False):


        template_list, search_list = [], []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img_1 = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)


        _, b, _, _, _ = data['template_images'].shape
        text_data = NestedTensor(data['nl_token_ids'].reshape(b, -1), data['nl_token_masks'].reshape(b, -1))
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])  # (B, 64): center point = 1

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])



        if with_bbox and with_mask:
            out_dict = self.net(template=template_list,
                                search=search_list,
                                search_anno=data['search_anno'],
                                search_attn_mask=data['search_att'][0],  # attention mask
                                search_segmask_vertices=data['search_mask_vertices'][0], # segmentation mask vertices
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        elif with_mask:
            out_dict = self.net(template=template_list,
                                search=search_list,
                                search_attn_mask=data['search_att'][0],
                                search_segmask_vertices=data['search_mask_vertices'][0],
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        elif with_bbox and with_language:
            out_dict = self.net(template=template_list,
                                search=search_list,
                                search_anno=data['search_anno'],
                                search_attn_mask=data['search_att'][0],
                                text=text_data,               # language descriptions
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        elif with_bbox:
            out_dict = self.net(template=template_list,
                                search=search_list,
                                search_anno=data['search_anno'],
                                search_attn_mask=data['search_att'][0],
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        # #muyh 24/9/17

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        #gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_bbox = gt_dict['search_anno'].view(-1,4)
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        num_queries = pred_boxes.size(1)
        if torch.isnan(pred_boxes).any():
            import warnings
            warnings.warn("Detected NaN in pred_boxes, replacing with GT bbox")
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                      max=1.0)
            pred_boxes = torch.where(torch.isnan(pred_boxes), gt_boxes_vec.view(pred_boxes.shape), pred_boxes)

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        if torch.isnan(location_loss):
            import warnings
            warnings.warn("location_loss is NaN! Setting to zero and clamping gradients.")
            location_loss = torch.tensor(0.0, device=location_loss.device, requires_grad=False)


        mid_loss = self.compute_distill_loss(pred_dict)

        if torch.isnan(mid_loss) or torch.isinf(mid_loss):
            import warnings
            warnings.warn("Distill loss is invalid. Setting to 0.")
            mid_loss= torch.tensor(0.0, device=location_loss.device, requires_grad=False)
            # a = 0

        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss +mid_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item(),
                      "disloss":mid_loss.item()}
            return loss, status
        else:
            return loss

    def compute_distill_loss(self, pred_dict):

        teacher_hidden = pred_dict['teacher_layer_mid']
        student_hidden = pred_dict['student_layer_mid']
        teacher_atts = pred_dict['teacher_attention_mid']
        student_atts = pred_dict['student_attention_mid']
        teacher_text = pred_dict['teacher_text']
        student_text = pred_dict['student_text']
        att_score_map = pred_dict['att_score']


        device = teacher_atts[0].device
        loss_mse = MSELoss()
        att_loss = 0.
        rep_loss = 0.
        end_loss = 0.
        # 对齐层数：假设教师12层，学生4层，则每3层选1层
        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]

        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                      teacher_att)
            att_loss += loss_mse(student_att, teacher_att)

        new_teacher_reps = [teacher_hidden[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_hidden

        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            rep_loss += loss_mse(student_rep, teacher_rep)

        # end_loss = loss_mse(teacher_text, student_text)#(64,40) (64,40,512) (64,40,512)
        end_loss = self.weighted_mse_loss(teacher_text, student_text, att_score_map)

        return rep_loss + att_loss +end_loss


    def weighted_mse_loss(self, teacher, student, weights):
        # 确保权重维度正确（例如 [batch_size, seq_len, 1]）
        if weights.dim() == 2:
            weights = weights.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]

        # 对权重进行预处理，防止极端值导致的梯度不稳定
        # 使用softmax替代简单归一化，确保权重分布更均匀
        weights = F.softmax(weights, dim=1)  # 在seq_len维度上进行softmax

        # 计算平方差
        squared_diff = (teacher - student) ** 2  # [batch_size, seq_len, feature_dim]

        # 应用权重
        weighted_squared_diff = squared_diff * weights

        # 按batch维度聚合损失，使用mean而非sum避免梯度爆炸
        loss_per_token = weighted_squared_diff.mean(dim=2)  # [batch_size, seq_len]

        # 对每个token的损失进行裁剪，防止异常值影响
        loss_per_token = torch.clamp(loss_per_token, max=10.0)  # 限制最大损失值

        # 最终聚合损失，使用mean保持梯度稳定
        final_loss = loss_per_token.mean()

        return final_loss