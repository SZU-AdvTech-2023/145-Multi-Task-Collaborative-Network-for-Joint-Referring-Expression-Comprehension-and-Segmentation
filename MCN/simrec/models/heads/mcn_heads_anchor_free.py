# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from simrec.layers.aspp import aspp_decoder
from ..utils.box_op import bboxes_iou

from simrec.layers.activation import get_activation

from ..utils.box_op import bboxes_iou
from ..losses.iou_loss import IOUloss


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


"""
改为anchor-free方法
"""

class MCNhead_anchor_free(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    refine anchor-bash to anchor-free
    """

    def __init__(
            self,
            hidden_size=512,
            label_smooth=0.0,
            num_classes=0,
            width=1.0,
            strides=[32, ],
            in_channels=[512, ],
            act="silu",
            depthwise=False,
    ):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(MCNhead_anchor_free, self).__init__()

        self.label_smooth = label_smooth
        self.n_anchors = 1  # 每个grid只有一个anchor
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()  # 空
        self.reg_convs = nn.ModuleList()  # depthwise为False时，[(3*3)卷积，批标准化，(silu)激活，(3*3)卷积，批标准化，(silu)激活] 将特征图大小不变，特征通道数不变
        self.cls_preds = nn.ModuleList()  # 空
        self.reg_preds = nn.ModuleList()  # [(1*1)卷积] 将图像特征通道变为 4
        self.obj_preds = nn.ModuleList()  # [(1*1)卷积] 将图像特征通道变为 1
        self.stems = nn.ModuleList()  # [(1*1)卷积，批标准化，(silu)激活] 将图像特征通道变为 256*width

        # self.seg_preds = nn.ModuleList()

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):  # 该程序中只有一层循环
            self.stems = nn.ModuleList(  # TODO 当in_channels被指定为长度>1的列表时，会发生重复赋值self.stems，可能有问题
                [BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                ), ]
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )


        self.stride = strides[0]  # strides[layer_no]

        # TODO 由于这里暂时设定只有一层in_channel,为方便，暂时直接取 in_channels[0]
        self.d_proj = nn.Conv2d(in_channels[0], 1, kernel_size=3, padding=1)
        self.s_proj = nn.Conv2d(in_channels[0], 1, kernel_size=3, padding=1)

        self.sconv = nn.Sequential(aspp_decoder(in_channels[0], hidden_size // 2, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=8)
                                   )
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def nls(self, pred_seg, pred_box, weight_score=None, lamb_au=-1., lamb_bu=2, lamb_ad=1., lamb_bd=0):
        if weight_score is not None:
            # asnls
            mask = torch.ones_like(pred_seg) * weight_score.unsqueeze(1).unsqueeze(1) * lamb_ad + lamb_bd
            pred_box = pred_box[:, :4].long()
            for i in range(pred_seg.size()[0]):
                mask[i, pred_box[i, 1]:pred_box[i, 3] + 1, pred_box[i, 0]:pred_box[i, 2] + 1, ...] = weight_score[
                                                                                                         i].item() * lamb_au + lamb_bu
        else:
            # hard-nls
            mask = torch.zeros_like(pred_seg)
            pred_box = pred_box[:, :4].long()
            for i in range(pred_seg.size()[0]):
                mask[i, pred_box[i][1]:pred_box[i][3] + 1, pred_box[i][0]:pred_box[i][2] + 1] = 1.
        return pred_seg * mask

    def co_energe(self, x_map, y_map, x_attn, y_attn, eps=1e-6):
        """
        :param x_map:  h*w
        :param y_map: h*w
        :param x_attn: B,c,h,w
        :param y_attn: B,c,h,w
        :return:
        """
        b, c, h, w = x_attn.size()
        # TODO 这一步好像是多余的，在前面的garan_attention层已经计算，这里重复了
        x_map = F.softmax(self.s_proj(x_attn).view(b, -1), -1)  # (b, 1, h_x*w_x)
        y_map = F.softmax(self.d_proj(y_attn).view(b, -1), -1)  # (b, 1, h_y*w_y)

        x_attn = F.normalize(x_attn, dim=1).view(b, c, -1)  # (b, c, h_x*w_x)
        y_attn = F.normalize(y_attn, dim=1).view(b, c, -1)  # (b, c, h_y*w_y)
        cosin_sim = torch.bmm(x_attn.transpose(1, 2), y_attn)  # b, h_x*w_x, h_y*w_y
        cosin_sim = (cosin_sim + 1.) / 2.
        co_en = torch.einsum('blk,bl,bk->b', [cosin_sim, x_map, y_map])
        return -torch.log(co_en + eps)

    def forward(self, xin, yin, x_label=None, y_label=None, x_map=None, y_map=None, x_attn=None, y_attn=None):

        # 1 卷积, assp+上采样
        # output = self.dconv(xin)  # (b, 3*5, h, w)

        outputs = []  # TODO 观察最后REChead层如何进行前向传播并输出最后结果
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        # 将yolo型的相对坐标标签[xc, yc, w, h]转换为经过缩放和pad的正方形上的绝对坐标标签
        if x_label is not None:
            x_label *= xin.size()[-1] * self.strides[0]  # 标签值 * xin最后一个维度(卷积后的像素长度) * strides([32,]) 将标签放大到传入网络图片的值
        if isinstance(xin, torch.Tensor):
            xin = [xin]  # 将类型为Tensor的xin放入python list中，下面统一处理list类型
        for k, (reg_conv, stride_this_level, x) in enumerate(
                zip(self.reg_convs, self.strides, xin)
        ):  # 该程序中只有一层
            x = self.stems[k](x)  # [(1*1)卷积，批标准化，(silu)激活] 将图像特征通道变为 256*width
            reg_x = x
            reg_feat = reg_conv(reg_x)  # [(3*3)卷积，批标准化，(silu)激活，(3*3)卷积，批标准化，(silu)激活] 将特征图大小不变，特征通道数不变
            reg_output = self.reg_preds[k](reg_feat)  # [(1*1)卷积] 将图像特征通道变为 4  【相对grid的xc,yc,w,h】
            obj_output = self.obj_preds[k](reg_feat)  # [(1*1)卷积] 将图像特征通道变为 1

            if self.training:
                output = torch.cat([reg_output, obj_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )  # 将特征图上计算出的相对每个grid的xc yc w h值，还原为绝对值且放大到输入网络图片大小的尺度 【绝对xc,yc,w,h】
                x_shifts.append(grid[:, :, 0])  # [(1,169)]
                y_shifts.append(grid[:, :, 1])  # [(1,169)]
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                        .fill_(stride_this_level)
                        .type_as(xin[0])
                )  # [(1, 169)值：32, ]
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid()], 1
                )

            outputs.append(output)  # 该程序中只有一层

        mask = self.sconv(yin)  # (b, 1, h*8(416), w*8(416))

        batchsize = yin.shape[0]


        if not self.training:

            # TODO anchor-free
            self.hw = [x.shape[-2:] for x in outputs]   # 该程序中只有一层, [13, 13]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)  # TODO [(1, 5, 13, 13)] --> [(1, 5, 169)] --> (1, 5, 169) --> (1, 169, 5)
            outputs = self.decode_outputs(outputs, dtype=xin[0].type())  # 计算每个回归框的绝对坐标
            outputs[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2  # x1 = xc - w/2
            outputs[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2  # y1 = yc - h/2 左上角顶点
            outputs[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2]      # x2 = x1 + w
            outputs[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3]      # y2 = y1 + h   右下角顶点
            score = outputs[:, :, 4]    # 置信分数
            batchsize = outputs.size()[0]
            ind = torch.argmax(score, -1).unsqueeze(1).unsqueeze(1).repeat(1, 1, outputs.size()[-1])    # TODO 取最大置信分数的下标 [60] --> [[[60]]] --> [[[60,60,60,60,60]]]
            pred = torch.gather(outputs, 1, ind)    # TODO outputs(1, 169, 5), ind(1, 1, 5)
                                                    #                          ind[0, 0, 0] [0, 0, 1] [0, 0, 2] [0, 0, 3] [0, 0, 4]
                                                    #                       -->   [0, ind[0, 0, 0], 0], [0, ind[0, 0, 1], 1], [0, ind[0, 0, 2], 2], [0, ind[0, 0, 3], 3], [0, ind[0, 0, 4], 4]
                                                    #                       例如：ind = [[[60,60,60,60,60]]]
                                                    #                       -->   [0, 60, 0] [0, 60, 1] [0, 60, 2] [0, 60, 3] [0, 60, 4] 为gather取值后的坐标+置信度


            mask = (mask.squeeze(1).sigmoid() > 0.35).float()  # TODO 这个mask的阈值是经验得来的吗

            return pred.view(batchsize, -1), mask



            # loss calculation
        # TODO 参与loss计算的元素
        #  1.【score_obj】 所有小于ignore_thre的output中的每一个grid中的每一个anchor 或 target所在grid且面积IOU最大的anchor
        #  2.【xy_obj】 target所在grid且面积IOU最大的anchor
        #  3.【wh_obj】 target所在grid且面积IOU最大的anchor，同时乘以一个系数
        #  4.【mask_obj】
        #  5.【co_energy】

        # 计算损失 seg、det、cem
        # loss_seg = nn.BCEWithLogitsLoss(reduction='sum')(mask, y_label) / 640. / batchsize  # TODO 640是怎么来的，为什么要除以640
        loss_seg = torch.tensor(-1)
        if y_label is not None:
            loss_seg = nn.BCEWithLogitsLoss(reduction='sum')(mask,
                                                             y_label) / 640. / batchsize  # TODO 640是怎么来的，为什么要除以640

        # TODO
        loss_det = self.get_losses(
            # imgs,  # None
            None,  # None  TODO
            x_shifts,  # x坐标 [(1,169)]
            y_shifts,  # y坐标 [(1,169)]
            expanded_strides,  # 步长  [(1,169)]
            # labels,  # xc, yc, w, h
            x_label,
            torch.cat(outputs, 1),  # xc yc w h 相对图片的绝对值 (8,169,5)
            origin_preds,  # []
            dtype=xin[0].dtype,  # torch.float32
        )
        # loss_det /= float(batchsize)
        loss_det = loss_det[0]
        loss_cem = self.co_energe(x_map, y_map, x_attn, y_attn)
        # loss = loss_det.sum() + loss_seg.sum() + loss_cem.sum()  # TODO 少了权重
        # TODO 针对seg_label为None的情况
        loss = loss_det.sum() + loss_cem.sum()
        if y_label is not None:
            loss += loss_seg
        return loss, loss_det, loss_seg

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)  # (1,1,13,13,2)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)  # (8,1,5,13,13)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )  # (8,1,13,13,5) --> (8,169,5)
        grid = grid.view(1, -1, 2)  # (1,1,13,13,2) --> (1,169,2)
        output[..., :2] = (output[..., :2] + grid) * stride  # TODO xc,yc
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # w,h
        return output, grid

    def decode_outputs(self, outputs, dtype):
        """
        对网络计算得到的每个grid中的相对坐标，转换为输入网络图的绝对坐标
        xc: 框的中心点x坐标
        yc: 框的中心点y坐标
        w : 框的宽
        h : 框的高

        Args:
            outputs:
            dtype:

        Returns:

        """
        grids = []
        strides = []
        #   (13, 13), 32
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid(
                [torch.arange(hsize), torch.arange(wsize)])  # tuple(2)    (Tensor(13,13), Tensor(13,13))
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)  # (13,13)&(13,13) --> (13, 13, 2) --> (1,169,2)
            grids.append(grid)
            shape = grid.shape[:2]  # [1, 169]
            strides.append(torch.full((*shape, 1), stride))  # (1, 169, 1) 值：32

        grids = torch.cat(grids, dim=1).type(dtype)  # [(1, 169, 2)] --> (1,169,2)
        strides = torch.cat(strides, dim=1).type(dtype)  # [(1, 169, 1)] --> (1,169,1)
        # 由grid的相对坐标转换为输入网络图的绝对坐标
        outputs[..., :2] = (outputs[..., :2] + grids) * strides  # ((1, 169, 2) + (1, 169, 2)) * (1, 169, 2)
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # exp((1, 169, 2)) * (1, 169, 1)
        return outputs

    def get_losses(
            self,
            imgs,  # None
            x_shifts,  # [(1,169)]
            y_shifts,  # [(1,169)]
            expanded_strides,  # [(1,169)]
            labels,  # (8,1,4)
            outputs,  # (8,169,5)
            origin_preds,  # []
            dtype,  # torch.float32
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        # cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # 标签存在，对应标志为1，否则为0
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # 每张图片总的anchor数
        total_num_anchors = outputs.shape[1]  # 169
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)  # (1, 169)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0  # 记录筛选出的k之和
        num_gts = 0.0  # 记录总的真实值标签个数

        for batch_idx in range(outputs.shape[0]):  # 一一处理每个batch
            num_gt = int(nlabel[batch_idx])  # 获取当前批次中一条数据的标签个数
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :4]  # (1,4)
                # gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # (169,4)

                try:
                    (
                        fg_mask,  # 最小动态k个cost的位置标记
                        pred_ious_this_matching,  # k个grid位置标记上的iou
                        matched_gt_inds,  # k个标签位置（一般标签数为1，所以值为k个0）
                        num_fg_img,  # 筛选出的最小cost的个数k
                    ) = self.get_assignments(  # noqa
                        batch_idx,  # batch中的第几个
                        num_gt,  # 真实标签个数 值1
                        total_num_anchors,  # （anchor数）总的grid数 值169
                        gt_bboxes_per_image,  # 坐标标签 (1,4)
                        None,
                        bboxes_preds_per_image,  # 每个grid预测的框 (169,4)
                        expanded_strides,  # 步长  (1,169)
                        x_shifts,  # x坐标 (1,169)
                        y_shifts,  # y坐标 (1,169)
                        None,
                        bbox_preds,
                        obj_preds,  # 预测目标的置信分数 (8,169,1)
                        labels,
                        imgs,  # None
                    )
                except RuntimeError:
                    torch.cuda.empty_cache()
                    (
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        None,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        None,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                obj_target = fg_mask.unsqueeze(-1)  # 目标gird标记（真实值） (169,1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # 目标框坐标  （真实值） (k,4)
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            # cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))  # 将 true 和 false 转换为 1.0 和 0.0
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
        # 对整个batch
        # cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)  # list --> (all k,4)   坐标真实值 batch个不一样的标签，其余一样
        obj_targets = torch.cat(obj_targets, 0)  # list --> (1352,1)    目标真实值（筛选的所有k个） 1352 = 8*169
        fg_masks = torch.cat(fg_masks, 0)  # list --> (1352,)     满足所有k的标记
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets.clamp(self.label_smooth,
                                                                                     1. - self.label_smooth) if self.label_smooth > 0. else obj_targets)
                   ).sum() / num_fg

        # loss_cls = (
        #     self.bcewithlog_loss(
        #         cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
        #     )
        # ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_l1

        return (
            loss, loss
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,  # batch中的第几个
            num_gt,  # 真实标签个数 值1
            total_num_anchors,  # （anchor数）总的grid数 值169
            gt_bboxes_per_image,  # 坐标标签 (1,4)
            gt_classes,
            bboxes_preds_per_image,  # 每个grid预测的框 (169,4)
            expanded_strides,  # 步长  (1,169)
            x_shifts,  # x坐标 (1,169)
            y_shifts,  # y坐标 (1,169)
            cls_preds,
            bbox_preds,
            obj_preds,  # 预测目标的置信分数 (8,169,1)
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            # gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        # 计算真实框和所有grid及其中心点的关系 fg_mask：在真实框或者定义框中的格子grid标记 is_in_boxes_and_center：在真实框或定义框的格子中，又同时在真实框和定义框的格子标记
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # 过滤没在框 且 没在中心点附近对应grid的预测值
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        # 求筛选后的预测值与真实值的IOU
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # 加一个极小值后（防止log(0)），取负对数，求loss，loss越大IoU越接近0，loss值越小IoU越接近1
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            # cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
            obj_preds_ = obj_preds_.cpu()
        # 放大没有同时在box（真实框）和center周围（定义框）的损失值
        cost = (3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
                )
        # 动态求 10 --> k 个iou较大的框，会改变传入的fg_mask，
        (
            num_fg,  # k
            pred_ious_this_matching,  # k个iou
            matched_gt_inds,  # k个0（标签下标，但只有一个标签）
        ) = self.dynamic_k_matching(cost, pair_wise_ious, num_gt, fg_mask)
        del cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            fg_mask,  # 最小动态k个cost的位置标记（相对于特征图13*13）
            pred_ious_this_matching,  # k个grid位置标记上的iou
            matched_gt_inds,  # k个标签位置（一般标签数为1，所以值为k个0）
            num_fg,  # 筛选出的最小cost的个数k
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )
        # 将真实值xc yc w h 转为左上和右下顶点坐标，并重复广播至anchors（grid）数
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        #
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # grid中心点是否在真实框内
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # 取最小值，若有负数则grid中心点在真实框外
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        #
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        # grid中心点是否在定义框（以真实框的中心点为中心，2*center_radius[2.5]个步长[即5个grid]为边长的自定义正方形框）内
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0  # grid中心点是否在框中心点的radius内

        # in boxes or in centers 取范围更大的
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        # 在真实框或定义框的格子中，同时在真实框和定义框的格子
        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))  # 挑选 在真实框或定义框的grid个数 或<=10个 的top IOU
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # 对上面挑选的top IOU求和取整得到k，挑选k(>=1)个最小的cost
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            # 对符合的k个cost标记位置为1，其余为0
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx
        # 对所有真实标签（一般为一个，只是去掉第一维），求在真实框或在定义框中的grid，满足最小k个cost的标签数
        anchor_matching_gt = matching_matrix.sum(0)  # (1,25) --> (25,)
        if (anchor_matching_gt > 1).sum() > 0:  # 单个真实值标签，不会有>1
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # 求在真实框或在定义框中的grid，满足任意一个最小k个cost的grid位置标记
        num_fg = fg_mask_inboxes.sum().item()  # 满足最小k个cost的grid数

        fg_mask[
            fg_mask.clone()] = fg_mask_inboxes  # 在框或在中心点附近的gird格子中，再挑选损失值较小的k个框，打上gird位置标记True, 其余为False TODO 改变了外部变量fg_mask (169,) --> (169,)
        # 求最小k个cost下，匹配标签最多的、在真实框和定义框grid中的，grid下标（一般匹配标签都为1个，返回第一个匹配的下标,即k个0）
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 过滤出cost最小的k个iou，并对标签数维度求总和（一般标签数为1，直接去掉标签数维度），后过滤出k个有值的iou
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        # 挑选的最小cost个数k，对应的k个iou，k个0（每个符合最小cost的grid中，第n（一般为0）个标签）
        return num_fg, pred_ious_this_matching, matched_gt_inds
