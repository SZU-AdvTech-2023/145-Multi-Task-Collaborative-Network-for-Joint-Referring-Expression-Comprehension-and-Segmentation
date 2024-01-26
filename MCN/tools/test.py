"""2023.1.24 黄立宏
对指定图片，根据输入的指称表达语句，利用训练的模型，进行指称表达理解，框出指定的人物
流程：
    1.数据
        1.1 文字
            将指称表达语句转换为词嵌入列表
        1.2 图片
            将图片数据转为训练时输入模型的图片格式
    2.模型
        2.1 根据配置文件创建模型
        2.2 加载模型参数
    3.运行
        3.1 输入处理后文字和图片数据后，经过模型运算，得到结果
    4.展示
        4.1 对结果进行变换，以图片的形式展示
"""
import os.path
import re
import cv2
import torch
import time
import numpy as np
from simrec.config import instantiate, LazyConfig

from PIL import Image, ImageDraw, ImageColor, ImageFont
from matplotlib import pyplot as plt

# TODO 不知道图片保存到哪
# import matplotlib
# matplotlib.use('Agg')

from simrec.utils.checkpoint import save_for_predict

# config_file = './configs/mcn_refcoco_scratch_1_fusion3_multscale_ema.py'
# config_file = './configs/mcn_refcoco_scratch_yolov8x.py'
config_file = './configs/mcn_refcoco_scratch_yolov8x_fusion3_multiscale_ema_anchorfree.py'
# config_file = './configs/mcn_refcoco_finetune_nls.py'
save_path = './data/output/inference'
# picture_name = "2023pp.jpg"
# picture_name = "2023pp1.jpg"
picture_name = "1507.jpg"
# picture_name = "2023yy.jpg"
weight_name = 'det_best_model.pth'
# weight_name = 'seg_best_model.pth'



def proc_ref(reference, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        reference.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def trans_box(boxes: np.ndarray,
              deviation: tuple,  # 偏置
              ratio):
    left, top, right, bottom = boxes
    dx, dy = deviation
    left = (left - dx) * ratio
    right = (right - dx) * ratio
    top = (top - dy) * ratio
    bottom = (bottom - dy) * ratio

    return np.asarray([left, top, right, bottom])


def trans_mask(mask: np.ndarray,
               deviation: tuple,
               pic_size: tuple):
    dx, dy = deviation
    mask = mask[dy:-dy - 1, :] if dx == 0 else mask[:, dx:-dx]  # 去除边框

    mask = np.expand_dims(mask, -1).astype(np.float32)
    h, w = pic_size
    sized_mask = cv2.resize(mask, (w, h))
    return sized_mask


def save_img_bgr_ndarray(image: np.ndarray,
                         box_4: np.ndarray = None,
                         mask_2: np.ndarray = None,
                         information: str = None):
    if information:
        path = os.path.join(save_path, picture_name.split('.')[0], information)
    else:
        path = os.path.join(save_path, picture_name.split('.')[0])
    os.makedirs(path, exist_ok=True)

    image_for_mask = image.copy()
    if box_4 is not None:
        x1, y1, x2, y2 = box_4
        x1 = int(x1)
        y1 = int(y1)
        x1 = max(x1, 0)
        y1 = max(y1, 0)

        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6)  # bgr
        cv2.imwrite(os.path.join(path, f"{picture_name.split('.')[0]}__box.png"), image)
    if mask_2 is not None:

        image_for_object = image_for_mask.copy()  # 提取只有分割物体的图片，需要添加alpha通道，增加透明度
        alpha_channel = np.ones(image_for_object.shape[:2], dtype=np.uint8) * 255
        image_for_object = cv2.merge((image_for_object, alpha_channel))  # 添加alpha通道
        _, binary_mask = cv2.threshold(mask_2.astype("uint8") * 255, 1, 255,
                                       cv2.THRESH_BINARY)  # TODO bitwise_and不仅需要形状相同，还要类型相同
        image_for_object[:, :, 3] = cv2.bitwise_and(image_for_object[:, :, 3], binary_mask)  # 提取分割目标物体
        cv2.imwrite(
            os.path.join(path, f"{information}__object.png"),
            image_for_object)  # 保存分割物体

        cv2.imwrite(os.path.join(path, f"{picture_name.split('.')[0]}__m.png"), mask_2 * 255)
        score = int(mask_2[y1:y2, x1:x2].mean() * 100)
        print(f"mask score:[{score}]")
        image_for_mask[:, :, 0][mask_2[:, :] > 0] = 255  # bgr
        cv2.imwrite(os.path.join(path, f"{picture_name.split('.')[0]}__mask.png"), image_for_mask)
        if box_4 is not None:
            image[:, :, 0][mask_2[:, :] > 0] = 255
            cv2.imwrite(os.path.join(path, f"{picture_name.split('.')[0]}__box_mask.png"), image)
    if box_4 is None and mask_2 is None:
        cv2.imwrite(os.path.join(path, f"{picture_name.split('.')[0]}_{information if information else ''}.png"), image)


# 加载图片，对图片进行等比例缩放，调整为正方形大小，并转为tensor格式，进行正则化
def load_img(path, input_shape, transforms):
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    imgsize = input_shape[0]
    new_ar = w / h
    if new_ar < 1:
        ratio = h / imgsize
        nh = imgsize
        nw = nh * new_ar
    else:
        ratio = w / imgsize
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    dx = (imgsize - nw) // 2
    dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy + nh, dx:dx + nw, :] = img
    # TODO 显示图片
    # 将图像颜色通道翻转 BGR --> RGB
    # pil_img = Image.fromarray(sized)
    # show_img(pil_img)
    img_bgr_resized = cv2.cvtColor(sized, cv2.COLOR_RGB2BGR)
    save_img_bgr_ndarray(img_bgr_resized)

    # ratio 原图中最长边和input shape中边长的比例
    return transforms(sized), sized, (dx, dy), ratio


def test(model, ref_ix, img):
    model.eval()

    with torch.no_grad():
        start = time.time()
        ref_ix = ref_ix.cuda()
        img = img.cuda()
        box, mask = model(img, ref_ix)
        print('运算耗时：%.8f s' % (time.time() - start))
        return box, mask


if __name__ == '__main__':

    # config_file = './configs/mcn_refcoco_scratch_1_fusion3_multscale_ema.py'
    parameters_file = './inference_parameters/parameters_for_refcoco_predict.pth'
    # 加载配置文件
    cfg = LazyConfig.load(config_file)

    weight_file = os.path.join(cfg.train.output_dir, weight_name)

    save_path = os.path.join(save_path, cfg.train.output_dir, weight_name.split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    picture_file = f'./data/test_images/{picture_name}'
    # weight_file = os.path.join(cfg.train.output_dir, 'seg_best_model.pth')
    # TODO 定义词嵌入索引，加快向量转换

    cfg = LazyConfig.apply_overrides(cfg, None)
    # 加载训练数据集，获取其中的pretrained_emb和token_size，用于创建模型中的语言特征提取模型，同时调用其中的load_refs函数将表达语句转为embedding的索引
    train_set = instantiate(cfg.dataset)
    save_for_predict(cfg, train_set)  # TODO 提前加载统计数据中的token_to_ix等数据，并保存起来（只需要第一次加载时运行）

    # 加载模型参数
    checkpoint = torch.load(weight_file, map_location=lambda storage, loc: storage.cuda())
    parameters = torch.load(parameters_file)
    # 根据配置文件实例化模型
    cfg.model.language_encoder.pretrained_emb = checkpoint['state_dict']['lang_encoder.embedding.weight'].cpu().numpy()
    cfg.model.language_encoder.token_size = checkpoint['state_dict']['lang_encoder.embedding.weight'].shape[0]

    model = instantiate(cfg.model)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded weight to model!")

    print("loading model to gpu...")
    model.cuda()
    print("loaded model to gpu!")

    # 加载图片数据并进行转换
    # img_trans: 转换成tensor后的图片
    # sized: 缩放调整后的正方形图片，np.ndarray
    # deviation: 调整后的图片与原始图片的坐标轴（以图片左上角为坐标轴远点）偏移
    # ratio: 缩放比例
    transforms = instantiate(cfg.dataset.transforms)
    print("loaded image transforms!")
    img_trans, sized, deviation, ratio = load_img(picture_file, parameters["input_shape"], transforms)
    print("image loaded!")


    img_trans = img_trans.unsqueeze(0)

    while True:
        ref = input('input the ref(or input exit): ')
        if ref == 'exit':
            break

        ques_ix = proc_ref(ref, parameters["token_to_ix"], parameters["max_token"])
        ques_ix = torch.from_numpy(ques_ix).long()
        ques_ix = ques_ix.unsqueeze(0)

        box, mask = test(model, ques_ix, img_trans)
        print("置信度：%f" % box.cpu().numpy()[0, -1])


        img_bgr = cv2.imread(picture_file)
        tran_box = trans_box(box.cpu().numpy()[0, :-1], deviation, ratio)
        img_size = (img_bgr.shape[0], img_bgr.shape[1])  # h, w
        tran_mask = trans_mask(mask.cpu().numpy()[0], deviation, img_size)
        save_img_bgr_ndarray(img_bgr, tran_box, tran_mask, ref)
