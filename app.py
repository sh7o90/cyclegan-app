import os
import streamlit as st
import torch
from torchvision.transforms import Resize, ToTensor, ToPILImage
from PIL import Image
from mlmodel.models.cycle_gan_model import CycleGANModel
import pickle
import torchvision.models as models
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn


# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Options:
    def __init__(self):
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        # self.netG = 'resnet_9blocks'
        self.netG = 'unet_256'
        self.netD = 'basic'
        self.no_dropout = True
        self.gpu_ids = []  # GPUを使用する場合は、使用するGPUのIDを指定する
        self.isTrain = False  # 推論時はFalseに設定
        self.checkpoints_dir = ''  # チェックポイントの保存先ディレクトリ
        self.name = ''  # モデル名
        self.preprocess = 'scale_width'  # 前処理の方法
        self.norm = 'instance'  # 正規化の方法
        self.init_type = 'normal'  # 重み初期化の方法
        self.init_gain = 0.02  # 重み初期化の係数
        self.direction = 'AtoB'  # 入力データの方向 ('AtoB' or 'BtoA')

# モデルの定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 4)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

def apply_cyclegan(image, model):
    """
    CycleGANモデルを使用して画像を変換する関数

    Args:
        image (PIL.Image): 入力画像
        model (CycleGANModel): CycleGANモデルのインスタンス

    Returns:
        PIL.Image: 変換後の画像
    """
    # 画像をモデルに入力して変換
    input_image = Resize((256, 256))(image)
    input_tensor = ToTensor()(input_image).unsqueeze(0).to(model.device)
    model.set_input({'A': input_tensor, 'B': input_tensor})
    model.forward()
    fake_B, rec_A, fake_A, rec_B = model.fake_B, model.rec_A, model.fake_A, model.rec_B

    # TensorをPIL Imageに変換
    if model.direction == 'AtoB':
        output_image = (fake_B[0].cpu().detach() * 0.5 + 0.5).clamp(0, 1)
        output_image = Image.fromarray((output_image.permute(1, 2, 0) * 255).byte().cpu().numpy(), 'RGB')
    else:
        output_image = (fake_A[0].cpu().detach() * 0.5 + 0.5).clamp(0, 1)
        output_image = Image.fromarray((output_image.permute(1, 2, 0) * 255).byte().cpu().numpy(), 'RGB')

    return output_image

opt = Options()

model_spring = CycleGANModel(opt)
model_spring.direction = opt.direction
model_spring.load_networks('mlmodel/checkpoints/summer_spring_cyclegan/latest')
model_spring.eval()

model_autumn = CycleGANModel(opt)
model_autumn.direction = opt.direction
model_autumn.load_networks('mlmodel/checkpoints/summer_autumn_cyclegan/latest')
model_autumn.eval()

model_winter = CycleGANModel(opt)
model_winter.direction = opt.direction
model_winter.load_networks('mlmodel/checkpoints/summer_winter_cyclegan/latest')
model_winter.eval()

# モデルのロード
net = Net()
# 四季の画像分類モデルをロード
with open('image_classification_resnet18.pkl', 'rb') as f:
    net.load_state_dict(pickle.load(f))
net.eval()  # 推論モードに設定

st.title("画風変換アプリ")
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 画像をロードして表示
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='入力画像', use_column_width=True)

    # 画像分類を実行
    input_image = transform(image).unsqueeze(0)

    outputs = net(input_image)
    # 最も確率の高いクラスのインデックスを取得
    _, predicted_class_idx = torch.max(outputs, 1)

    # クラスラベルを取得
    class_labels = ['秋', '春', '夏', '冬']  # クラスラベルを手動で定義
    predicted = predicted_class_idx.item()
    predicted_class = class_labels[predicted]
    st.sidebar.success(f"画像は {predicted_class} だと判断されました。")
    st.sidebar.write(f"CycleGANモデルの direction:  \n春 - {model_spring.direction} , 秋 - {model_autumn.direction} , 冬 - {model_winter.direction}")

    # CycleGANモデルのDirection変更と夏画像の生成（1st Step)
    if predicted_class == '秋':
        model_autumn.direction = 'BtoA'
        st.sidebar.write(f"変更点 {predicted_class} : {model_autumn.direction}")
        output_image_autumn = Resize((256, 256))(image)
        output_image_summer = apply_cyclegan(output_image_autumn, model_autumn)
    elif predicted_class == '春':
        model_spring.direction = 'BtoA'
        st.sidebar.write(f"変更点 {predicted_class} : {model_spring.direction}")
        output_image_spring = Resize((256, 256))(image)
        output_image_summer = apply_cyclegan(output_image_spring, model_spring)
    elif predicted_class == '冬':
        model_winter.direction = 'BtoA'
        st.sidebar.write(f"変更点 {predicted_class} : {model_winter.direction}")
        output_image_winter = Resize((256, 256))(image)
        output_image_summer = apply_cyclegan(output_image_winter, model_winter)
    else :
        output_image_summer = Resize((256, 256))(image)

    # CycleGANモデルで四季画像の生成（2nd Step）
    if predicted_class == '秋':
        output_image_spring = apply_cyclegan(output_image_summer, model_spring)
        output_image_winter = apply_cyclegan(output_image_summer, model_winter)
    elif predicted_class == '春':
        output_image_autumn = apply_cyclegan(output_image_summer, model_autumn)
        output_image_winter = apply_cyclegan(output_image_summer, model_winter)
    elif predicted_class == '冬':
        output_image_autumn = apply_cyclegan(output_image_summer, model_autumn)
        output_image_spring = apply_cyclegan(output_image_summer, model_spring)
    else :
        output_image_autumn = apply_cyclegan(output_image_summer, model_autumn)
        output_image_spring = apply_cyclegan(output_image_summer, model_spring)
        output_image_winter = apply_cyclegan(output_image_summer, model_winter)

    # メイン画面に表示
    col1, col2 = st.columns(2)

    with col1:
        st.image(output_image_spring, caption='春', use_column_width=True)

    with col2:
        st.image(output_image_summer, caption='夏', use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(output_image_autumn, caption='秋', use_column_width=True)

    with col2:
        st.image(output_image_winter, caption='冬', use_column_width=True)
