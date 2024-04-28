"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

画像変換のための汎用的なテストスクリプトです。

train.pyでモデルを訓練した後、このスクリプトを使ってモデルをテストできます。
このスクリプトは、--checkpoints_dir から保存されたモデルを読み込み、結果を --results_dir に保存します。

このスクリプトは、最初に指定されたオプションに基づいてモデルとデータセットを作成します。一部のパラメータはハードコードされます。
その後、--num_test で指定された数の画像に対して推論を行い、結果をHTMLファイルとして保存します。

例 (モデルを訓練するか、ウェブサイトから事前訓練済みモデルをダウンロードする必要があります):
    CycleGANモデルをテストする（両方向で）:
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    CycleGANモデルをテストする（一方向のみ）:
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

--model test オプションは、CycleGAN結果を一方向のみ生成するために使用されます。

このオプションを使用すると、--dataset_mode single が自動的に設定され、1つのセットからのみ画像を読み込むようになります。

一方、--model cycle_gan を使用すると、両方向で結果を生成するため、時には不要な場合もあります。結果は ./results/ に保存されます。
結果を保存するディレクトリを指定するには、--results_dir <結果を保存するディレクトリパス> を使用します。

    pix2pixモデルをテストする:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
テストオプションの詳細については、options/base_options.py と options/test_options.py を参照してください。
トレーニングとテストのヒントについては、こちらのドキュメント を参照してください。
よくある質問については、こちらのFAQ を参照してください。
"""

from .options.test_options import TestOptions
from .data import create_dataset
from .models import create_model
from .util.visualizer import save_images
from .util import html

import os
import numpy as np
import streamlit as st
from PIL import Image

def cycle_gan(folder_path):
    st.write("folder_path: ", folder_path)
    opt = TestOptions().parse()  # get test options

    st.write("opt: ", opt)

    # hard-code some parameters for test
    opt.dataroot = folder_path
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        print("test_mode")
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        processed = np.array((visuals["fake"][0].detach().numpy().copy().transpose(1,2,0) + 1) *128, dtype="uint8")
        return processed

if __name__ == '__main__':
    ret = cycle_gan()
    result = Image.fromarray(np.uint8(ret))
    result.save("result.png")
