# 四季の画像変換アプリ
このアプリは、Streamlitを使って開発された四季の画像変換アプリです。

# 機能

* 入力画像をResnet18を用いて分類し、その季節を判別します。
* CycleGANモデルを使用して、入力画像を他の季節の画像に変換します。
* 変換後の春、夏、秋、冬の4つの画像を表示します。

# 使用技術

* Python
* Streamlit
* PyTorch
* Resnet18
* CycleGAN

# 実行方法
このアプリは、GitHub上のリポジトリから入手できます。以下のコマンドを使ってリポジトリをクローンしてください。

```git clone https://github.com/sh7o90/cyclegan-app```

このコマンドを実行すると、cyclegan-appというディレクトリにアプリのソースコードがダウンロードされます。

クローンが完了したら、以下の手順に従ってアプリを実行してください。

１．必要なライブラリをインストールします。

```cd cyclegan-app```

```pip install -r requirements.txt```

２．アプリを起動します。

```streamlit run app.py```

３．ブラウザで以下にアクセスすると、アプリが表示されます。

```http://localhost:8501```

# デモ
https://github.com/sh7o90/cyclegan-app/assets/158803446/85778374-05a2-42e4-a3b0-249d6c0cdbb2

# ライセンス
このプロジェクトは MIT ライセンスの下で公開されています。[MIT license](https://en.wikipedia.org/wiki/MIT_License)

# 作者
奥野 翔太(sh7o90)
