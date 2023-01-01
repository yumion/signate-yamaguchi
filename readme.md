# githubのclone
submoduleを含めてcloneする
```bash
git clone --recursive https://github.com/yumion/signate-yamaguchi.git
```

後からsubmoduleをcloneする場合
```bash
git submodule update --init --recursive
```
---


# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法について説明する.

1. [配布データ](#配布データ)
1. [応募用ファイルの作成方法](#応募用ファイルの作成方法)

## 配布データ

配布されるデータは以下の通り.

- [Readme](#readme)
- [学習用データ](#学習用データ)
- [動作確認用のプログラム](#動作確認用のプログラム)
- [応募用サンプルファイル](#応募用サンプルファイル)

### Readme

本ファイル(readme.md)で, 配布用データの説明と応募用ファイルの作成方法を説明したドキュメント. マークダウン形式で, プレビューモードで見ることを推奨する.

### 学習用データ

学習用データは"train.zip"で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
train
├─ scene_00.json
├─ scene_00.mp4
└─ ...
```

内容は以下の通り.

- [動画データ](#動画データ)
- [アノテーションデータ](#アノテーションデータ)

#### 動画データ

シーン別の動画データ. "scene_00.mp4"などが本体で, 内容は以下の通り.

- ファイル形式はmp4.
- 解像度は1920×1080.
- fpsは10.0.

#### アノテーションデータ

シーン別の動画データに対応するアノテーションデータ. "scene_00.json"などが本体で, 内容は以下の通り.

- ファイル形式はjson.
- encodingはutf-8.

データのフォーマットは以下の通り.

```json
[
    {
        "frame_id": 0,
        "labels": {
            category_1: [
                [
                    [
                        lefttopx,
                        lefttopy
                    ],
                    [
                        rightbottomx,
                        rightbottomy
                    ]
                ],
                ...
            ],
            ...
        },
        ...
    },
    ...
]
```

- "frame_id"は動画におけるフレーム番号である. 0から始まり, 時系列順に+1される.
- "labels"は対応するフレームにおける物体の矩形情報である.
  - カテゴリ名一覧は以下の通り.
    - 要補修-1.区画線
    - 要補修-2.道路標識
    - 要補修-3.照明
    - 補修不要-1.区画線
    - 補修不要-2.道路標識
    - 補修不要-3.照明
  - category_1などには上記の一覧の中のどれかが入る.
  - lefttopx, lefttopy, rightbottomx, rightbottomyはそれぞれフレームの左上端を原点としたときの矩形の左上のx座標, 左上のy座標, 右下のx座標, 右下のy座標である.

以下, 認識対象となるカテゴリについて説明する.

##### 対象物体

###### 1.区画線

路面や縁石, 橋桁などにペイントなどを用いて交通の案内誘導, 指示などを与えるもので, 中央線や車線, 追越し禁止や駐停車禁止, 横断歩道およびその予告, 通行区分, 右左折の方法や転回禁止, 規制走行速度などさまざまなものがこれらをすべてまとめて"区画線"と定義する.

###### 2.道路標識

道路の傍ら若しくは上空に設置され, 利用者に必要な情報を提供する表示板である. 様々な種類があるが, 今回のデータではすべてまとめて"道路標識"と定義する.

###### 3.照明

主に道路を照らす為に立てられている電灯のことである.

##### 要補修の基準

以下に該当する状態が画像から確認できる場合は, "要補修" と判断する.

- 区画線は消え始めていると確認できる.
- 標識と照明は折れ, 曲がり, 錆付き, 破損, 文字のかすれが確認できる.

ただし, 部分的にしか映っていない, 遠すぎるなどが原因で画像から状態が確認できない場合は"補修不要", もしくは矩形は囲わないとする. よって, 同じ物体でも近づいた結果"補修不要"から"要補修"となることがあり(少ないが逆もしかり), 切り替わるタイミングは目視による判断となっているため, 一定のばらつきが存在することに注意. "要補修"の判定は与えられたデータを正とすること.

### 動作確認用のプログラム

予測を行うプログラムの動作を確認するためのプログラム. "run_test.zip"が本体で, 解凍すると, 以下のようなディレクトリ構造のデータが作成される.

```bash
run_test
└─ run.py
```

使用方法等については[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照すること.

### 応募用サンプルファイル

応募用のサンプルファイル. 実体は"sample_submit.zip"で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
sample_submit
├── model
│   └── ...
├── src
│   └── predictor.py
└── requirements.txt
```

sample_submit/modelは空のディレクトリとなる. sample_submit/requirements.txtには何も書いていない. 詳細や作成方法については[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照すること.

## 応募用ファイルの作成方法

学習済みモデルを含めた, 予測を実行するためのソースコード一式をzipファイルでまとめたものとする.

### ディレクトリ構造

以下のようなディレクトリ構造となっていることを想定している.

```bash
.
├── model              必須: 学習済モデルを置くディレクトリ
│   └── ...
├── src                必須: Pythonのプログラムを置くディレクトリ
│   ├── predictor.py   必須: 最初のプログラムが呼び出すファイル
│   └── ...            その他のファイル (ディレクトリ作成可能)
└── requirements.txt   任意: 追加で必要なライブラリ一覧
```

- 学習済みモデルの格納場所は"model"ディレクトリを想定している.
  - 学習済みモデルを使用しない場合でも空のディレクトリを作成する必要がある.
  - 名前は必ず"model"とすること.
- Pythonのプログラムの格納場所は"src"ディレクトリを想定している.
  - 学習済みモデル等を読み込んで推論するためのメインのソースコードは"predictor.py"を想定している.
    - ファイル名は必ず"predictor.py"とすること.
  - その他予測を実行するために必要なファイルがあれば作成可能である.
  - ディレクトリ名は必ず"src"とすること.
- 実行するために追加で必要なライブラリがあれば, その一覧を"requirements.txt"に記載することで, 評価システム上でも実行可能となる.
  - インストール可能で実行可能かどうか予めローカル環境で試しておくこと.
  - 評価システムの実行環境については, [*こちら*](https://github.com/signatelab/runtime-gpu)を参照すること.

### predictor.pyの実装方法

以下のクラスとメソッドを実装すること.

#### ScoringService

予測実行のためのクラス. 以下のメソッドを実装すること.

##### get_model

モデルを取得するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数model_path(str型)を指定すること.
  - 学習済みモデルが格納されているディレクトリのパスが渡される.
- 学習済みモデルの読み込みに成功した場合はTrueを返す.
  - モデル自体は任意の名前(例えば"model")で保存しておく.

##### predict

予測を実行するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数input(str型)を指定すること.
  - 予測対象となる動画のパス名がstr型で渡される.
  - `get_model`メソッドで読み込んだ学習済みモデルを用いて対応する動画に対する予測を行う想定である.

以下のフォーマットで予測結果を**list型**で返す.

```json
[
    {
        "frame_id": 0,
        "line": line,
        "sign": sign,
        "light": light
    },
    {
        "frame_id": 1,
        "line": line,
        "sign": sign,
        "light": light
    },
    ...
]
```

- 各要素は以下を満たすdict型.
  - "frame_id"には対応する動画におけるフレーム番号を記載する. 0から始まり, 時系列順に+1される.
  - "line"は対応するフレームにおいて要補修の区画線が存在するかのフラグを記載する. 存在する場合は1, しない場合は0.
  - "sign"は対応するフレームにおいて要補修の道路標識が存在するかのフラグを記載する. 存在する場合は1, しない場合は0.
  - "light"は対応するフレームにおいて要補修の照明が存在するかのフラグを記載する. 存在する場合は1, しない場合は0.

なお, 対応するフレームにおいて要補修物体の矩形領域が一つでも存在した場合に要補修物体が存在すると判断する.

以下は実装例.

```Python
import cv2


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.model = load_model(model_path)

        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: Data of the sample you want to make inference from (str)

        Returns:
            list: Inference for the given input.

        """
        prediction = []
        cap = cv2.VideoCapture(input)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if ret:
                preprocessed = preprocess_frame(frame)
                pred = cls.model(preprocessed)
                postprocessed = postprocess_pred(pred)
                prediction.append(postprocessed)
                frame_id += 1
            else:
                break

        return prediction


def load_model(model_path):
    ...
    return model


def preprocess_frame(frame):
    ...
    return preprocessed


def postprocess_pred(pred):
    """
    returns: dict
    """
    ...
    return postprocessed
```

応募用サンプルファイル"sample_submit.zip"も参照すること.

### 推論テスト

予測を行うプログラムが実装できたら, 正常に動作するか確認する.

#### 環境構築

評価システムと[同じ環境](https://github.com/signatelab/runtime-gpu)を用意する.

#### 予測の実行

配布データを用いてモデル学習などを行い, [動作確認用のプログラム](#動作確認用のプログラム)を用いて検証用のデータを作成し, 予測を実行する. 最後に精度(MAE)を確認する.

```bash
$ cd /path/run_test
$ python run.py  --exec-path /path/to/submit/src --data-dir /path/to/train --scene-id scene_id
...
```

- 引数"--exec-path"には実装した予測プログラム("predictor.py")が存在するパス名を指定する.
- 引数"--data-dir"には配布された学習用データ"train"のパス名を指定する.
- 引数"--scene_id"には学習用データにおいて, 評価したいシーンIDを指定する. デフォルトでは"00".

"run.py"の実行に成功すると, 検証用のデータを作成し, 実装したプログラムによる予測が行われ, 最後に精度が出力される. プログラム上のエラーがない場合でも, モデルの読み込みに失敗したときや, 予測結果のフォーマットが正しくないときは途中でエラーのメッセージが返され, 止まる. 具体的には, `get_model`メソッドを呼んだときに`True`を返さない, 予測結果がlist型ではない, keyが'frame_id', 'line', 'sign', 'light'を含んでいない, 'line', 'sign', 'light'のデータ型がintでかつ0または1になっていない場合.  

投稿する前にエラーが出ずに実行が成功することを確認すること.

### 応募用ファイルの作成

上記の[ディレクトリ構造](#ディレクトリ構造)となっていることを確認して, zipファイルとして圧縮する.
