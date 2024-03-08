<div class="title" align=center>
    <h1>GPT-SoVITS RefAudio Tester</h1>
	<div>GPT-SoVITS 参考音频推理效果批量试听</div>
    <br/>
    <p>
        <img src="https://img.shields.io/github/license/2DIPW/GPT-SoVITS-RefAudio-Tester">
    	<img src="https://img.shields.io/badge/python-3.9-blue">
        <img src="https://img.shields.io/github/stars/2DIPW/GPT-SoVITS-RefAudio-Tester?style=social">
        
</div>

## 🚩 简介
本项目是一个拥有 WebUI 的 GPT-SoVITS 批量推理器，旨在快速试听多个候选参考音频的推理效果，以筛选出其中效果最令人满意的参考音频。

推理部分的源码基于 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 修改，Gradio 部分的源码参考了 [cronrpc/SubFix](https://github.com/cronrpc/SubFix) 的写法。

## 📥 部署
### 克隆
```shell
git clone https://github.com/2DIPW/GPT-SoVITS-RefAudio-Tester.git
cd GPT-SoVITS-RefAudio-Tester
```
### 安装依赖
本项目相比 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 而言没有引入更多的依赖，可以直接使用为其配置的环境。

### 配置预训练模型
与 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 中相同，需要将预训练模型放置于 `GPT_SoVITS/pretrained_models` 目录。
## 🗝 使用方法
### 准备自己训练的 GPT 和 SoVITS 模型
- 将自己训练的 GPT 模型放入 `GPT_weights` 目录

- 将自己训练的 SoVITS 模型放入 `SoVITS_weights` 目录
### 准备参考音频及参考音频注释列表文件
- 参考音频注释列表 .list 文件格式与 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 中相同：

    ```
    参考音频文件名或绝对路径|说话人|语言|参考文本
    ```
- 其中说话人虽然在本项目中无用，但为了与其他项目生成的 list 文件兼容，仍将其保留，可以随意填写。

- 语言字典：

  - "ZH": "中文"
  - "zh": "中文"
  - "JP": "日文"
  - "jp": "日文"
  - "JA": "日文"
  - "ja": "日文"
  - "EN": "英文"
  - "en": "英文"
  - "En": "英文"

- 示例：
    ```
    ATR_b102_006.wav|アトリ|jp|へっちゃらです。高性能ですから
    ```
    ```
    D:\xxx\ATR_b102_006.wav|アトリ|jp|へっちゃらです。高性能ですから
    ```
### 运行
- 使用`webui.py`
    ```shell
    python webui.py -l 参考音频注释列表文件 -f 参考音频所在目录 -b 10
    ```
    可指定的参数:
    - `-l` | `--list`: 参考音频注释列表文件的位置。默认值：`ref.list`
    - `-p` | `--port`: WebUI的监听端口。默认值：14285
    - `-f` | `--folder`: 参考音频所在目录。**如果参考音频注释列表中第一列内容仅为文件名，或虽为绝对路径，但是音频文件已移动至其他位置，则需要指定该参数。程序会将指定的目录与文件名拼接，作为最终的参考音频文件路径。** 默认值：None
    - `-b` | `--batch`: 每一批最多处理多少个音频。因 Gradio 不支持动态增减控件数量，此值需要预先指定以生成控件，且在运行过程中无法修改。默认值：10
    - `-cd` | `--check_duration`：是否检查音频时长。启用此选项后，在启动时会检查每条音频时长是否在 3~10s 的范围内，若不在则不会加载。若准备的参考音频中均无时长超过范围的，可不开启此项，以缩短启动时间。默认值：不启用
    - `-r` | `--random_order`：是否乱序参考音频列表。启用此选项后，将会把从文件中读取的参考音频列表打乱顺序。默认值：不启用
- 在`试听文本`中填入用来测试推理效果的文本，点击`合成试听语音`，即可为当前批次的参考音频生成对应的试听音频。
- 在`满意的参考音频复制到`中设置目的目录，在点击音频旁的`满意`按钮后，该条参考音频将以`{参考文本}.wav`的文件名复制到该目录。

## ⚖ 开源声明
本项目基于 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 修改，并以 [GNU General Public License v3.0](https://github.com/2DIPW/GPT-SoVITS-RefAudio-Tester/blob/master/LICENSE) 开源

本项目基于 LGPL 2.1 协议包含一份 FFmpeg 的可执行文件

*世界因开源更精彩*
