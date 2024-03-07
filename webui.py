from GPT_SoVITS import inference_main
import gradio as gr
import argparse
import os
import re
import csv
import shutil
import librosa
from tqdm import tqdm
import random
import logging

logging.getLogger("PIL.Image").propagate = False

language_v1_to_language_v2 = {
    "ZH": "中文",
    "zh": "中文",
    "JP": "日文",
    "jp": "日文",
    "JA": "日文",
    "ja": "日文",
    "EN": "英文",
    "en": "英文",
    "En": "英文",
}
dict_language = {
    "中文": "all_zh",  # 全部按中文识别
    "英文": "en",  # 全部按英文识别#######不变
    "日文": "all_ja",  # 全部按日文识别
    "中英混合": "zh",  # 按中英混合识别####不变
    "日英混合": "ja",  # 按日英混合识别####不变
    "多语种混合": "auto",  # 多语种启动切分识别语种
}
dict_how_to_cut = {
    "不切": 0,
    "凑四句一切": 1,
    "凑50字一切": 2,
    "按中文句号。切": 3,
    "按英文句号.切": 4,
    "按标点符号切": 5
}


def check_audio_duration(path):
    try:
        wav16k, sr = librosa.load(path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            return False
        else:
            return True
    except Exception as e:
        print(f"Error when checking audio {path}: {e}")
        return False


def remove_noncompliant_audio_from_list():  # 从参考音频文件列表中清除长度不符合要求的音频
    print("Checking audio duration ...")
    global g_ref_list, g_ref_list_max_index
    new_ref_list = []
    for line in tqdm(g_ref_list):
        if check_audio_duration(line[0]):
           new_ref_list.append(line)
    g_ref_list = new_ref_list
    g_ref_list_max_index = len(g_ref_list) - 1


def load_ref_list_file(path):  # 加载参考音频列表文件
    global g_ref_list, g_ref_list_max_index
    with open(path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='|')
        g_ref_list = list(reader)
        if g_ref_folder:  # 如果指定了参考文件目录参数，则拼接文件名
            for _ in g_ref_list:
                _[0] = os.path.join(g_ref_folder, os.path.basename(_[0]))
        g_ref_list_max_index = len(g_ref_list) - 1


def get_weights_names():  # 获取模型列表
    SoVITS_names = []
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = []
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return sorted(SoVITS_names, key=custom_sort_key), sorted(GPT_names, key=custom_sort_key)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def refresh_model_list():  # 刷新模型列表
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": SoVITS_names, "__type__": "update"}, {
        "choices": GPT_names, "__type__": "update"}


def reload_data(index, batch):  # 从index起始，由文件列表中加载一批数据
    global g_index
    g_index = index
    global g_batch
    g_batch = batch
    datas = g_ref_list[index:index + batch]
    output = []
    for d in datas:
        try:
            output.append(
                {
                    "path": d[0],
                    "lang": d[2],
                    "text": d[3]
                }
            )
        except IndexError:
            pass
    return output


def change_index(index, batch):  # 起始索引更改后，将新的一批数据填充进表格
    global g_index, g_batch, g_ref_audio_path_list
    g_ref_audio_path_list = []
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)
    output = []
    # 参考音频
    for i, _ in enumerate(datas):
        output.append(
            {
                "__type__": "update",
                "label": f"参考音频 {os.path.basename(_['path'])}",
                "value": _["path"]
            }
        )
        g_ref_audio_path_list.append(_['path'])
    for _ in range(g_batch - len(datas)):
        output.append(
            {
                "__type__": "update",
                "label": "参考音频",
                "value": None
            }
        )
        g_ref_audio_path_list.append(None)
    # 参考音频语言
    for _ in datas:
        output.append(_["lang"])
    for _ in range(g_batch - len(datas)):
        output.append(None)
    # 参考文本
    for _ in datas:
        output.append(_["text"])
    for _ in range(g_batch - len(datas)):
        output.append(None)
    # 试听音频
    for _ in range(g_batch):
        output.append(None)
    # 满意按钮
    for _ in datas:
        output.append(
            {
                "__type__": "update",
                "value": "满意",
                "interactive": True
            }
        )
    for _ in range(g_batch - len(datas)):
        output.append(
            {
                "__type__": "update",
                "value": "满意",
                "interactive": False
            }
        )

    return output


def previous_index(index, batch):  # 上一批数据
    if (index - batch) >= 0:
        return index - batch, *change_index(index - batch, batch)
    else:
        return 0, *change_index(0, batch)


def next_index(index, batch):  # 下一批数据
    if (index + batch) <= g_ref_list_max_index:
        return index + batch, *change_index(index + batch, batch)
    else:
        return index, *change_index(index, batch)


def copy_proved_ref_audio(index, text, out_dir):  # 满意按钮点击
    os.makedirs(out_dir, exist_ok=True)
    filename = re.sub(r'[/\\:*?\"<>|]', '', text)  # 删除不能出现在文件名中的字符
    try:
        shutil.copy2(g_ref_audio_path_list[int(index)], os.path.join(out_dir, filename+".wav"))
        return {
                    "__type__": "update",
                    "value": "已复制!",
                    "interactive": False
                }
    except Exception as e:
        print(e)
        return {
            "__type__": "update",
            "value": "复制失败",
            "interactive": True
        }


def generate_test_audio(test_text, language, how_to_cut, top_k, top_p, temp, *widgets):  # 生成试听音频
    output = []
    for _ in range(g_batch):
        r_audio = g_ref_audio_path_list[_]
        if r_audio:
            try:
                r_lang = dict_language[language_v1_to_language_v2[widgets[_]]]
                r_text = widgets[_ + g_batch]
                gen_audio = inference_main.get_tts_wav(r_audio, r_text, r_lang, test_text, dict_language[language], dict_how_to_cut[how_to_cut],
                                                       top_k, top_p, temp)
                sample_rate, array = next(gen_audio)
                output.append((sample_rate, array))
            except OSError:  # 若音频长度不符合要求
                print(f"Duration of {r_audio} is not compliant, skip")
                output.append(None)
        else:
            output.append(None)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', type=str, default="ref.list", help='List of ref audio files, default is ref.list')
    parser.add_argument('-p', '--port', type=int, default=14285, help='Port of WebUI, default is 14285')
    parser.add_argument('-f', '--folder', type=str, default="",
                        help='The directory of ref audio files, if not specified, abs path in the list file will be used, default is None.')
    parser.add_argument('-b', '--batch', type=int, default=10,
                        help='How many ref audio files will be processed at one time, default is 10')
    parser.add_argument('-cd', '--check_duration', action='store_true', default=False,
                        help='Whether to check if the duration of every ref audio is between 3 and 10 seconds.')
    parser.add_argument('-r', '--random_order', action='store_true', default=False,
                        help='Whether to randomize the ref audio list.')

    args = parser.parse_args()

    SoVITS_weight_root = "SoVITS_weights"
    GPT_weight_root = "GPT_weights"
    os.makedirs(SoVITS_weight_root, exist_ok=True)
    os.makedirs(GPT_weight_root, exist_ok=True)

    g_ref_audio_widget_list = []  # 参考音频播放控件列表
    g_ref_audio_path_list = []  # 当前批次参考音频地址列表
    g_ref_lang_widget_list = []  # 参考语言控件列表
    g_ref_text_widget_list = []  # 参考文本控件列表
    g_test_audio_widget_list = []   # 试听音频播放控件列表
    g_save_widget_list = []  # 满意按钮控件列表

    g_ref_list = []  # 参考音频文件列表
    g_ref_list_max_index = 0  # 参考音频文件列表索引最大值

    g_index = 0  # 当前的索引

    g_ref_folder, g_batch = args.folder, args.batch

    load_ref_list_file(args.list)

    if args.check_duration:
        remove_noncompliant_audio_from_list()  # 检查音频长度功能

    if args.random_order:
        random.shuffle(g_ref_list)  # 检查音频顺序功能

    g_SoVITS_names, g_GPT_names = get_weights_names()

    # 默认加载第一个模型
    if g_GPT_names and g_SoVITS_names:
        inference_main.change_gpt_weights(g_GPT_names[0])
        inference_main.change_sovits_weights(g_SoVITS_names[0])
    else:
        print("No model found! Please put your model into SoVITS_weights and GPT_weights.")
        exit()

    with gr.Blocks(title="GPT-SoVITS RefAudio Tester WebUI") as app:
        gr.Markdown(value="# GPT-SoVITS RefAudio Tester WebUI\nDeveloped by 2DIPW Licensed under GNU GPLv3 ❤ Open source leads the world to a brighter future!")
        with gr.Group():
            gr.Markdown(value="模型选择")
            with gr.Row():
                dropdownGPT = gr.Dropdown(label="GPT模型", choices=g_GPT_names, value=g_GPT_names[0],
                                          interactive=True)
                dropdownSoVITS = gr.Dropdown(label="SoVITS模型", choices=g_SoVITS_names, value=g_SoVITS_names[0],
                                             interactive=True)
                textboxOutputFolder = gr.Textbox(
                    label="满意的参考音频复制到",
                    interactive=True,
                    value="output/")
                btnRefresh = gr.Button("刷新模型列表")
                btnRefresh.click(fn=refresh_model_list, inputs=[], outputs=[dropdownSoVITS, dropdownGPT])
                dropdownSoVITS.change(inference_main.change_sovits_weights, [dropdownSoVITS], [])
                dropdownGPT.change(inference_main.change_gpt_weights, [dropdownGPT], [])
            gr.Markdown(value="合成选项")
            with gr.Row():
                textboxTestText = gr.Textbox(
                    label="试听文本",
                    interactive=True,
                    placeholder="用以合成试听音频的文本")
                dropdownTextLanguage = gr.Dropdown(
                    label="合成语种",
                    choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], value="中文",
                    interactive=True
                )
                dropdownHowToCut = gr.Dropdown(
                    label="切分方式",
                    choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"],
                    value="凑四句一切",
                    interactive=True
                )
                sliderTopK = gr.Slider(minimum=1, maximum=100, step=1, label="top_k", value=5, interactive=True)
                sliderTopP = gr.Slider(minimum=0, maximum=1, step=0.05, label="top_p", value=1, interactive=True)
                sliderTemperature = gr.Slider(minimum=0, maximum=1, step=0.05, label="temperature", value=1,
                                              interactive=True)
            gr.Markdown(value="试听批次")
            with gr.Row():
                sliderStartIndex = gr.Slider(minimum=0, maximum=g_ref_list_max_index, step=g_batch, label="起始索引",
                                             value=0,
                                             interactive=True, )
                sliderBatchSize = gr.Slider(minimum=1, maximum=100, step=1, label="每批数量", value=g_batch,
                                            interactive=False)
                btnPreBatch = gr.Button("上一批")
                btnNextBatch = gr.Button("下一批")
                btnInference = gr.Button("生成试听语音", variant="primary")
            gr.Markdown(value="试听列表")
            with gr.Row():
                with gr.Column():
                    for i in range(g_batch):
                        with gr.Row():
                            ref_no = gr.Number(
                                value=i,
                                visible=False)
                            ref_audio = gr.Audio(
                                label="参考音频",
                                visible=True,
                                scale=5
                            )
                            ref_lang = gr.Textbox(
                                label="参考文本语言",
                                visible=True,
                                scale=1
                            )
                            ref_text = gr.Textbox(
                                label="参考文本",
                                visible=True,
                                scale=5
                            )
                            test_audio = gr.Audio(
                                label="试听音频",
                                visible=True,
                                scale=5
                            )
                            save = gr.Button(
                                value="满意",
                                scale=1
                            )
                            g_ref_audio_widget_list.append(ref_audio)
                            g_ref_text_widget_list.append(ref_text)
                            g_ref_lang_widget_list.append(ref_lang)
                            g_test_audio_widget_list.append(test_audio)
                            save.click(
                                copy_proved_ref_audio,
                                inputs=[
                                    ref_no,
                                    ref_text,
                                    textboxOutputFolder
                                ],
                                outputs=[
                                    save
                                ]
                            )
                            g_save_widget_list.append(save)

            sliderStartIndex.change(
                change_index,
                inputs=[
                    sliderStartIndex,
                    sliderBatchSize
                ],
                outputs=[
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list
                ])

            btnPreBatch.click(
                previous_index,
                inputs=[
                    sliderStartIndex,
                    sliderBatchSize
                ],
                outputs=[
                    sliderStartIndex,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list
                ],
            )

            btnNextBatch.click(
                next_index,
                inputs=[
                    sliderStartIndex,
                    sliderBatchSize
                ],
                outputs=[
                    sliderStartIndex,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list
                ],
            )

            btnInference.click(
                generate_test_audio,
                inputs=[
                    textboxTestText,
                    dropdownTextLanguage,
                    dropdownHowToCut,
                    sliderTopK,
                    sliderTopP,
                    sliderTemperature,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list
                ],
                outputs=[
                    *g_test_audio_widget_list
                ]
            )

            app.load(
                change_index,
                inputs=[
                    sliderStartIndex,
                    sliderBatchSize
                ],
                outputs=[
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list
                ],
            )

    app.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        quiet=True,
        share=False,
        server_port=args.port
    )
