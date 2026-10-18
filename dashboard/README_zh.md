# XLS-R 音频深度伪造检测器 · Dashboard Demo

在你笔记本上运行的本地网页 dashboard。它既可位于论文工作区的 `03_Demo_Code/xlsr_dashboard`，也可位于公开仓库的 `dashboard/`；程序会自动寻找对应的 `pilot_la_to_itw`。里面跑的就是试点 `run_20261011_113206` 的检测器：同样的音频预处理（16 kHz、去静音、取前 6 秒、归一化），同样的 `wav2vec2-large-xlsr-53` 权重（commit `c3f9d88`），同样的逻辑回归探针（seed-42 训练子集、源域验证选出的层和 C、源域验证 EER 点作阈值）。In-the-Wild 的标签只用来显示对错，不参与任何训练或调参。

论文工作区中旧的 `..\audio_deepfake_demo.py`（wav2vec2-base + 合成数据）没有改动；公开仓库不包含该旧 demo。

## 四个页面

| 页面 | 内容 |
|---|---|
| **Detect** | 上传或现场录音，或者从 In-the-Wild 随机抽一条（可按真/假、说话人筛选）。显示判定、伪造分数、波形+频谱（灰色=被裁掉的静音，黄色斜线=超过 6 秒没分析的部分）、六个模型的对比、以及 13 个层各自的判定。抽 In-the-Wild 时还会显示真实标签和试点当时记录的分数。 |
| **Batch** | 一次拖入多条音频，出表格，可下载 CSV。 |
| **Pilot results** | 预登记结果表、同层比较表、试点的图。 |
| **About** | 检测器怎么训练的、怎么看界面、局限。 |

六个模型 = 试点的六个条件，各自用源域验证选出的层：No augmentation（第 2 层）、Codec（18）、Noise（10）、Channel（4）、Mix（20，预登记处理组，默认用它出判定）、All（20）。下拉框可以换成任一模型出判定，其他模型的分数始终都会显示。

## 第一次使用（约 5–10 分钟）

打开 **Anaconda Prompt**：

```bat
conda activate xlsr
cd /d 你的路径\Audio-Deepfake-Detection\dashboard
pip install -r requirements_dashboard.txt
set PILOT_DATA_ROOT=你的本地数据目录
set HF_HOME=%PILOT_DATA_ROOT%\hf_cache
python detector.py --check
```

也可以直接使用 `python detector.py --data-root 你的数据目录 --check`。模型已缓存且会议室网络不稳定时，
先设置 `set HF_HUB_OFFLINE=1`。

`--check` 会做三件事：

1. 用 pilot 配置中 `data_root` 下的 `pilot_la_to_itw\embeddings` 缓存训练/准备 6 个主模型入口和 3 个条件 × 13 层的逐层入口，并把缓存存到同一数据根目录的 `demo_cache\`。相同的 condition、layer 和 C 会复用已拟合模型。只有第一次需要，之后几秒就能启动。
2. 从 `HF_HOME` 加载已缓存的 XLS-R；未缓存时 Transformers 会尝试从 Hugging Face 下载公开模型。
3. 随机抽 5 条 In-the-Wild，把现场算出的分数和试点 `predictions_target.csv` 里记录的分数对比。最后应看到 `CHECK PASSED`。

## 启动

- 双击 `run_dashboard.bat`，或者在上面的 Anaconda Prompt 里运行 `python app.py`。
- 浏览器会自动打开 http://127.0.0.1:7860 。关掉命令行窗口即停止。

可选参数（两种启动方式都能加）：

| 参数 | 作用 |
|---|---|
| `--share` | 额外生成一个临时公网链接（有效期见启动时的提示），导师可以在自己电脑上打开。需要联网，且检测仍在你的笔记本上运行。 |
| `--host 0.0.0.0` | 同一局域网的其他设备可以用 `http://你的IP:7860` 访问。 |
| `--port 7861` | 7860 被占用时换端口。 |
| `--device cpu` | 不用显卡（慢一些，单条约几秒）。 |
| `--retrain` | 忽略缓存，重新训练探针。 |

## 给导师演示的建议顺序

1. **Detect → Random In-the-Wild clip → True label 选 Fake → Draw and analyse**：展示判定、真实标签、以及“现场分数与试点记录一致”，说明整个流程可复现。
2. 看下面的 **Score at every XLS-R layer**：同一条音频在不同层可能一会儿判真、一会儿判假，正好对应论文里的选层混杂问题。
3. 再抽几条 **Real**：可以看到 No augmentation（第 2 层）几乎把所有 In-the-Wild 都判成假（试点里它的 balanced accuracy 只有 0.505），而 Noise / Mix 好很多。
4. 用 **麦克风** 录一句自己的话，现场检测。
5. 需要数字时切到 **Pilot results**。

## 需要知道的局限

- 所有模型只在 ASVspoof 2019 LA（2019 年的 TTS/VC 攻击）上训练。对现在的语音克隆可能不准；在 In-the-Wild 上最好的模型 EER 也有约 17%。
- “Fake score” = sigmoid(logit − 阈值)，0.5 就是阈值。它不是校准过的概率。
- Noise 的优势是事后探索性发现，还需要确认性实验。
- 研究原型，不是取证工具，不能作为任何真实录音的证据。

## 常见问题

| 现象 | 处理 |
|---|---|
| 装完 gradio 后 transformers 或 numpy 报错 | `pip install transformers==4.44.2 numpy==1.26.4` 恢复试点版本 |
| 会议室没网，启动卡在加载模型 | 启动前 `set HF_HUB_OFFLINE=1` |
| 顶部显示 “In-the-Wild unavailable” | 检查 pilot 配置记录的 In-the-Wild 压缩包路径；其他功能不受影响 |
| 顶部显示 “Device: cpu” | `python -c "import torch; print(torch.cuda.is_available())"` 应为 True；否则按试点 RUN_GUIDE 重装 cu118 版 torch |
| `--check` 显示 CHECK WARNING | 现场分数和试点记录差得较多，把命令行输出发给我 |
| 显示 “Missing source embeddings” | `data_root\pilot_la_to_itw\embeddings` 被移动或删除；dashboard 需要试点第 2 步的特征 |

## 文件

| 文件 | 作用 |
|---|---|
| `app.py` | Gradio 界面 |
| `detector.py` | 检测引擎：直接调用试点 `src` 里的预处理、编码器和训练代码；也可单独运行 `--check` |
| `plots.py` | 波形/频谱、模型对比、逐层分数三张图 |
| `run_dashboard.bat` | 一键启动（自动激活 `xlsr` 环境） |
| `requirements_dashboard.txt` | 额外依赖（只有 gradio） |

dashboard 只读取仓库的 `pilot_la_to_itw`（或论文工作区中的同名试点文件夹），不会修改其中的结果文件。
