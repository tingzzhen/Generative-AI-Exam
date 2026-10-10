# 📚 生成式 AI 認證考試重點整理（概念與理解導向）

> 💡 **定位原則**：
> 1. 本筆記專注於「觀念理解、名詞定義與技術應用」，不涉及複雜數學計算。
> 2. 全面採用**「5 秒掃讀法（1 行流程箭頭 ＋ 極簡關鍵字表格 ＋ 折疊白話故事）」**，並於各章節首次登場時標註**專有名詞英文全稱**，助你考前極速複習。

---

<a id="top"></a>
## 📑 五大核心主題與章節目錄
- [第一部分 生成式AI基礎知識](#第一部分-生成式ai基礎知識)
  - [1.0 AI 技術發展時間軸與地圖](#10-ai-技術發展時間軸與地圖)
  - [1.1 神經網路基礎 感知器與反向傳播](#11-神經網路基礎-感知器與反向傳播)
  - [1.2 CNN vs RNN](#12-cnn-vs-rnn)
  - [1.3 RNN LSTM 與 GRU](#13-rnn-lstm-與-gru)
  - [1.4 VAE GAN 與 Seq2Seq](#14-vae-gan-與-seq2seq)
  - [1.5 Transformer 自注意力與解碼參數](#15-transformer-自注意力與解碼參數)
  - [1.6 NLP BERT GPT 與 LLM 演進](#16-nlp-bert-gpt-與-llm-演進)
  - [1.7 擴散模型 (Diffusion)](#17-擴散模型-diffusion)
  - [1.8 多模態 AI 與影片生成技術](#18-多模態-ai-與影片生成技術)
  - [1.9 模型預訓練 從無到有的知識建立](#19-模型預訓練-從無到有的知識建立)
  - [1.10 模型的運作 (推理) AI 怎麼看懂輸入與生出結果](#110-模型的運作-推理-ai-怎麼看懂輸入與生出結果)
  - [1.11 模型最佳化 訓練技巧與過擬合](#111-模型最佳化-訓練技巧與過擬合)
  - [1.12 資料清理與 ETL](#112-資料清理與-etl)
  - [1.13 正規化 vs 標準化](#113-正規化-vs-標準化)
  - [1.14 五大學習範式與經典演算法](#114-五大學習範式與經典演算法)
  - [1.15 模型評估指標 F1-score R-squared 與 BLEU](#115-模型評估指標-f1-score-r-squared-與-bleu)
  - [1.16 AIGC 核心概念](#116-aigc-核心概念)
- [第二部分 生成式AI能力強化](#第二部分-生成式ai能力強化)
  - [2.1 大語言模型 (LLM) 原理 微調與應用模式](#21-大語言模型-llm-原理-微調與應用模式)
  - [2.2 提示工程 In-Context Learning 與思維樹 (ToT)](#22-提示工程-in-context-learning-與思維樹-tot)
  - [2.3 程式邏輯理解 語法分析器](#23-程式邏輯理解-語法分析器)
- [第三部分 生成式AI應用技能](#第三部分-生成式ai應用技能)
  - [3.1 影片生成工具與模型](#31-影片生成工具與模型)
  - [3.2 資料視覺化設計原則](#32-資料視覺化設計原則)
  - [3.3 商業文案與簡報生成](#33-商業文案與簡報生成)
  - [3.4 AI 學習平台 Kaggle](#34-ai-學習平台-kaggle)
  - [3.5 自動化工具 Zapier](#35-自動化工具-zapier)
  - [3.6 音樂創作 AIVA](#36-音樂創作-aiva)
  - [3.7 程式生成 Copilot](#37-程式生成-copilot)
- [第四部分 生成式AI倫理法律](#第四部分-生成式ai倫理法律)
  - [4.1 深度偽造 (Deepfake)](#41-深度偽造-deepfake)
  - [4.2 歐盟 AI 法案 AI 治理與臺灣 TAIDE 計畫](#42-歐盟-ai-法案-ai-治理與臺灣-taide-計畫)
  - [4.3 資料治理三大目標與資安防護](#43-資料治理三大目標與資安防護)
  - [4.4 隱私法規 GDPR HIPAA 與個資法](#44-隱私法規-gdpr-hipaa-與個資法)
  - [4.5 模型可解釋性 LIME vs SHAP](#45-模型可解釋性-lime-vs-shap)
- [第五部分 生成式AI工具與開發商總覽](#第五部分-生成式ai工具與開發商總覽)
  - [5.1 文本與對話語言模型 (LLMs)](#51-文本與對話語言模型-llms)
  - [5.2 圖像生成與編輯平台 (Image Generation)](#52-圖像生成與編輯平台-image-generation)
  - [5.3 影片生成與剪輯工具 (Video Generation)](#53-影片生成與剪輯工具-video-generation)
  - [5.4 音樂與語音生成工具 (Audio and Music)](#54-音樂與語音生成工具-audio-and-music)
  - [5.5 程式輔助與工作流程自動化 (Coding and Automation)](#55-程式輔助與工作流程自動化-coding-and-automation)
  - [5.6 資料科學與開源社群平台 (Platforms and Communities)](#56-資料科學與開源社群平台-platforms-and-communities)
  - [5.7 開發商 vs 代表工具速查矩陣](#57-開發商-vs-代表工具速查矩陣)

---

## 第一部分 生成式AI基礎知識

### 1.0 AI 技術發展時間軸與地圖
* **核心名詞全稱**：
  * **鑑別式 AI（Discriminative AI）** vs **生成式 AI（Generative AI）**
* **時代演進主軸**：`1960~2010s 鑑別式 AI（理解/分類/辨識）` ➔ `2014 分水嶺（GAN/VAE/Seq2Seq）` ➔ `2017+ 生成式 AI 爆發（無中生有/多模態創作）`
* **三大技術里程碑**：`CNN（看懂世界：影像）` ➔ `RNN（聽懂語言：時序）` ➔ `Transformer（不只理解還能創作：生成基石）`

![圖1-1 AI技術發展地圖，垂直虛線的左邊為鑑別式AI，右邊為生成式AI](assets/ai_timeline_map.png)

* **視覺化分水嶺結構圖（垂直虛線左邊 vs 右邊）**：
```text
【1960 ~ 2010s 初：鑑別式 AI 時代】        ┃ 【2014 ~ 2020s：生成式 AI 爆發時代】
（核心：理解、分類、辨識、判斷）           ┃ （核心：無中生有、內容創作、多模態生成）
                                           ┃
                                           ┃  2023-24 ● GPT-4V & Sora (OpenAI 生影片工具)
                                           ┃  2022    ● Stable Diffusion (Stability AI 開源生圖)
                                           ┃  2021    ● CLIP & DALL·E (OpenAI 畫圖工具)
                                           ┃  2020    ● Diffusion Model (擴散模型原型)
                                           ┃  2020    ● GPT-3 (生成式 AI 大語言模型)
                                           ┃  2018    ● GPT-1 (生成式 AI)
                                           ┃  2018    ● BERT (⚠️ 特例：雖在2018，但屬「鑑別式AI」！)
                                           ┃  2017    ● Transformer (內含「自注意力」機制)
                                           ┃  2014    ● Seq2Seq (序列到序列)
                                           ┃  2014    ● VAE (變分自編碼器)
                                           ┃  2014    ● GAN (生成對抗網路)
  1998 ● CNN (卷積神經網路)                ┃
  1997 ● LSTM (長短期記憶)                 ┃
  1986 ● RNN (遞歸神經網路)                ┃
  1985 ● MLP (多層感知機)                  ┃
  1960 ● Perceptron (感知機)               ┃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1960      1970      1980      1990      2000      2010      2014(分水嶺)    2020+
```

* **圖 1-1 全節點速查表（含英文全稱）**：
  | 分水嶺 | 年代 | 模型 / 工具名稱（英文全稱） | AI 類型歸屬 | 核心地位與關鍵字 |
  | :--- | :---: | :--- | :---: | :--- |
  | **虛線左** | ~1960 | **感知機 (Perceptron)** | **鑑別式 AI** | 最早神經網路單元（<span style="color: #7aa2f7">單層線性分類</span>） |
  | **虛線左** | ~1985 | **多層感知機 (MLP, Multilayer Perceptron)** | **鑑別式 AI** | 引入<span style="color: #7aa2f7">隱藏層與反向傳播</span>，深層網路基礎 |
  | **虛線左** | ~1986 | **遞歸神經網路 (RNN, Recurrent Neural Network)** | **鑑別式 AI** | 具<span style="color: #7aa2f7">短期記憶</span>，處理時序與自然語言理解 |
  | **虛線左** | ~1997 | **長短期記憶 (LSTM, Long Short-Term Memory)** | **鑑別式 AI** | <span style="color: #7aa2f7">三個門控</span>解決長序列<span style="color: #e06c75">梯度消失</span>問題 |
  | **虛線左** | ~1998 | **卷積神經網路 (CNN, Convolutional Neural Network)** | **鑑別式 AI** | <span style="color: #7aa2f7">卷積＋池化</span>萃取空間特徵，專攻圖像分類 |
  | **虛線右** | ~2014 | **生成對抗網路 (GAN, Generative Adversarial Network)** | **生成式 AI** | <span style="color: #7aa2f7">生成器＋判別器</span>對抗，<span style="color: #7aa2f7">隨機雜訊</span>生逼真圖 |
  | **虛線右** | ~2014 | **變分自編碼器 (VAE, Variational Autoencoder)** | **生成式 AI** | 編碼壓縮＋<span style="color: #7aa2f7">機率分佈抽樣</span>生成 |
  | **虛線右** | ~2014 | **序列到序列 (Seq2Seq, Sequence-to-Sequence)** | **生成式 AI** | <span style="color: #7aa2f7">Encoder-Decoder</span> 架構，奠定機器翻譯基礎 |
  | **虛線右** | 2017 | **Transformer** | **生成式基石** | 內含「<span style="color: #7aa2f7">自注意力</span> (Self-Attention)」，支援<span style="color: #7aa2f7">平行運算</span> |
  | **虛線右** | 2018 | **BERT (Bidirectional Encoder Representations from Transformers)** | <span style="color: #e06c75">⚠️ 鑑別式 AI</span> | <span style="color: #e06c75">圖上唯一特例！</span> 雙向 Encoder，專攻閱讀理解與分類 |
  | **虛線右** | 2018/20 | **GPT-1 / GPT-3 (Generative Pre-trained Transformer)** | **生成式 AI** | <span style="color: #7aa2f7">單向 Decoder</span> 文字接龍，引爆大型語言模型 (LLM, Large Language Model) 浪潮 |
  | **虛線右** | 2020 | **擴散模型 (Diffusion Model)** | **生成式 AI** | <span style="color: #7aa2f7">正向加噪＋反向去噪</span>生圖，取代 GAN 成主流 |
  | **虛線右** | 2021 | **CLIP (Contrastive Language-Image Pre-training) & DALL·E** | **生成式 AI** | **OpenAI** <span style="color: #7aa2f7">畫圖工具</span>（跨模態圖文對齊生圖） |
  | **虛線右** | 2022 | **Stable Diffusion** | **生成式 AI** | **Stability AI** <span style="color: #7aa2f7">開源免費</span>生圖模型，可本地部署 |
  | **虛線右** | 2023/24 | **GPT-4V (GPT-4 with Vision) & Sora** | **生成式 AI** | **OpenAI** <span style="color: #7aa2f7">多模態與生影片工具</span>（60 秒連貫影片） |

* **三大代表技術定位速查表**：
  | 代表技術 | 技術類型 | 主要功能 | 典型應用場景 |
  | :--- | :--- | :--- | :--- |
  | **CNN** | **鑑別式 AI** | <span style="color: #7aa2f7">圖像分類</span> | 臉部辨識、光學字元辨識 (OCR, Optical Character Recognition)、自駕車視覺、醫療影像診斷 |
  | **RNN** | **鑑別式 AI** | <span style="color: #7aa2f7">語言理解</span> | 語音辨識、機器翻譯、對話系統 |
  | **Transformer** | **生成式 AI** | <span style="color: #7aa2f7">內容生成</span> | ChatGPT、DALL·E、AI 作曲等內容創作 |

* **1.0 考點精華標記**：
  * **🔥 考試常考**：
    1. **鑑別式陣營（分類/理解）**：Perceptron、MLP、RNN、LSTM、CNN、<span style="color: #e06c75">BERT</span>。
    2. **生成式陣營（無中生有）**：GAN、VAE、Seq2Seq、GPT 系列、Diffusion、DALL·E、Stable Diffusion、Sora。
    3. **OpenAI 工具定位**：**CLIP & DALL·E** 是<span style="color: #7aa2f7">畫圖工具</span>；**GPT-4V & Sora** 是<span style="color: #7aa2f7">多模態與生影片工具</span>。
  * **⚠️ 易錯陷阱**：
    1. **BERT 年代假象陷阱（必考！）**：BERT 雖於 2018 年誕生（位於分水嶺右側），但屬於「<span style="color: #e06c75">鑑別式 AI</span>」（專精閱讀理解與分類），<span style="color: #e06c75">絕非生成式 AI</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.1 神經網路基礎 感知器與反向傳播
* **核心名詞全稱**：
  * **Perceptron（感知器）**、**MLP（多層感知器, Multilayer Perceptron）**、**DNN（深度神經網路, Deep Neural Network）**
  * **BM / RBM（波爾茲曼機 / 受限波爾茲曼機, Boltzmann Machine / Restricted Boltzmann Machine）**
  * **Weight（權重）**、**Bias（偏移 / 偏差）**、**Loss Function（損失函數）**、**MSE（均方誤差, Mean Squared Error）**
  * **Backpropagation（反向傳播演算法）**、**Gradient Descent（梯度下降法）**
  * **XOR（互斥或, Exclusive OR）**、**SGD（隨機梯度下降, Stochastic Gradient Descent）**、**Adam（自適應矩陣估計, Adaptive Moment Estimation）**

* **1. 網路架構演進（從單層到多層）**：
  `感知器 Perceptron (1960)` ➔ `多層感知器 MLP (1980s)` ➔ `波爾茲曼機 RBM (1985)` ➔ `RNN / CNN`

  | 模型名稱 | 結構特徵 | 解決能力與歷史定位 | 關鍵限制 / 備註 |
  | :--- | :--- | :--- | :--- |
  | **感知器 (Perceptron)** | <span style="color: #7aa2f7">單層</span>（僅輸入＋輸出層） | 最早神經網路單元，只能畫一條直線做<span style="color: #7aa2f7">簡單線性分類</span> | <span style="color: #e06c75">無法解決 XOR</span>（互斥或, Exclusive OR）這種<span style="color: #e06c75">複雜非線性問題</span> |
  | **多層感知器 (MLP)** | **多層**（中間多加了<span style="color: #7aa2f7">隱藏層</span> Hidden Layer） | 能畫出彎曲分界線，<span style="color: #7aa2f7">成功解決複雜非線性問題</span> | 現代深層神經網路 (DNN, Deep Neural Network) 的標準原型 |
  | **波爾茲曼機 (RBM)** | 隨機生成式雙層網路 | 不用給標準答案，就能從無標籤資料中<span style="color: #7aa2f7">自動摸索出特徵分佈規律</span> | Hinton 用來幫深層網路「暖身預訓練」，突破早期訓練瓶頸 |

* **2. 神經網路訓練 4 步閉環（AI 怎麼學會調參數？）**：
  `① 權重與偏移（做預測）` ➔ `② 損失函數 / MSE（算誤差）` ➔ `③ 反向傳播（算梯度）` ➔ `④ 梯度下降（改參數）`

  | 訓練步驟與核心名詞 | 扮演角色 | 一句話白話秒懂（它到底在做什麼？） | 考試必背關鍵字 |
  | :--- | :--- | :--- | :--- |
  | **① 權重 (Weight) 與偏移 (Bias)** | **模型內部參數** | <span style="color: #7aa2f7">權重</span> $w$ 決定「這項特徵有多重要」；<span style="color: #7aa2f7">偏移</span> $b$ 是「基本起跳門檻」（如計程車起跳價 $y=wx+b$） | 訓練的終極目標就是找出最佳的 $w$ 與 $b$ |
  | **② 損失函數 (Loss) 與 MSE** | **誤差成績單** | 算出「AI 猜的答案跟真實答案差多少」；MSE (Mean Squared Error) 會把<span style="color: #7aa2f7">誤差平方後取平均</span>（錯越大罰越重） | MSE 用於<span style="color: #7aa2f7">迴歸數值預測</span>（如猜房價），數值越小越準 |
  | **③ 反向傳播 (Backpropagation)** | **從後往前抓戰犯** | 像改完考卷<span style="color: #7aa2f7">從最後結果往回檢查</span>，揪出哪個參數害答案算錯，並算出修正方向（梯度） | 只負責「<span style="color: #7aa2f7">反向傳遞誤差、計算梯度</span>」，<span style="color: #e06c75">不直接改權重</span> |
  | **④ 梯度下降 (Gradient Descent)** | **下山找最低谷** | 順著剛剛算出的下坡方向，<span style="color: #7aa2f7">實際動手修改權重</span> $w$ <span style="color: #7aa2f7">與偏移</span> $b$，讓下次猜得更準 | 搭配優化器（SGD, Stochastic Gradient Descent / Adam, Adaptive Moment Estimation）最小化 Loss |

<details>
<summary><b>💡 點擊展開白話案例：計程車起跳價與考數學檢討四部曲</b></summary>

> 🚕 **1. 權重 (Weight) 與偏移 (Bias) 的計程車比喻**：
> 搭計程車算車資 $y = w \cdot x + b$：里程數 $x$ 乘上的「每公里單價 $w$」就是**權重**（決定里程對車資的影響力）；而一上車不管開多遠都固定先收的「起跳價 85 元 ($b$)」就是**偏移**（基準門檻）。
>
> 📝 **2. 神經網路「考數學檢討四部曲」**：
> 1. **寫考卷（權重與偏移）**：學生（感知器/MLP）用腦中現有的公式參數 ($w, b$) 算出預測答案。
> 2. **對答案算扣分（損失函數 / MSE）**：老師批改考卷，算出預測跟標準答案差多少分（MSE 會把粗心大錯平方加倍扣分！）。
> 3. **倒推哪一步算錯（反向傳播）**：從最後結果一步步往回檢查，揪出到底是哪一層神經元的權重害答案算錯（算出梯度）。
> 4. **訂正改掉壞習慣（梯度下降）**：順著正確方向調整腦中的權重與偏移，下次考試誤差就變小了！
</details>

* **1.1 考點精華標記**：
  * **🔥 考試常考**：
    1. **單層感知器 vs MLP**：單層感知器只能解線性問題（<span style="color: #e06c75">無法解 XOR</span>）；MLP 加入「<span style="color: #7aa2f7">隱藏層</span>」後可解決<span style="color: #7aa2f7">非線性問題</span>。
    2. **波爾茲曼機 (BM / RBM)**：基於能量的隨機神經網路，用於學習<span style="color: #7aa2f7">資料機率分佈</span>與早期深層網路特徵預訓練。
    3. **MSE (均方誤差)**：最經典的<span style="color: #7aa2f7">迴歸任務損失函數</span>，將<span style="color: #7aa2f7">誤差平方後取平均</span>，數值越小代表預測越準。
  * **⚠️ 易錯陷阱**：
    1. **反向傳播 vs 梯度下降分工混淆（高頻陷阱！）**：**反向傳播**負責「<span style="color: #7aa2f7">由後往前算梯度</span>（找方向）」；**梯度下降**負責「<span style="color: #7aa2f7">順著梯度更新權重</span>（跨步下山）」，兩者<span style="color: #e06c75">一算一改，切勿記反</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.2 CNN vs RNN
* **核心名詞全稱**：
  * **CNN（卷積神經網路, Convolutional Neural Network）**
  * **RNN（循環神經網路, Recurrent Neural Network）**
  * **Recurrent Connections（記憶機制 / 循環連結）**、**Sequential Dependency（時序依賴性）**、**Vanishing Gradient（梯度消失）**
  * **OCR（光學字元辨識, Optical Character Recognition）**、**NST（神經風格轉換, Neural Style Transfer）**、**GPU（圖形處理器, Graphics Processing Unit）**
* **CNN 視覺辨識 3 步流水線**：`① 卷積層 (Convolutional Layer：掃描找特徵)` ➔ `② 池化層 (Pooling Layer：降維記重點)` ➔ `③ 全連接層 (Fully Connected Layer, FC：整合分類)`
* **RNN 時序理解 2 大關鍵機制**：`① 記憶機制 (Recurrent Connections：循環連結將前一刻資訊傳給現在的自己)` ＋ `② 時序依賴性 (Sequential Dependency：捕捉語音/翻譯/音樂的前後順序關係)`

* **CNN vs RNN 核心對照表**：
  | 比較維度 | 卷積神經網路 (CNN) | 循環神經網路 (RNN) |
  | :--- | :--- | :--- |
  | **專攻資料** | <span style="color: #7aa2f7">空間與網格資料</span>（影像、圖片、照片） | <span style="color: #7aa2f7">時間與序列資料</span>（文字、語音、音樂旋律） |
  | **核心機制** | <span style="color: #7aa2f7">卷積層</span>找局部特徵 ＋ <span style="color: #7aa2f7">池化層</span>降維保留重點 | <span style="color: #7aa2f7">記憶機制</span>（循環連結）＋ <span style="color: #7aa2f7">時序依賴性</span>（順序關係） |
  | **平行運算** | <span style="color: #7aa2f7">極佳</span>（可快速用 GPU, Graphics Processing Unit 平行加速） | <span style="color: #e06c75">極差</span>（必須依時間先後一個字一個字排隊算） |
  | **致命弱點** | <span style="color: #e06c75">不擅長捕捉長時序</span>前後文關聯 | <span style="color: #e06c75">梯度消失</span>（反向傳播時誤差訊號「<span style="color: #e06c75">指數級衰減</span>」，句子一長就忘記開頭講什麼） |
  | **典型應用** | 臉部辨識、**光學字元辨識 (OCR, Optical Character Recognition)**、自駕車視覺、<span style="color: #7aa2f7">神經風格轉換</span> (NST, Neural Style Transfer) | ① <span style="color: #7aa2f7">語音辨識</span> (Speech Recognition：**Google Assistant / Siri**)、② <span style="color: #7aa2f7">自動字幕生成</span> (Automatic Subtitling：**YouTube 即時字幕**)、③ <span style="color: #7aa2f7">機器翻譯</span> (Machine Translation：**Google 翻譯**)、音樂旋律生成 |

<details>
<summary><b>💡 點擊展開白話案例：CNN 拼圖偵探 vs RNN 傳話遊戲與三大時序應用（原著範例）</b></summary>

> 🔍 **1. CNN（拼圖偵探）**：
> 就像偵探拿放大鏡看照片，先用「卷積層」掃描貓咪的耳朵、鬍鬚（**找特徵**）；再用「池化層」濾除背景雜訊、縮小存檔（**記住重要資訊**）；最後「全連接層」整合所有特徵判定：「95% 是一隻貓！」。
>
> 🎧 **2. RNN 為何需要「記住過去」與「理解順序」？（時序依賴三例）**：
> * **語音識別**：AI 必須記住前面講過的單詞，才能聽懂整句話的意思（如 **Google Assistant、Apple Siri**）。
> * **機器翻譯**：AI 要根據前後語境，才能翻出語意連貫的句子（如 **Google 翻譯**）。
> * **音樂生成**：AI 必須記得前幾節音符，才能延續旋律的邏輯與風格。
>
> 📢 **3. RNN 的「傳話遊戲」悲劇（梯度消失 Vanishing Gradient）**：
> 想像全班玩「傳話遊戲」，前幾個人還能把訊息準確傳下去，但經過太多個人後，原本的內容就越傳越模糊走樣。RNN 處理長文字或長語音時也是如此——**誤差訊號在反向傳播過程中會「指數級衰減」**，假設主詞和形容詞隔了十幾個字，RNN 傳到後面早就把開頭主詞忘光了！
</details>

* **1.2 考點精華標記**：
  * **🔥 考試常考**：
    1. **CNN 三大層分工**：<span style="color: #7aa2f7">卷積層找特徵</span>、<span style="color: #7aa2f7">池化層記住重要資訊</span>（降維）、<span style="color: #7aa2f7">全連接層整合特徵</span>做出分類決策；延伸應用為<span style="color: #7aa2f7">神經風格轉換</span> (NST)。
    2. **RNN 兩大關鍵機制**：<span style="color: #7aa2f7">記憶機制</span> (Recurrent Connections，循環連結把前一刻資訊傳給現在的自己) 與 <span style="color: #7aa2f7">時序依賴性</span> (Sequential Dependency，捕捉順序關係)。
    3. **RNN 三大經典應用代表**：<span style="color: #7aa2f7">語音辨識</span>（Google Assistant、Siri）、<span style="color: #7aa2f7">自動字幕生成</span>（YouTube 即時字幕）、<span style="color: #7aa2f7">機器翻譯</span>（Google 翻譯）。
    4. **RNN 梯度消失數學本質**：誤差訊號在反向傳播過程中呈「<span style="color: #e06c75">指數級衰減</span>」，導致<span style="color: #e06c75">無法學習長距離依賴</span>關係。
  * **⚠️ 易錯陷阱**：
    1. CNN 與 RNN 傳統上皆屬於「<span style="color: #e06c75">鑑別式 AI</span>」，<span style="color: #e06c75">切勿誤選為生成式 AI</span>。
    2. 看到「<span style="color: #7aa2f7">空間特徵 / 風格轉換 (NST)</span>」選 **CNN**；看到「<span style="color: #7aa2f7">時間前後 / 循環連結 / 時序依賴 / 語音轉錄</span>」選 **RNN**！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.3 RNN LSTM 與 GRU
* **🔗 上下游串聯**：承接 1.2 節——為解決傳統 **RNN** 遇長句子會「<span style="color: #e06c75">梯度消失（忘記開頭）</span>」的弱點，工程師加裝「門控開關」改良出 **LSTM** 與 **GRU**。
* **核心名詞全稱**：
  * **RNN（循環神經網路, Recurrent Neural Network）**
  * **LSTM（長短期記憶模型, Long Short-Term Memory）**
  * **GRU（門控循環單元, Gated Recurrent Unit）**
* **序列模型三代演進**：`傳統 RNN（沒開關門，句子一長誤差訊號越來越弱就忘光）` ➔ `LSTM（裝 3 道開關門＋長途傳送帶，記住長句重點）` ➔ `GRU（精簡成 2 道門，LSTM 輕量加速版）`

* **序列模型三部曲速查表**：
  | 演進階段 | 結構與門控數量 | 記憶能力 | 運算速度 / 參數量 | 考試核心定位 |
  | :--- | :--- | :--- | :--- | :--- |
  | **傳統 RNN** | <span style="color: #e06c75">無門控</span>（像沒篩選機制的傳聲筒） | 僅短期記憶（句子一長誤差訊號越來越弱，造成<span style="color: #e06c75">梯度消失</span>） | 最快 / 參數最少 | 序列起點，<span style="color: #e06c75">長句前面講什麼必忘光</span> |
  | **LSTM** | <span style="color: #7aa2f7">3 個門</span>（遺忘門 Forget、輸入門 Input、輸出門 Output）＋ <span style="color: #7aa2f7">細胞狀態</span> | **長記憶**（決定什麼該記、什麼該忘，<span style="color: #7aa2f7">解決長期依賴</span> Long-term Dependencies） | 較慢 / 結構最繁重 | <span style="color: #7aa2f7">解決長句遺忘代表</span>，但門太多算得慢 |
  | **GRU** | <span style="color: #7aa2f7">2 個門</span>（更新門 Update Gate、重置門 Reset Gate） | **長記憶**（一樣能記住長句重點） | 較快 / <span style="color: #7aa2f7">比 LSTM 輕量</span> | <span style="color: #7aa2f7">LSTM 的瘦身簡化版</span>，參數少、算得快 |

* **LSTM 與 GRU 共同典型應用**：<span style="color: #7aa2f7">聊天機器人</span> (Chatbots)、<span style="color: #7aa2f7">語音轉文字</span> (Speech-to-Text)、<span style="color: #7aa2f7">文本摘要</span> (Text Summarization)。

<details>
<summary><b>💡 點擊展開白話案例：LSTM 的三道嚴格安檢門 vs GRU 的輕量行李箱</b></summary>

> 🚪 **LSTM（三道安檢門）**：
> 為了解決傳統 RNN 忘東忘西的毛病，LSTM 裝設了一條長途傳送帶（細胞狀態）和三道安檢門：「遺忘門」扔掉無用舊資訊、「輸入門」放行關鍵新資訊、「輸出門」決定現在輸出什麼。雖然能記得久，但門太多導致檢查很耗時（運算繁重）。
>
> 🧳 **GRU（精簡版行李箱）**：
> 工程師覺得 LSTM 太重了，幫它「瘦身」，把三道門精簡為「更新門」與「重置門」（只有兩道門），保留抗遺忘能力的同時，讓行李更輕、運算速度大幅飛躍！
</details>

* **1.3 考點精華標記**：
  * **🔥 考試常考**：LSTM 相比傳統 RNN 的最大優勢是「<span style="color: #7aa2f7">更好地處理長期依賴關係</span> (Long-term Dependencies)」。
  * **⚠️ 易錯陷阱**：LSTM 是 <span style="color: #7aa2f7">3 個門</span>（較複雜），GRU 才是 <span style="color: #7aa2f7">2 個門</span>的「<span style="color: #7aa2f7">簡化版</span>」，<span style="color: #e06c75">切勿將兩者門控數與複雜度記反</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.4 VAE GAN 與 Seq2Seq
* **🔗 上下游串聯**：承接 1.2～1.3 節——AI 從「只會辨識分類（鑑別式）」跨入 2014「無中生有（生成式）」分水嶺，誕生圖像生成雙雄（**VAE、GAN**）與結合 RNN/LSTM 的序列翻譯模型（**Seq2Seq**）。
* **核心名詞全稱**：
  * **AE（自編碼器, Autoencoder）** ➔ **VAE（變分自編碼器, Variational Autoencoder）**
  * **GAN（生成對抗網路, Generative Adversarial Network）**
  * **Seq2Seq（序列到序列模型, Sequence-to-Sequence）**
  * **Word Embedding（詞向量 / 詞嵌入）**、**Sentence Embedding（句向量 / 句嵌入）**、**LDM（潛在擴散模型, Latent Diffusion Model）**
* **VAE 機率生成 3 步流**：`① 編碼器 (Encoder：把圖片特徵壓縮成一個「機率範圍 μ, σ」，而非死背單一密碼)` ➔ `② 潛在空間抽樣 (Latent Space Sampling：從範圍內抽樣座標)` ➔ `③ 解碼器 (Decoder：還原畫出全新的相似圖片)`
* **Seq2Seq 向量空間翻譯 2 階段流（以 `How are you?` ➔ `你好嗎` 為例）**：
  `① 編碼階段 (Encoding：將詞/句轉為詞向量或句向量，壓縮成單一高維「語意向量 V₁」)` ➔ `② 解碼階段 (Decoding：解碼器依「語意向量 V₁ ＋ 目前已生成上下文」一詞一詞逐步預測：「你」➔「好」➔「嗎」)`
* **2014 早期生成三劍客演進**：`VAE（機率分佈抽樣生圖）` ＆ `GAN（對抗博弈生逼真圖）` ＆ `Seq2Seq（序列編解碼生文字）` ➔ `突破記憶與檢查瓶頸，催生 Attention 與 Transformer`

* **VAE vs GAN vs Seq2Seq 核心對照表（🔥 2014 生成分水嶺三大代表）**：
  | 比較維度 | VAE (變分自編碼器, ~2014) | GAN (生成對抗網路, 2014) | Seq2Seq (序列到序列, Google 2014) |
  | :--- | :--- | :--- | :--- |
  | **核心定位** | 早期機率式圖像生成與特徵壓縮（非監督式） | 早期高逼真圖像對抗生成基石 | 早期文字序列生成與機器翻譯基石 |
  | **運作架構** | <span style="color: #7aa2f7">編碼器</span> (Encoder) ＋ 潛在空間 ＋ <span style="color: #7aa2f7">解碼器</span> (Decoder) | <span style="color: #7aa2f7">生成器</span> (Generator) ＋ <span style="color: #7aa2f7">判別器</span> (Discriminator) 博弈對抗 | <span style="color: #7aa2f7">編碼器</span> (Encoder) ＋ 語意向量 $V_1$ ＋ <span style="color: #7aa2f7">解碼器</span> (Decoder) |
  | **招牌特色** | 學會資料的「<span style="color: #7aa2f7">潛在機率分佈</span>」（神韻範圍），從中抽樣生新圖，訓練穩定 | 像假鈔集團與警察過招，擅長從「<span style="color: #7aa2f7">隨機雜訊</span> (Random Noise)」畫出超逼真圖片 | 打破輸入與輸出「<span style="color: #7aa2f7">序列長度相等</span>」（字數要一樣多）的限制，在向量空間逐步生成 |
  | **代表應用** | 影像去噪、人臉表情平滑插值、潛在擴散模型 (LDM, Latent Diffusion Model) 壓縮底座 | **StyleGAN (NVIDIA)** 生成虛擬人臉、**Google** 黑白老照片上色 | 跨語言整句機器翻譯、早期語義理解與生成回答 |
  | **致命瓶頸** | 生成<span style="color: #e06c75">圖像邊緣較模糊</span> (Blurry，因追求平均穩健而帶柔焦感) | 訓練易失衡、<span style="color: #e06c75">模式崩潰</span> (Mode Collapse，只會畫同一種圖、沒變化) | 整句壓入<span style="color: #e06c75">單一固定向量</span>且<span style="color: #e06c75">無法回頭檢查</span>原句字詞，句子一長就忘光 |

* **跨領域橋樑與最終催生工具**：
  * **向量 (Vector) 與向量空間**：將影像特徵（如體型、耳朵大小）或文字轉為空間座標點（<span style="color: #7aa2f7">詞向量 / 句向量</span>）；特徵相似者<span style="color: #7aa2f7">向量距離近</span>，不同類者<span style="color: #7aa2f7">向量距離遠</span>，成為跨越影像分類與語言翻譯的橋樑。
  * **催生現代代表工具**：突破 Seq2Seq 限制後誕生 **Attention（注意力機制）** 與 **Transformer**，進而催生 **ChatGPT**（**OpenAI** 對話大模型）、**DALL-E**（**OpenAI** 文字生圖像模型）。

<details>
<summary><b>💡 點擊展開白話案例：貓狗 2D 向量空間、VAE 配方師、GAN 賽跑失衡與 Seq2Seq 長句失憶（原著範例）</b></summary>

> 🐾 **1. 貓和狗在 2D 向量空間的距離（原著圖 4-3 範例）**：
> 把 X 軸設為「體型大小」、Y 軸設為「耳朵大小」：波斯貓 ($A_1$) 與暹羅貓 ($A_2$) 體型和耳朵相似，座標點距離很近；兩隻貴賓狗 ($B_1, B_2$) 座標點也很靠近，但**貓和狗的向量距離很遠**。此時只要畫一條直線，就能清楚把左邊的貓和右邊的狗區分開來！
>
> 🧪 **2. 傳統 AE（影印壓縮包）vs VAE（掌握「神韻機率分佈」的配方師）**：
> 傳統 AE 像把貓照片壓成固定密碼 `88`，只能還原同一隻貓；**VAE** 則學會貓咪特徵的「**機率分佈範圍**」，隨機抽樣座標就能畫出全新的貓，只是因追求平均穩健，畫面常帶點「柔焦模糊感」。
>
> 🏃 **3. GAN 的「賽跑失衡」與模式崩潰 (Mode Collapse)（原著範例）**：
> 生成器（負責畫）與判別器（負責挑錯）像在賽跑：若**判別器太強**，生成器畫什麼都被嫌棄，永遠學不會；若**生成器跑太快**，判別器還沒搞清楚真假就輸了。這種「沒好好競爭」的失衡結果，會導致**模式崩潰**——AI 只學會畫出一種風格或幾張類似的圖，沒創意也沒變化！
>
> 📝 **4. Seq2Seq 怎麼翻譯 `How are you?`？為何遇長句就失憶？（原著範例）**：
> * **兩階段翻譯**：先在編碼階段把 `How are you?` 壓縮成代表整句意思的「語意向量 $V_1$」；解碼器再看著 $V_1$ 猜第一個字是「你」➔ 結合「你＋$V_1$」猜下一個字「好」➔ 再猜「嗎」，**一詞一詞拼湊出來**（絕非直接把英文向量換成中文向量找句子）。
> * **長句失憶悲劇**：短句如「我喜歡貓」壓成一個向量還 OK；但長句如「*昨天我去臺北車站地下一樓的誠品書店買了一本關於 ChatGPT 的新書，覺得超棒！*」硬塞進**單一向量**，記憶力就招架不住了，而且還**無法回頭檢查**原句某個字詞！
</details>

* **1.4 考點精華標記**：
  * **🔥 考試常考**：
    1. **VAE (變分自編碼器)**：透過「<span style="color: #7aa2f7">編碼器＋解碼器</span>」學習資料的<span style="color: #7aa2f7">潛在機率分佈</span>並抽樣生成新資料（訓練穩定，但<span style="color: #e06c75">圖像較模糊</span>）。
    2. **GAN (生成對抗網路)**：由「<span style="color: #7aa2f7">生成器＋判別器</span>」相互對抗，以從「<span style="color: #7aa2f7">隨機雜訊</span>」產生逼真圖片聞名；若兩者競爭失衡（一方太強或跑太快）會導致<span style="color: #e06c75">模式崩潰</span>（只畫單一風格、喪失多樣性）。
    3. **Seq2Seq 運作與瓶頸**：編碼階段將單詞/句子轉為<span style="color: #7aa2f7">詞向量 / 句向量</span>並壓成「<span style="color: #7aa2f7">語意向量</span> $V_1$」，解碼階段依「<span style="color: #7aa2f7">$V_1$ ＋ 已生成上下文</span>」<span style="color: #7aa2f7">一詞一詞逐步生成</span>；致命瓶頸為整句僅靠<span style="color: #e06c75">單一固定向量</span>且<span style="color: #e06c75">無法回頭檢查原句字詞</span>，長句易失憶。
  * **⚠️ 易錯陷阱**：
    1. **VAE vs GAN vs LLM 缺陷辨析**：**VAE** 缺點是「<span style="color: #e06c75">圖像較模糊</span>」；**GAN** 缺點是「<span style="color: #e06c75">模式崩潰</span>」；而「<span style="color: #e06c75">幻覺</span>」則是 **LLM** 的缺陷，三者切勿張冠李戴！
    2. **Seq2Seq 解碼陷阱**：Seq2Seq 翻譯<span style="color: #e06c75">並非</span>「將英文向量直接轉成中文向量找最接近句子」，而是透過解碼器<span style="color: #7aa2f7">一詞一詞逐步預測生成</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.5 Transformer 自注意力與解碼參數
* **🔗 上下游串聯**：承接 1.3～1.4 節——為徹底突破 **RNN** 無法平行運算與 **Seq2Seq** 單一向量長句失憶的瓶頸，2017 年 **Transformer** 橫空出世；而其第 ③ 層「**解碼層**」在做文字接龍選字時，則搭配三大「**解碼參數（Temperature / Top-K / Top-P）**」控制說話風格。
* **核心名詞全稱**：
  * **Attention Mechanism（注意力機制）**、**Transformer**（Google 2017 論文《Attention Is All You Need》提出）
  * **Input Embedding Layer（輸入嵌入層）**、**Self-Attention Layer（自注意力層）**、**Decoder Layer（解碼層）**
  * **Auto-regressive（自迴歸機制）**、**Multi-Head Attention（多頭注意力機制）**、**Positional Encoding（位置編碼）**
  * **Temperature（溫度參數）**、**Top-K Sampling（Top-K 取樣）**、**Top-P Sampling / Nucleus Sampling（Top-P 取樣 / 核取樣）**
* **Transformer 突破 RNN 與 Seq2Seq 的「三大絕招」**：
  `第 1 招：一次性整句處理（捨棄 RNN 逐字慢讀）` ＋ `第 2 招：每個詞都有自己獨立的向量（不再用單一向量壓縮整句）` ＋ `第 3 招：搭載「自注意力機制」（像聚焦的眼睛自動抓出重要詞彙）`
* **Transformer 生成與選字完整流水線**：
  `① 輸入嵌入層 (文字轉獨立向量)` ➔ `② 自注意力層 (Q×K 算關聯權重乘 V 抓重點)` ➔ `③ 解碼層 (靠「自迴歸機制」算出下一詞候選機率表)` ➔ `④ 解碼參數 (透過「溫度 / Top-K / Top-P」從機率表抽籤選出下一個字)`

* **1. Transformer 三大關鍵神經層與生成機制表（🔥 核心必考架構）**：
  | 核心神經層 / 生成機制 | 英文名稱 | 一句話白話秒懂（它到底在做什麼？） | 考試必背關鍵字 |
  | :--- | :--- | :--- | :--- |
  | **① 輸入嵌入層** | **Input Embedding Layer** | 把每個文字轉成電腦看得懂的數值座標（向量），讓意思相近的詞靠在一起 | <span style="color: #7aa2f7">每個詞都有獨立向量</span>（搭配位置編碼） |
  | **② 自注意力層** | **Self-Attention Layer** | 像讀文章拿螢光筆劃重點，自動算出句中每個詞跟其他詞的關聯有多強 | 自動聚焦<span style="color: #7aa2f7">抓出重要詞彙</span>與上下文語意關聯 |
  | **③ 解碼層** | **Decoder Layer** | 看懂上文重點後，一個字一個字往後預測並拼湊出完整目標句子 | 依靠<span style="color: #7aa2f7">自迴歸機制</span>將預測詞拼湊成完整句子 |
  | **自迴歸機制（生成核心）** | **Auto-regressive** | 每猜出一個新詞，就把「<span style="color: #7aa2f7">剛生出來的詞</span>」塞回輸入端，繼續猜「<span style="color: #7aa2f7">下一個詞</span>」 | Transformer <span style="color: #7aa2f7">生成句子</span>的核心機制（文字接龍） |

* **2. Transformer 核心特性速查表**：
  | 核心特性 / 元件 | 一句話白話運作邏輯 | 解決痛點與關鍵價值 |
  | :--- | :--- | :--- |
  | **平行化運算 (Parallel Processing)** | 不再像 RNN 一個字一個字排隊慢讀，而是<span style="color: #7aa2f7">一次性把整句話讀進去</span> | 發揮 GPU (Graphics Processing Unit) <span style="color: #7aa2f7">平行算力</span>，訓練速度飛躍 |
  | **自注意力機制 (Self-Attention)** | 讓句中每個詞同時跟<span style="color: #7aa2f7">其他所有詞互相比較算關聯</span>，看誰跟誰最有關 | 精準看懂長句子與代名詞指代，<span style="color: #7aa2f7">生成式 AI 最常搭配技術</span> |
  | **多頭注意力 (Multi-Head)** | 派出<span style="color: #7aa2f7">好幾組注意力「頭」</span>分工合作，同時從不同角度讀句子 | 同時兼顧文法結構、語氣與前後文語意 |
  | **位置編碼 (Positional Encoding)** | 因為整句一起讀會分不清誰前誰後，必須幫每個字貼上「<span style="color: #7aa2f7">位置順序號碼牌</span>」 | 彌補平行運算<span style="color: #e06c75">天生沒有前後詞序感</span>的缺陷 |

* **3. 解碼層怎麼挑出下一個字？——三大解碼參數（Temperature / Top-K / Top-P）白話速查表**：
  | 參數名稱 | 一句話白話秒懂（它在控制什麼？） | 怎麼運作？（直接看例子最懂） | 數值調整與說話風格 |
  | :--- | :--- | :--- | :--- |
  | **溫度 (Temperature)** | 調整 AI 回答要<span style="color: #7aa2f7">多冒險還是多保守</span>（像控制丟骰子瘋不瘋的旋鈕） | 決定解碼層要乖乖挑最安全的詞，還是大膽挑冷門有創意的詞 | <span style="color: #7aa2f7">設得低 (如 0.2)</span>：說話很保守、精準理性；<span style="color: #7aa2f7">設得高 (如 0.9)</span>：說話較放飛、大膽有創意 |
  | **Top-K 取樣** | <span style="color: #7aa2f7">保留前幾個高機率的詞</span>，從中選一個 | 例如設定 $K = 3$，AI 就只挑<span style="color: #7aa2f7">機率最高的前 3 個詞</span>來抽籤，把太冷門的詞直接踢掉 | $K$ 是<span style="color: #7aa2f7">固定數量</span>（挑前幾名）；缺點是<span style="color: #e06c75">名額死板固定</span>，不會隨語境變動 |
  | **Top-P 取樣 (核取樣)** | <span style="color: #7aa2f7">保留累積機率達到 P 的詞</span>，剩下的就不看了 | 例如設定 $P = 0.8$，AI 會從最可能的詞開始<span style="color: #7aa2f7">把機率加總到 80%</span>，其他的詞直接不看 | $P$ 是<span style="color: #7aa2f7">累加機率百分比</span>；能依當下語境<span style="color: #7aa2f7">自動伸縮入圍的候選詞數量</span> |

* **4. 解碼參數實務調校與 Transformer 應用大爆發**：
  * **解碼參數實務調校法則**：「溫度」與「Top-P」都是在控制選詞的隨機性，實務上建議<span style="color: #7aa2f7">二選一調整</span>就好，不要<span style="color: #e06c75">兩個同時改以免互相干擾</span>。
  * **分支 ① 語言理解高手 — BERT (Google, 2018)**：繼承 Transformer 的「<span style="color: #7aa2f7">Encoder 編碼器</span>」，專攻雙向閱讀理解、判斷與分類（屬<span style="color: #e06c75">鑑別式 AI</span>）。
  * **分支 ② 創作文學天才 — GPT (OpenAI, 2018)**：繼承 Transformer 的「<span style="color: #7aa2f7">Decoder 解碼器</span>」，專攻單向自迴歸文字接龍與創作（屬<span style="color: #7aa2f7">生成式 AI</span>）。

<details>
<summary><b>💡 點擊展開白話案例：速讀 5,000 字報告、抬頭看黑板、QKV 搜尋與 5 個候選詞算 Top-K / Top-P（原著範例）</b></summary>

> 📄 **1. 自注意力機制 ＝「速讀 5,000 字報告抓重點」（原著範例）**：
> 拿到一篇 5,000 字的報告要快速看懂，你不會一字一句死讀，而是：**先看標題**知道主題 ➔ **快速掃過內文抓出現最多的關鍵字**（如「AI」、「教育」、「政策」）➔ **最後看總結段落**把重點串聯起來。Transformer 的「自注意力機制」就像這雙聚焦的眼睛，根據不同詞彙互相比較、連結上下文，自動抓出整句話的重點！
>
> 📢 **2. 排隊傳話 (RNN) vs 抬頭看黑板 (Transformer) ＆ QKV 搜尋比喻**：
> * **RNN** 像全班排成一列傳話；**Transformer** 像老師把整句話一次投影在黑板上，全班**一次性整句閱讀**（平行運算），且每個詞都有自己的獨立筆記本（獨立向量）。
> * **QKV**：Query 是你在 YouTube 輸入的「搜尋關鍵字」，Key 是各影片的「標題標籤」，兩者比對算權重，最後取出影片的「具體內容 (Value)」。
>
> 📊 **3. 解碼層 5 個候選詞算給你看——Top-K (K=3) vs Top-P (P=0.8)（原著範例）**：
> 假設解碼層正在寫句子，算出下一個字有 5 個候選詞，機率由高到低分別是：`會 (50%)`、`是 (30%)`、`可能 (10%)`、`變得 (7%)`、`發生 (3%)`
> * **如果用 Top-K（設 $K = 3$）**：硬性規定**只留前 3 名** ➔ 留下「**會、是、可能**」3 個詞抽籤。
> * **如果用 Top-P（設 $P = 0.8$，即 80%）**：從第一名往下加總機率，「**會 (50%) ＋ 是 (30%) ＝ 80%**」剛好湊滿 80% 達標！只留「**會、是**」這 2 個詞，後面三個詞直接不看！
</details>

* **1.5 考點精華標記**：
  * **🔥 考試常考**：
    1. **Transformer 三大關鍵神經層與生成機制**：① <span style="color: #7aa2f7">輸入嵌入層</span>、② <span style="color: #7aa2f7">自注意力層</span>（生成式 AI 最常搭配技術）、③ <span style="color: #7aa2f7">解碼層</span>（依靠「<span style="color: #7aa2f7">自迴歸機制 Auto-regressive</span>」逐詞預測）。
    2. **Transformer 三大突破絕招**：① <span style="color: #7aa2f7">一次性整句平行處理</span>、② <span style="color: #7aa2f7">每個詞擁有獨立向量</span>、③ <span style="color: #7aa2f7">自注意力機制自動抓出重要詞彙</span>。
    3. **三大解碼參數**：<span style="color: #7aa2f7">低溫 (0.2) 保守精準 vs 高溫 (0.9) 放飛有創意</span>；**Top-K** 挑「<span style="color: #7aa2f7">機率最高的前 K 個詞</span>（固定數量）」；**Top-P** 把最可能的詞「<span style="color: #7aa2f7">累加機率到 P% 為止</span>（動態調整入圍詞數）」。
  * **⚠️ 易錯陷阱**：
    1. 生成式 AI 最常搭配<span style="color: #7aa2f7">自注意力機制</span>（切勿選 <span style="color: #e06c75">SVM、CNN 或模糊推理</span>）；且 Transformer <span style="color: #e06c75">天生沒有詞序資訊</span>，須靠<span style="color: #7aa2f7">位置編碼</span>補上。
    2. <span style="color: #e06c75">切勿把低溫誤認為更有創意</span>，也別把 Top-P 的 $P$ <span style="color: #e06c75">誤認為固定挑幾個字</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.6 NLP BERT GPT 與 LLM 演進
* **🔗 上下游串聯**：承接 1.5 節——2017 年 **Transformer** 問世後，2018 年同一年分化誕生兩位語言界巨星：繼承 **Encoder（編碼器）** 成為理解王 **BERT**，繼承 **Decoder（解碼器）** 成為創作王 **GPT** 與現代 **LLM**。
* **核心名詞全稱**：
  * **NLP（自然語言處理, Natural Language Processing）**：涵蓋 **NLU（自然語言理解, Natural Language Understanding）** 與 **NLG（自然語言生成, Natural Language Generation）**
  * **BERT（來自 Transformer 的雙向編碼語言表示模型, Bidirectional Encoder Representations from Transformers）**
  * **GPT（生成式預訓練 Transformer, Generative Pre-trained Transformer）**
  * **LLM（大型語言模型, Large Language Model）**
  * **MLM（遮蔽語言模型, Masked Language Model）**、**NSP（下一句預測, Next Sentence Prediction）**、**NER（命名實體辨識, Named Entity Recognition）**
* **前後因果演進流（先有 Transformer，才有 BERT 與 GPT！）**：
  `2017 年 Transformer 架構出現（AI 語言能力開外掛）` ➔ `2018 年同一年應用大爆發，誕生兩位語言界超級明星（名字最後一個字 T 都是 Transformer！）`：
  * **BERT（AI 的語言理解專家）**：繼承 Transformer 的「<span style="color: #7aa2f7">Encoder 編碼器</span>」，是那個「<span style="color: #7aa2f7">聽得懂你在說什麼</span>」的人（AI 界閱讀王）。
  * **GPT（AI 的創作文學天才）**：繼承 Transformer 的「<span style="color: #7aa2f7">Decoder 解碼器</span>」，是那個「<span style="color: #7aa2f7">會接話、能創作</span>」的人（超強寫作機器）。
* **詞向量演進（破解一詞多義 Polysemy）**：`獨熱編碼 (One-Hot Encoding：只給每個字編流水號，完全看不出字義遠近)` ➔ `靜態詞嵌入 (Static Word Embedding, Word2Vec [Word to Vector] / GloVe [Global Vectors]：一個詞只有固定一組數字，分不清「吃蘋果」與「蘋果手機」)` ➔ `動態上下文詞嵌入 (Contextualized Embedding, BERT/Transformer：依前後文動態換座標，徹底解決一詞多義)`
* **BERT 輸入向量合成**：`Token Embedding ([CLS] Classification 分類標記 / [SEP] Separator 分句標記)` ＋ `Segment Embedding (句A/句B)` ＋ `Position Embedding (位置)`

* **BERT vs GPT 雙雄終極對照表（🔥 Transformer 雙胞胎對決）**：
  | 比較維度 | BERT（AI 的語言理解專家） | GPT（AI 的創作文學天才） |
  | :--- | :--- | :--- |
  | **開發商 / 誕生年** | **Google** (2018) | **OpenAI** (2018，同年推出) |
  | **白話角色定位** | 「<span style="color: #7aa2f7">聽得懂你在說什麼</span>」的語言理解高手（AI 界閱讀王） | 「<span style="color: #7aa2f7">會接話、能創作</span>」的創作文學天才（超強寫作機器） |
  | **與 Transformer 關係** | 繼承 Transformer 的 <span style="color: #7aa2f7">Encoder</span>（<span style="color: #7aa2f7">自編碼模型</span> Autoencoder） | 繼承 Transformer 的 <span style="color: #7aa2f7">Decoder</span>（<span style="color: #7aa2f7">自迴歸模型</span> Autoregressive） |
  | **AI 類別歸屬** | <span style="color: #e06c75">⚠️ 鑑別式 AI</span> (Discriminative AI，因擅長判斷與分類) | <span style="color: #7aa2f7">生成式 AI</span> (Generative AI，擅長無中生有接龍寫作) |
  | **NLP 子領域** | <span style="color: #7aa2f7">NLU (自然語言理解, Natural Language Understanding)</span>：判斷語意、做分類、回答問題 | <span style="color: #7aa2f7">NLG (自然語言生成, Natural Language Generation)</span>：產出自然流暢的文章與對話 |
  | **語意方向性** | <span style="color: #7aa2f7">雙向</span> (Bidirectional)：同時參考句子的前後文來理解語境 | <span style="color: #7aa2f7">單向自迴歸</span> (Autoregressive)：每生出一個詞，就把整段話拿回來再接下一個 |
  | **訓練方式（怎麼練功？）** | 玩「<span style="color: #7aa2f7">猜字遊戲</span>」：① <span style="color: #7aa2f7">MLM（遮蔽語言模型</span>, **Masked Language Model**，靠前後文雙向猜遮住的字）＋ ② <span style="color: #7aa2f7">NSP（下一句預測</span>, **Next Sentence Prediction**，判斷兩句接不接得通） | 玩「<span style="color: #7aa2f7">文字接龍</span>」：<span style="color: #7aa2f7">次字預測</span> (**Next-Token Prediction**，給開頭預測下一個最自然的詞） |
  | **典型應用** | **Google 搜尋引擎核心**、情緒分析、關鍵字推薦、問答系統、**命名實體辨識 (NER, Named Entity Recognition)** | 寫新聞、說明書、小說、詩歌、聊天、解釋、翻譯、寫程式碼 |

* **BERT 經典變體速記**：
  * **RoBERTa (Robustly Optimized BERT Approach, Meta)**：<span style="color: #7aa2f7">拿掉 NSP 任務</span>、改用<span style="color: #7aa2f7">動態遮字 (Masking)</span> 並餵入更多文章，理解力更強。
  * **DistilBERT (Distilled BERT, Hugging Face)**：透過<span style="color: #7aa2f7">知識蒸餾</span>（大模型老師教小模型學生）縮小 40% 體積、快 60%，仍保留 97% 理解力。
  * **ALBERT (A Lite BERT, Google)**：讓<span style="color: #7aa2f7">不同層共用參數</span>來幫模型大幅瘦身、減少記憶體消耗。

* **大型語言模型 (LLM) 表現改進關鍵與多元影響（GPT-1 ➔ GPT-3 / LLaMA 演進核心）**：
  | 核心主題 | 關鍵考點結論 | 常見誘答與陷阱排除 |
  | :--- | :--- | :--- |
  | **LLM 表現改進關鍵** | <span style="color: #7aa2f7">擴展多樣化的語言模式和結構</span> | ❌ <span style="color: #e06c75">非單靠增加參數或提高算力</span>；❌ <span style="color: #e06c75">絕不可減少訓練資料</span> |
  | **多元 LLM 的技術影響** | <span style="color: #7aa2f7">提升生成內容的多樣性和品質</span> | ❌ LLM 本身「<span style="color: #e06c75">不會自產/提供更多訓練資料</span>」（資料由外部收集） |

* **LLM 演進核心邏輯**：`擴展多樣化的語言模式和結構（表現改進關鍵）` ➔ `提升生成內容的多樣性和品質`（💡 *臺灣繁中在地化 LLM 代表實踐：**國科會 TAIDE 計畫**，詳見 4.2 與 5.1 節*）

<details>
<summary><b>💡 點擊展開白話案例：BERT 猜字遊戲、GPT 文字接龍、蘋果的多重身分與讀萬卷書的文豪（原著範例）</b></summary>

> 🔍 **1. BERT 的「猜字遊戲」（遮蔽語言模型 MLM, Masked Language Model，原著範例）**：
> 輸入一句話：「`我 ___ 買了一本書`」，BERT 的任務就是猜出中間被遮蔽的那個詞（如「昨天」或「去書店」）。為了猜中這個詞，它會**同時參考句子的前後文**（也就是「**雙向**」理解語境），因此練就一身閱讀理解本事，成為 Google 搜尋引擎的核心之一！而且因為它擅長「判斷和分類」，所以被歸類為**鑑別式 AI**。
>
> ✍️ **2. GPT 的「文字接龍」（自迴歸模型 Autoregressive，原著範例）**：
> 給它一句開頭：「`我今天早上去吃了……`」，GPT 就會預測下一個最自然的詞，可能是「`拉麵`」、「`蛋餅`」或「`炒飯`」。**每生成一個詞，它會把整段話拿回來，再接下一個**——這種做法就叫做**自迴歸 (Autoregressive) 模型**，讓 GPT 成為幾乎無所不能的寫作機器！
>
> 🍎 **3. 蘋果的多重身分（動態詞向量破局）**：
> 傳統 Word2Vec 就像給每個單字發一張「永久身分證」，不管在哪個句子裡，「蘋果」都只有同一組固定數字。但基於 Transformer 的 BERT 能看到「我咬了一口蘋果」貼上「水果」向量；看到「蘋果推出新 iPhone」貼上「科技公司」向量——徹底解決一詞多義！
>
> 📚 **4. 為什麼 LLM 改進關鍵是「擴展多樣化的語言模式和結構」？（讀萬卷書的文豪）**：
> 增加參數像給作家「更大的腦袋」，提高算力像喝「提神飲料」；但如果他一輩子只讀同一本童話書，文筆永遠不會進步！唯有廣泛閱讀詩歌、論文、法律與程式碼（**擴展多樣化的語言模式和結構**），才能真正提升生成內容的多樣性和品質。
</details>

* **1.6 考點精華標記**：
  * **🔥 考試常考**：
    1. **歷史因果關係**：2017 年先有 <span style="color: #7aa2f7">Transformer</span> 架構，2018 年才從中分化出 **BERT**（繼承 <span style="color: #7aa2f7">Encoder 自編碼模型</span>）與 **GPT**（繼承 <span style="color: #7aa2f7">Decoder 自迴歸模型</span>）。
    2. **NLU vs NLG**：<span style="color: #7aa2f7">NLU 重理解</span>（代表：**BERT**，語言理解高手，屬<span style="color: #e06c75">鑑別式 AI</span>）；<span style="color: #7aa2f7">NLG 重生成</span>（代表：**GPT**，創作文學天才，屬<span style="color: #7aa2f7">生成式 AI</span>）。
    3. **訓練方式對比**：BERT 玩「<span style="color: #7aa2f7">猜字遊戲</span>」——<span style="color: #7aa2f7">遮蔽語言模型 (MLM, Masked Language Model)</span> ＋ <span style="color: #7aa2f7">下一句預測 (NSP, Next Sentence Prediction)</span>；GPT 玩「<span style="color: #7aa2f7">文字接龍</span>」——<span style="color: #7aa2f7">自迴歸次字預測 (Next-Token Prediction)</span>。
    4. **解決一詞多義**：靠 **BERT / Transformer** <span style="color: #7aa2f7">動態上下文詞嵌入</span>（Word2Vec <span style="color: #e06c75">靜態向量無法解決</span>）。
    5. **LLM 演進關鍵**：表現改進關鍵為「<span style="color: #7aa2f7">擴展多樣化的語言模式和結構</span>」；多元 LLM 發展能「<span style="color: #7aa2f7">提升生成內容的多樣性和品質</span>」。
  * **⚠️ 易錯陷阱**：
    1. **BERT** <span style="color: #e06c75">絕非生成式 AI</span>！因為它擅長判斷和分類，屬於<span style="color: #e06c75">鑑別式 AI</span>（雙向 Encoder）。
    2. **方向性切勿記反**：BERT 是<span style="color: #7aa2f7">雙向</span>（同時看前後文猜中間字）；GPT 是<span style="color: #7aa2f7">單向自迴歸</span>（每生一詞把整段話拿回來再接下一個）。
    3. **LLM 演進陷阱**：LLM 本身<span style="color: #e06c75">不會自己提供更多訓練資料</span>，且改進關鍵<span style="color: #e06c75">非單靠增加參數或減少訓練資料</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.7 擴散模型 (Diffusion)
* **🔗 上下游串聯**：承接 1.4 節——為解決早期 **GAN** 訓練易失衡「模式崩潰」與 **VAE**「畫面模糊」的痛點，2020 年 **擴散模型 (Diffusion)** 以穩定去噪生圖成為新一代視覺生成霸主。
* **核心名詞全稱**：
  * **Diffusion Model（擴散模型）**、**LDM（潛在擴散模型, Latent Diffusion Model）**
  * **SR（超解析重建, Super-Resolution）**
* **Diffusion 訓練與生成雙過程**：`正向過程 (Forward Process：清晰原圖 ➔ 逐步加噪破壞成純雜訊)` ➔ `反向過程 (Reverse Process：隨機純雜訊 ➔ 逐步去噪修復生成新圖)`
* **打敗 GAN 的 3 大優勢**：`① 訓練極穩定（無模式崩潰）` ＋ `② 畫質超細膩` ＋ `③ 提示詞 (Prompt) 精準導引創意`

* **主流圖像生成工具速查表**：
  | 工具名稱 | 開發商 | 授權性質 | 技術基礎 | 核心考點標籤 |
  | :--- | :--- | :--- | :--- | :--- |
  | **Stable Diffusion** | **Stability AI** | <span style="color: #7aa2f7">開源免費</span> (Open Source) | LDM (潛在擴散模型, Latent Diffusion Model) | <span style="color: #7aa2f7">可自由修改與本地部署</span> |
  | **DreamStudio** | **Stability AI** | 閉源網頁 (Closed Source) | LDM (Latent Diffusion Model) | 官方簡化介面，<span style="color: #7aa2f7">適合初學者入門</span> |
  | **Midjourney** | **Midjourney** | 閉源平台 | 自研 Diffusion 變體 | <span style="color: #7aa2f7">畫面風格穩定、藝術美感極強</span> |
  | **Leonardo AI** | **Leonardo.Ai** | 閉源平台 | 微調 Stable Diffusion | 擅長<span style="color: #7aa2f7">遊戲美術、概念圖與幻想風格</span> |
  | **DALL·E 2 / 3** | **OpenAI** | **閉源服務** | Diffusion + CLIP (Contrastive Language-Image Pre-training) | 語意對齊佳，常用於生成<span style="color: #7aa2f7">電影風格圖片</span> |
  | **Adobe Firefly** | **Adobe** | 商業閉源 | 自研 Diffusion | <span style="color: #7aa2f7">合法授權素材訓練，商用版權安全無虞</span> |

* **影像增強：超解析重建 (SR) vs 干擾項對照表**：
  | 處理方法 | 一句話白話效果 | 對圖片細節影響 | 考試選型定位 |
  | :--- | :--- | :--- | :--- |
  | **超解析重建 (SR, Super-Resolution)** | AI 自動腦補並補齊模糊照片遺失的細節紋理 | ✅ <span style="color: #7aa2f7">使圖片細節更清晰、提升解析度</span> | 「<span style="color: #7aa2f7">提升細節清晰度</span>」唯一正解 |
  | **降低解析度** | 把圖片像素砍少、縮小尺寸 | ❌ <span style="color: #e06c75">變模糊、產生馬賽克顆粒</span> | 僅用於縮小檔案體積 |
  | **模糊處理** | 把相鄰像素的顏色抹平柔化 | ❌ <span style="color: #e06c75">主動消除細節與邊緣</span> | 用於去雜點或背景虛化 |
  | **黑白轉彩色** | AI 猜測並塗上合理的顏色 | ❌ <span style="color: #e06c75">僅加上顏色，不提升清晰度</span> | 用於歷史黑白老照片上色 |

<details>
<summary><b>💡 點擊展開白話案例：電視雪花螢幕擦拭術（加噪 vs 去噪的繪畫奇蹟）</b></summary>

> 📺 **電視雪花螢幕擦拭術**：
> * **正向破壞（加噪）**：把一張虎斑貓照片一步步潑上黑白雪花雜訊，直到變成滿屏亂碼雪花點。
> * **反向還原（去噪作畫）**：AI 像神級修復師，盯著滿屏雪花點，依照提示詞「畫一隻虎斑貓」，一步步擦掉雜訊還原出輪廓、貓毛與鬍鬚，無中生有畫出全新影像！
</details>

<details>
<summary><b>💡 點擊展開白話案例：開放共創廚房 (Stable Diffusion) vs 封閉頂級私廚 (DALL-E)</b></summary>

> 🍳 **開源 vs 閉源比喻**：
> * **DALL-E（閉源私廚）**：OpenAI 的頂級私廚，只能點菜取餐，看不到食譜也無法自己改底層權重。
> * **Stable Diffusion（開放廚房）**：Stability AI 把完整食譜（開源權重）免費公開，人人都能搬回自己電腦（本地部署）微調新畫風！
</details>

<details>
<summary><b>💡 點擊展開白話案例：從實驗室走進日常生活的商業生成平台（原著範例）</b></summary>

> 🎨 **原著精選範例**：
> 像 Midjourney、Leonardo、Firefly 這樣的商業平臺，背後都在延續同一件事——**讓圖像生成從實驗室走進日常生活**。不需要會寫程式，只要輸入一句話，就能創造出專屬插畫。
</details>

* **1.7 考點精華標記**：
  * **🔥 考試常考**：
    1. **從雜訊逐步去噪生圖**：<span style="color: #7aa2f7">擴散模型</span> (Diffusion Model)（生成作畫時走的是<span style="color: #7aa2f7">反向過程</span> Reverse Process）。
    2. **使圖片細節更清晰**：<span style="color: #7aa2f7">超解析重建</span> (Super-Resolution)。
    3. **生成電影風格圖片常用平台**：<span style="color: #7aa2f7">DALL·E</span>（OpenAI）。
    4. **商用版權安全首選**：<span style="color: #7aa2f7">Adobe Firefly</span>（合法素材訓練）；**開源免費本地部署首選**：<span style="color: #7aa2f7">Stable Diffusion</span>（Stability AI）。
  * **⚠️ 易錯陷阱**：
    1. 問哪個平台常用於生成電影風格圖片，選 <span style="color: #7aa2f7">DALL·E</span>！切勿選 <span style="color: #e06c75">Notion</span>（筆記）、<span style="color: #e06c75">GPT-3</span>（純文字）或 <span style="color: #e06c75">Perplexity</span>（搜尋）。
    2. Stable Diffusion 的開發商是 <span style="color: #7aa2f7">Stability AI</span>，<span style="color: #e06c75">切勿誤選 OpenAI 或 Google</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.8 多模態 AI 與影片生成技術
* **🔗 上下游串聯**：結合 1.5～1.6 節的 **語言導演 (Transformer / GPT)** ＋ 1.7 節的 **畫圖工人 (Diffusion)** ＋ 1.2 節的 **語音模型 (RNN / TTS)**，讓原本獨立的模型「共同理解語意」，跨入同時處理文、圖、音、影的多模態時代。
* **核心名詞全稱**：
  * **Multi-modal AI（多模態人工智慧）**
  * **Guidance Module（引導模組）**、**Condition Controller（條件控制器）**
  * **Space-Time U-Net（時空 U-Net 架構，Google Lumiere 核心技術）**
  * **TTS（文字轉語音, Text-to-Speech）**、**Emotional TTS（情感語音合成）**

* **承先啟後演進脈絡（為什麼從單一模型走向「多模態 AI」？）**：
  * **什麼是多模態 (Multi-modal) AI？**：能**同時處理多種感官輸入**的 AI（例如：<span style="color: #7aa2f7">文字、聲音、圖像、影片</span>等）。人類日常感知世界本來就是多模態的——我們一邊聽話、一邊看表情，也會從語境中理解意思。
  * **與前章節的前後呼應（從各自獨立到共同理解語意）**：前面章節介紹的**語言模型 (Transformer / GPT)**、**影像模型 (CNN / Diffusion)**、**語音模型 (RNN / TTS)** 原本都是各自獨立訓練的；要讓 AI 真正理解人類並與人互動，就必須把這些原本獨立的模型結合起來，找到一種「<span style="color: #7aa2f7">共同理解語意</span>」的方式。這也是為什麼現在的大型 AI 模型（如 **Gemini**、**GPT-4o**、**Claude**）都朝向<span style="color: #7aa2f7">多模態整合</span>發展！
  * **核心技術基石**：多模態模型常結合「<span style="color: #7aa2f7">深度學習與神經網路</span> (Deep Learning & Neural Networks)」（<span style="color: #e06c75">非支援向量機 SVM, Support Vector Machine / 決策樹 / 遺傳演算法 / 字典規則</span>）。
  * **跨模態資料結構與讀圖 3 步曲**：
    * **資料結構差異**：**文字** ＝ <span style="color: #7aa2f7">一維線性序列</span>；**圖像** ＝ <span style="color: #7aa2f7">二維空間網格</span>；**影片** ＝ <span style="color: #7aa2f7">時間＋空間多維結構</span>。
    * **Transformer 讀懂圖像 3 步曲**：`① 圖片切塊 (Patches, 16×16)` ➔ `② 攤平為數值向量 (Vector)` ➔ `③ 加上位置編碼 (Positional Encoding)`。

* **影片生成黃金拍檔：為什麼要結合 `Diffusion（畫圖工人）` ＋ `Transformer（導演／編劇）`？**：
  * 👷 **Diffusion 模型 ＝ 「<span style="color: #7aa2f7">畫圖工人</span>」**：真正動手把畫面從雜訊一步步去噪畫出來的主力，負責把每一張（每一幀）畫面畫得精細清晰。
  * 🎬 **Transformer 模型 ＝ 「<span style="color: #7aa2f7">在旁邊指揮的導演或編劇</span>」**：它<span style="color: #e06c75">不直接動手畫畫</span>（<span style="color: #e06c75">不是主導整個畫面的產生</span>），而是站在旁邊指揮——幫忙**理解前後文、語意順序**，控制 AI 要照著哪段文字、或哪一幀接下來該畫什麼。技術上，Transformer 是當成一個「<span style="color: #7aa2f7">引導模組</span> (Guidance Module)」或「<span style="color: #7aa2f7">條件控制器</span> (Condition Controller)」加入影片生成流程！

* **Sora (OpenAI, 2024) 影片生成三層架構分工表**：
  | 架構層次 | 角色比喻 | 採用技術元件 | 核心職責與效果 |
  | :--- | :--- | :--- | :--- |
  | **第 1 層：單幀畫面細節** | 👷 **畫圖工人** | <span style="color: #7aa2f7">Diffusion 模型</span> | 負責實際動手作畫，確保每幀<span style="color: #7aa2f7">細節豐富、光影自然、畫質穩定</span> |
  | **第 2 層：跨幀時間一致性** | 🎬 **導演控場** | <span style="color: #7aa2f7">時序感知模組 ＋ Transformer</span> | 指揮前後幀動作連貫性，<span style="color: #7aa2f7">控制每一幀接下來畫什麼，防止畫面閃爍跳幀</span> |
  | **第 3 層：語意故事線對齊** | 📝 **編劇導引** | <span style="color: #7aa2f7">語意編碼 (Transformer 條件控制器)</span> | 理解長篇文字前後文與語意順序，作為<span style="color: #7aa2f7">引導模組</span>讓畫面精準對齊故事線 |

* **多模態影片生成系統技術比較表（原著核心對照）**：
  | 系統名稱 | 技術基礎 | 特點／應用方向 | 優勢特色與必考重點 |
  | :--- | :--- | :--- | :--- |
  | **Sora (OpenAI)** | <span style="color: #7aa2f7">Diffusion + Transformer</span>（畫圖工人＋導演編劇） | 文生成影片（<span style="color: #7aa2f7">語意驅動敘事</span>） | <span style="color: #7aa2f7">畫質佳、敘事強、世界理解力高</span>（可生成長達 60 秒高連貫電影級影片） |
  | **Runway Gen-2** (及 Gen-1 ~ Gen-4) | <span style="color: #7aa2f7">改良版 Diffusion</span> | <span style="color: #7aa2f7">創作者影片草圖、剪輯特效</span> | <span style="color: #7aa2f7">操作簡易、生成快速、商業應用已落地</span>（專門協助「<span style="color: #7aa2f7">非專業創作者</span>」） |
  | **Pika Labs (Pika)** | <span style="color: #7aa2f7">動畫導向 Diffusion + 多模態控制</span> | <span style="color: #7aa2f7">原創角色動畫、社群短片</span> | <span style="color: #7aa2f7">風格自由、互動性高、創作流程友善</span> |
  | **Google Lumiere** (及 Google Veo) | <span style="color: #7aa2f7">Space-Time U-Net</span>（時空 U-Net 架構） | <span style="color: #7aa2f7">高畫質敘事影片</span> | <span style="color: #7aa2f7">畫面穩定、邏輯一致、細節表現極佳</span>（一次生成整段時空連貫影片） |

* **情感語音合成 (Emotional TTS, Text-to-Speech) 關鍵**：
  * **最重要成功要素**：「<span style="color: #7aa2f7">情緒標註資料</span> (Emotionally Annotated Data)」（讓模型學會情緒對應的語調起伏）。
  * **干擾項排除**：<span style="color: #e06c75">音量均衡</span>（僅標準化大小聲）、<span style="color: #e06c75">語速一致</span>（反而呆板無情緒）、<span style="color: #e06c75">音高轉換</span> (Pitch Shifting，僅變聲無情感)。

<details>
<summary><b>💡 點擊展開白話案例：畫圖工人 (Diffusion) 搭檔導演編劇 (Transformer) 與照片拼圖讀字術（原著範例）</b></summary>

> 🎬 **1. 為什麼影片生成要把 Diffusion 和 Transformer 加在一起？（畫圖工人 vs 導演編劇）**：
> 在影片生成流程裡，主要的「**畫圖工人**」還是 **Diffusion 模型**（負責一筆一畫把每張畫面去噪畫出來）；但 **Transformer** 就像站在旁邊指揮的「**導演或編劇**」——它不直接畫畫，而是幫忙理解前後文、語意順序，作為「**引導模組**」或「**條件控制器**」，指揮畫圖工人要照著哪段文字劇本、或哪一幀接下來該畫什麼，讓整部影片連貫不跳戲！
>
> 🧩 **2. 讓 Transformer 讀懂照片的「拼圖切塊術」**：
> 圖像沒有自然語序，不能直接餵進 Transformer。工程師把圖片切成 $16 \times 16$ 像素的小拼圖 (Patches)，每塊攤平成一組向量，再加上「位置編碼」標記座標，Transformer 就能像讀一串單字一樣讀懂整張圖！
</details>

* **1.8 考點精華標記**：
  * **🔥 考試常考**：
    1. **多模態核心價值與技術**：結合原本獨立的語言、影像、語音模型達成「<span style="color: #7aa2f7">共同理解語意</span>」；底層結合「<span style="color: #7aa2f7">深度學習與神經網路</span>」；代表大模型為 **Gemini、GPT-4o、Claude**。
    2. **影片生成角色分工（必考比喻）**：<span style="color: #7aa2f7">Diffusion 是「畫圖工人」</span>（負責單幀畫質細節）；<span style="color: #7aa2f7">Transformer 是「導演或編劇」</span>（作為<span style="color: #7aa2f7">引導模組 / 條件控制器</span>，管前後文、語意順序與跨幀連貫）。
    3. **四大影片生成系統配對**：**Sora**（<span style="color: #7aa2f7">Diffusion + Transformer</span>，語意驅動敘事）、**Runway Gen-2**（<span style="color: #7aa2f7">改良版 Diffusion</span>，影片草圖/剪輯特效，助非專業創作者）、**Pika Labs**（<span style="color: #7aa2f7">動畫導向 Diffusion + 多模態控制</span>，角色動畫/社群短片）、**Google Lumiere**（<span style="color: #7aa2f7">Space-Time U-Net</span>，高畫質敘事影片，畫面穩定邏輯一致）。
    4. **情感語音生成最重要要素**：<span style="color: #7aa2f7">情緒標註資料</span>。
  * **⚠️ 易錯陷阱**：
    1. Transformer 在影片生成中是「<span style="color: #7aa2f7">引導模組 / 條件控制器（導演）</span>」，<span style="color: #e06c75">並非直接動手畫畫或主導整個畫面的產生</span>（畫圖工人是 Diffusion）！
    2. 情感語音切勿選「<span style="color: #e06c75">語速一致</span>」或「<span style="color: #e06c75">音量均衡</span>」；多模態技術切勿選「<span style="color: #e06c75">SVM、決策樹、遺傳演算法</span>」。

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.9 模型預訓練 從無到有的知識建立
* **🔗 上下游串聯**：總結 1.1～1.8 節的各類 AI 模型——介紹 AI 學徒如何透過 **6 大預訓練步驟** 從無到有學會知識，並與下一節 **1.10 模型的運作（推理）** 共用同一個「**5 大內部結構單元**」舞台。
* **核心名詞全稱**：
  * **Pre-training（模型預訓練：從無到有的知識建立過程）**
  * **Encoder（編碼器）**、**Latent Space Representation（潛在空間表示）**、**Decoder（解碼器）**
  * **Hyperparameters（超參數：層數、學習率、批次大小等）**、**Loss Function（損失函數）**
  * **Training Set（訓練集）**、**Validation Set（驗證集 / 練習考）**、**Test Set（測試集 / 期末考）**
  * **Prompt（提示 / 提示詞）**

* **1. 深度生成式 AI 的「5 大內部結構單元」**：
  * **為什麼要懂預訓練？**：生成式 AI 能寫詩、畫圖、聊天、寫程式，除了靠模型本身的學習能力，還必須依賴「<span style="color: #7aa2f7">大量資料</span>」與「<span style="color: #7aa2f7">完整的預訓練與推理流程</span>」——就像**一位 AI 學徒從白紙一張、什麼都不懂，一步步學會理解資料、進行創作，最後正式畢業上線**的成長歷程。
  * **5 大內部結構單元（不論文字、圖像、語音系統，預訓練與實際運作都共用這 5 個單元）**：
    `① 接收輸入 (Input Reception)` ➔ `② 編碼器 (Encoder)` ➔ `③ 潛在空間表示 (Latent Space Representation)` ➔ `④ 解碼器 (Decoder)` ➔ `⑤ 輸出 (Output)`

* **2. AI 學徒的成長歷程——模型預訓練「6 大步驟」總覽表**：
  | 步驟順序 | 預訓練步驟名稱 | 所在內部結構單元 | 原著白話定位 | 一句話看懂這一步在做什麼？ |
  | :---: | :--- | :---: | :--- | :--- |
  | **Step 1** | **資料收集與處理** | `① 接收輸入` | 📚 **先給 AI 學徒教材** | 收集照片、文字、語音等資料，並先經過「<span style="color: #7aa2f7">清洗、統一格式、去除錯誤</span>」（詳見 1.12 節），準備好乾淨教材 |
  | **Step 2** | **選擇適當模型** | `② 編碼器` | 🧠 **選對大腦才能聰明學** | 依任務選對 AI 模型（如 RNN、Transformer、GAN、Diffusion），並微調「<span style="color: #7aa2f7">超參數</span>」（層數、學習率、批次大小） |
  | **Step 3** | **決定損失函數** | `③ 潛在空間表示` | 📏 **設計一個評分標準** | 用「<span style="color: #7aa2f7">損失函數 (Loss Function)</span>」幫 AI 打分數，衡量預測結果與真實答案差多遠（<span style="color: #7aa2f7">選對損失函數＝選對訓練方向</span>） |
  | **Step 4** | **參數最佳化** | `③ 潛在空間表示` | 🔧 **錯了就調整、變聰明** | 與 Step 3 同步互動；算出誤差後，用「<span style="color: #7aa2f7">梯度下降</span>」與「<span style="color: #7aa2f7">反向傳播</span>」自動微調內部參數，讓下次更準（詳見 1.11 節） |
  | **Step 5** | **生成、評估與反覆運算** | `④ 解碼器` | 📝 **驗收成績、持續修正** | 透過訓練中的「內部生成」練習，並用「<span style="color: #7aa2f7">驗證集（練習考）</span>」與「<span style="color: #7aa2f7">測試集（期末考）</span>」驗收成績、反覆檢討修正 |
  | **Step 6** | **結果轉成文字（正式上線）** | `⑤ 輸出` | 🎓 **畢業正式上線應用** | 預訓練完成且表現穩定後，根據使用者輸入的「<span style="color: #7aa2f7">提示 (Prompt)</span>」產生新文字、圖片、音樂（如 ChatGPT、文生圖工具，詳見 1.10 節） |

* **3. Step 2 重點拆解：不同任務怎麼選對「大腦（模型）」與調「超參數」？**：
  * **選對模型大腦（就像畫漫畫要選筆、紙和畫板，而不是選油漆和刷子）**：
    * 要理解**時間順序**（像聽一首歌）➔ 用 <span style="color: #7aa2f7">RNN</span> **(循環神經網路, Recurrent Neural Network)**
    * 要**快速吸收全域訊息** ➔ 用 <span style="color: #7aa2f7">Transformer</span>
    * 要**創造新圖片** ➔ <span style="color: #7aa2f7">GAN</span> **(生成對抗網路, Generative Adversarial Network)** 是經典選擇、<span style="color: #7aa2f7">Diffusion</span> **(擴散模型)** 正當紅
  * **微調「超參數 (Hyperparameters)」**：選好大腦後，還要調整腦迴路與神經元的連結設定（例如：<span style="color: #7aa2f7">層數、學習率、批次大小</span>等），決定這位學徒的學習效率與風格。

* **4. Step 5 重點拆解：內部生成 vs 成績檢討會**：
  * **什麼是訓練中的「內部生成（模擬推理）」？**：模型在訓練中會執行預測下一個字、還原遮蔽詞、從雜訊重建圖片等動作——⚠️ 注意：這是模型在訓練中用來**算損失、調參數的學習方式**，<span style="color: #e06c75">不會對外輸出給使用者</span>！
  * **成績不理想時的「檢討會」**：若評估結果不好，就要從頭檢查（是不是<span style="color: #7aa2f7">資料不夠乾淨</span>？<span style="color: #7aa2f7">模型設計錯了</span>？還是<span style="color: #7aa2f7">學習率設太高</span>？），反覆修正直到穩定。

* **5. 三種資料集（教材與考卷）白話對照表（訓練集 vs 驗證集 vs 測試集）**：
  | 資料集名稱 | 原著白話比喻 | 用在哪個階段？ | 考試必記重點 |
  | :--- | :--- | :--- | :--- |
  | **訓練集 (Training Set)** | 📖 **平時上課的課本教材** | **Step 1 ~ Step 4** 上課學習時 | 讓 AI 直接看教材練習、算損失並更新模型參數 |
  | **驗證集 (Validation Set)** | 📝 **平時的「練習考 / 模擬考」** | **Step 5** 訓練過程中 | 幫我們「<span style="color: #7aa2f7">邊學邊調整模型與超參數</span>」，看有沒有走偏或死背 |
  | **測試集 (Test Set)** | 🏆 **最後的「期末考」** | **Step 5** 訓練完成後的總驗收 | 拿模型「<span style="color: #7aa2f7">從沒見過的新題目</span>」來驗收最終真實成績（<span style="color: #e06c75">絕對不能拿來調參數作弊</span>！） |

<details>
<summary><b>💡 點擊展開白話案例：AI 學徒從「白紙一張」到「畢業上線」的 6 步修行故事（原著範例）</b></summary>

> 🧑‍🎨 **把模型預訓練想成「培養一位 AI 學徒」**：
> 1. **Step 1 給教材（接收輸入）**：不能把髒亂錯誤的廢紙丟給學徒，要先幫他把照片、文字、語音**清洗乾淨、統一格式**。
> 2. **Step 2 選大腦與調腦迴路（編碼器）**：就像**畫漫畫要選筆、紙和畫板，而不是拿油漆刷**——聽音樂看時間順序選 **RNN**、快速讀懂全篇選 **Transformer**、畫新圖選 **GAN / Diffusion**，還要幫他調好「**超參數**」（層數、學習率、批次大小）。
> 3. **Step 3 定評分標準 ＋ Step 4 錯了就改（潛在空間）**：先用「**損失函數**」幫學徒的作品打分數，再用「**梯度下降＋反向傳播**」告訴他哪裡畫歪了、怎麼修正變聰明。
> 4. **Step 5 練習考與期末考檢討會（解碼器）**：學徒在教室裡練習猜字、修圖（**內部生成，不對外公開**），先考「**驗證集（練習考）**」邊考邊調，再考「**測試集（期末考）**」驗收沒看過的新題；考不好就開檢討會找原因（教材太髒？大腦選錯？學習率太高？）。
> 5. **Step 6 正式畢業出道（輸出）**：學徒成績穩定畢業後，只要客人給一句「**提示 (Prompt)**」，就能立刻寫文章、畫美圖、譜曲（如 ChatGPT、Midjourney）！
</details>

* **1.9 考點精華標記**：
  * **🔥 考試常考**：
    1. **深度生成式 AI 的 5 大內部結構單元**：`接收輸入 ➔ 編碼器 ➔ 潛在空間表示 ➔ 解碼器 ➔ 輸出`。
    2. **模型預訓練 6 大步驟順序**：`① 資料收集與處理 ➔ ② 選擇適當模型 ➔ ③ 決定損失函數 ➔ ④ 參數最佳化 ➔ ⑤ 生成、評估與反覆運算 ➔ ⑥ 正式上線應用（結果輸出）`。
    3. **任務與模型大腦配對**：理解時間順序選 <span style="color: #7aa2f7">RNN</span>、快速吸收全域訊息選 <span style="color: #7aa2f7">Transformer</span>、創造新圖片選 <span style="color: #7aa2f7">GAN / Diffusion</span>；微調「<span style="color: #7aa2f7">超參數</span>」包含<span style="color: #7aa2f7">層數、學習率、批次大小</span>。
    4. **練習考 vs 期末考**：<span style="color: #7aa2f7">驗證集 (Validation Set)</span> 像「**練習考**」（邊學邊調整模型）；<span style="color: #7aa2f7">測試集 (Test Set)</span> 像「**期末考**」（用從沒見過的新題驗收成績）。
  * **⚠️ 易錯陷阱**：Step 5 訓練過程中的生成動作（如預測下一字、還原遮蔽詞、從雜訊重建圖片）屬於「<span style="color: #7aa2f7">內部生成 / 模擬推理</span>」，目的是幫模型算損失與調參數，<span style="color: #e06c75">不會直接對外輸出給使用者</span>（Step 6 才會根據 Prompt 正式對外輸出）！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.10 模型的運作 (推理) AI 怎麼看懂輸入與生出結果
* **🔗 上下游串聯**：承接 1.9 節預訓練的「**Step 6 正式上線**」——當 AI 畢業後，面對使用者提問時，如何在同一個「**5 大內部結構單元**」舞台上執行 5 步推理並生出答案。
* **核心名詞全稱**：
  * **Inference（推理：模型執行與生成的運作流程總稱）**
  * **Latent Vector Space（潛在向量空間 / 潛在空間表示 Latent Space Representation）**
  * **Token（標記：模型思考語言的最小拼圖單位）**、**Tokenization（標記化：將句子拆解成 Token 的動作）**
  * **Contextual Embedding（帶有語境資訊的語意向量 / 語境嵌入向量）**

* **1. 與 1.9 預訓練的前後呼應：同一個「5 大單元舞台」，從「上學練功」到「上場工作」**：
  * **什麼是「推理 (Inference)」？**：當 AI 完成預訓練（1.9 節）正式畢業上線後，當你要求它「幫我寫一首詩」或「明天的天氣？」，AI 從**看懂輸入、腦中思考到生出結果**的所有執行與生成運作，總稱為「<span style="color: #7aa2f7">推理 (Inference)</span>」。
  * **為什麼說它像「翻譯工作」？**：整個推理流程其實就像**翻譯官**——先把輸入的「<span style="color: #7aa2f7">人類世界的資料</span>」翻譯成「<span style="color: #7aa2f7">模型能懂的高維度向量</span>」，在**潛在向量空間（AI 的思考中心）**裡處理並得出結果，再**翻譯回人類可讀的文字、圖像或聲音**端上桌！（💡 *註：這套流程除生成式 AI 外，也同樣適用於有使用「潛在向量空間」的**鑑別式 AI 模型**！*）
  * **推理運作 5 大步驟必背順序（對應 5 大內部結構單元）**：
    `① 接收輸入` ➔ `② 資料編碼` ➔ `③ 語意理解` ➔ `④ 生成解碼` ➔ `⑤ 輸出回應`

* **2. 模型推理運作「5 大步驟」與內部結構單元總覽表**：
  | 步驟順序 | 推理運作步驟名稱 | 所在內部結構單元 | 原著白話外號 | 一句話看懂這一步在做什麼？ |
  | :---: | :--- | :---: | :--- | :--- |
  | **Step 1** | **接收輸入** | `① 接收輸入` | 👀 **AI 的感官啟動** | 打開眼睛耳朵接收原始訊號（文字的 <span style="color: #7aa2f7">Token 標記</span>、圖片的<span style="color: #7aa2f7">像素</span>、聲音的<span style="color: #7aa2f7">波形</span>）；此時模型<span style="color: #e06c75">還無法直接理解意思</span> |
  | **Step 2** | **資料編碼** | `② 編碼器` | 🌐 **翻譯成模型能懂的向量** | 用編碼器（**RNN / LSTM / GRU / Transformer**）抓住重點與脈絡，把輸入轉成潛在空間裡的<span style="color: #7aa2f7">高維度多維向量 (Vector)</span> |
  | **Step 3** | **語意理解** | `③ 潛在空間表示` | 🧠 **AI 的內心世界（思考中心）** | 在高維度潛在空間中先不急著產出結果，而是進行<span style="color: #7aa2f7">理解、推理與概念運算</span>（如比較貓狗、開心難過、畫風差異），建立內部語意 |
  | **Step 4** | **生成解碼** | `④ 解碼器` | ✍️ **從腦海拉出一句完整的話** | 解碼器根據預訓練知識與潛在語義開始產出：ChatGPT <span style="color: #7aa2f7">一個字一個字生出回應</span>、圖像模型<span style="color: #7aa2f7">一步步畫出像素</span>、語音模型<span style="color: #7aa2f7">生成音訊波形</span> |
  | **Step 5** | **輸出回應** | `⑤ 輸出` | 🍽️ **AI 把「答案」端上桌** | 綜合考慮<span style="color: #7aa2f7">語意一致性、上下文邏輯與使用者期待</span>，把解碼器的生成結果轉換成有用又有感的文字、圖片或語音呈現給使用者 |

* **3. 實際例子 5 步驟拆解：當你問 AI「明天的天氣？」➔ 回答「明天可能會下雨」**：
  | 推理步驟 | 🌤️ 問 AI「明天的天氣？」實際發生了什麼事？（白話拆解） | 必考核心技術名詞 |
  | :---: | :--- | :--- |
  | **(1) 接收輸入** | 先把「明天的天氣？」拆解成較小的語言拼圖零件：`「明天」、「的」、「天氣」、「？」`（或更細的單字），送進模型 | <span style="color: #7aa2f7">Token（標記：思考語言基本單位）</span>、<span style="color: #7aa2f7">Tokenization（標記化）</span> |
  | **(2) 資料編碼** | Token 通過 Transformer 自注意力編碼層，判斷「明天」是時間、「天氣」是主題、「？」代表問句，轉成**帶有語境資訊的向量** | <span style="color: #7aa2f7">Contextual Embedding（語境嵌入向量）</span> |
  | **(3) 語意理解** | 整句話轉成一組高維度潛在向量，放進 AI 的內部語意地圖，確立這句話的意圖是「**詢問明天天氣如何**」 | <span style="color: #7aa2f7">Latent Space（潛在空間語意地圖）</span> |
  | **(4) 生成解碼** | 解碼器像玩**文字接龍**，一次生成一個 Token：先產出 `「明」` ➔ 強語意連結產出 `「天」` ➔ 常見句型產出 `「可」` ➔ 依序接出 `「能」「會」「下」「雨」` | <span style="color: #7aa2f7">機率分佈計算</span> ＋ <span style="color: #7aa2f7">逐字文字接龍</span>（依賴前面已產生的字） |
  | **(5) 輸出回應** | 把產出的 Token 拼成完整句子「**明天可能會下雨**」送出——這是**一步步逐字推理計算**出來的，<span style="color: #e06c75">絕非事先背好答案直接吐出</span>！ | <span style="color: #7aa2f7">逐字計算推理生成</span> |

* **4. 🔥 必考觀念澄清：AI 回答「明天可能會下雨」，是真的去查氣象局資料嗎？**：
  * **這要看你問的是「哪一種 AI 模型」**：
    1. **不連網的純語言模型（如一開始推出的純 ChatGPT）**：<span style="color: #e06c75">不會去查氣象局資料！</span>它的回答只是「<span style="color: #7aa2f7">根據預訓練資料中的統計語言知識</span>」，用機率預測出一個看起來最合理、最通順的句子（因為訓練語料裡問天氣常接「可能會下雨」）。
    2. **有網路連線能力的 AI 系統（如 ChatGPT-4o 結合連網搜尋）**：才會真正去搜尋外部網站，查詢「你所在地明天的真實天氣」來回答你。

<details>
<summary><b>💡 點擊展開白話案例：翻譯官的大腦思考中心 ＆「明天的天氣？」文字接龍實況（原著範例）</b></summary>

> 🌐 **1. 把 AI 推理當成「跨界翻譯官」**：
> AI 其實看不懂中文字或照片，所以它工作時分三階段：① **Step 1~2（聽見並翻譯成腦波）**：把你說的「明天的天氣？」切成小拼圖 (**Token**)，用編碼器翻譯成帶有前後文語境的高維度密碼 (**Contextual Embedding**)；② **Step 3（思考中心運轉）**：在「**潛在空間**」這個內心世界裡搞懂你想問明天天氣；③ **Step 4~5（文字接龍說出答案）**：解碼器根據前一個字算下一個字的機率，像接龍一樣一個字一個字吐出 `「明」➔「天」➔「可」➔「能」➔「會」➔「下」➔「雨」`！
>
> ☔ **2. 為什麼沒連網的 AI 敢回答你「明天可能會下雨」？**：
> 因為純語言模型是個「**統計接話王**」，不是氣象預報員！如果不連網，它根本不知道明天有沒有鋒面，只是因為以前讀過幾億篇文章裡，「明天…」後面很高機率接著「可能會下雨」，所以它就順著統計機率接出來了！
</details>

* **1.10 考點精華標記**：
  * **🔥 考試常考**：
    1. **模型推理 (Inference) 5 大步驟必背順序**：`① 接收輸入 (Input Reception) ➔ ② 資料編碼 (Encoder) ➔ ③ 語意理解 (Latent Space) ➔ ④ 生成解碼 (Decoder) ➔ ⑤ 輸出回應 (Output)`。
    2. **Token 與 Contextual Embedding**：模型思考語言的最小拼圖單位叫 <span style="color: #7aa2f7">Token（標記）</span>，拆解動作叫 <span style="color: #7aa2f7">Tokenization（標記化）</span>；經過編碼層轉換成帶有語境資訊的向量叫 <span style="color: #7aa2f7">Contextual Embedding</span>。
    3. **編碼器四大常見結構**：<span style="color: #7aa2f7">RNN</span>（記住前一詞順序）、<span style="color: #7aa2f7">LSTM</span>（強化記憶防忘太快）、<span style="color: #7aa2f7">GRU</span>（簡化輕巧版 LSTM）、<span style="color: #7aa2f7">Transformer</span>（自注意力抓全域重點脈絡）。
  * **⚠️ 易錯陷阱**：
    1. **Step 1 原始訊號陷阱**：在「接收輸入」階段，文字 Token、圖片像素、聲音波形還只是原始訊號，此時模型<span style="color: #e06c75">還無法直接理解輸入的資料</span>（要經過編碼器轉成向量進入潛在空間才能理解）！
    2. **不連網模型陷阱**：不連網的基礎語言模型回答問題，是靠「<span style="color: #7aa2f7">預訓練資料中的統計語言知識與機率分佈逐字推理</span>」，<span style="color: #e06c75">並不是背好標準答案直接吐出</span>，也<span style="color: #e06c75">不會自己去查外部即時氣象資料</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.11 模型最佳化 訓練技巧與過擬合
* **🔗 上下游串聯**：直接對應 1.9 節預訓練的「**Step 4 參數最佳化**」與 1.1 節「**梯度下降**」——專門解決模型在訓練時卡關的 4 大問題（梯度消失/爆炸、過擬合/欠擬合）。
* **核心名詞全稱**：
  * **Vanishing Gradient（梯度消失）**、**Exploding Gradient（梯度爆炸）**
  * **Overfitting（過擬合）**、**Underfitting（欠擬合）**、**Generalization Ability（概括能力 / 泛化能力）**
  * **Activation Function（啟動函數，如 Sigmoid、ReLU [線性整流函數, Rectified Linear Unit]）**
  * **Optimizer（最佳化器 / 優化器）**：**SGD（隨機梯度下降, Stochastic Gradient Descent）**、**Momentum（動量法）**、**RMSProp（均方根傳播, Root Mean Square Propagation）**、**Adam（自適應矩陣估計, Adaptive Moment Estimation）**
  * **BN（批量正規化, Batch Normalization）**、**ResNet（殘差網路, Residual Network，引入跳接結構 Skip Connection）**、**Weight Initialization（權重初始化）**
  * **Normalization（正規化）** vs **Regularization（正則化：L1 / L2 正則化）**、**Dropout（隨機丟棄）**、**Data Augmentation（資料增強）**、**Early Stopping（早停法）**

* **1. AI 預訓練常見的「4 大卡關問題」**：
  1. <span style="color: #e06c75">梯度消失</span> **(Vanishing Gradient)**：**坡太緩、走得慢**——更新訊號往回傳時越變越小，模型學不動。
  2. <span style="color: #e06c75">梯度爆炸</span> **(Exploding Gradient)**：**坡太陡、滑過頭**——更新訊號往回傳時暴衝失控。
  3. <span style="color: #e06c75">過擬合</span> **(Overfitting)**：「<span style="color: #e06c75">太會背，學得太像</span>」——把訓練資料當成考古題死背，一遇到新資料就完蛋。
  4. <span style="color: #e06c75">欠擬合</span> **(Underfitting)**：「<span style="color: #e06c75">模型太笨，學不來</span>」——模型太簡單或訓練不夠久，連基本模式都學不起來。

* **2. 解決問題的「5 大核心訓練技巧」（簡單知道做什麼即可，不考深奧原理！）**：
  | 五大訓練技巧 | 常見代表名稱 | 一句話白話秒懂（它幫模型解決什麼事？） |
  | :--- | :--- | :--- |
  | **① 選擇合適的啟動函數** | **Sigmoid ➔ ReLU** | 神經元的「**訊號開關**」：早期用 **Sigmoid** 訊號易減弱；改用 <span style="color: #7aa2f7">ReLU</span> 讓訊號順暢通過、<span style="color: #7aa2f7">防梯度消失</span> |
  | **② 挑選適當優化器** | <span style="color: #7aa2f7">SGD、Momentum、RMSProp、Adam</span> | 模型訓練的「<span style="color: #7aa2f7">總指揮 / 聰明領航員</span>」，順著梯度方向<span style="color: #7aa2f7">更新權重參數</span> |
  | **③ 使用批量正規化** | <span style="color: #7aa2f7">BN (Batch Normalization)</span> | 網路層與層之間的「**數值穩定器**」，把跑偏的數字拉回相近範圍，讓訓練又快又穩 |
  | **④ 引入跳接結構** | <span style="color: #7aa2f7">跳接結構 (如 ResNet 殘差網路)</span> | 幫訊號蓋一條「**跨層直達捷徑**」，直接跳過中間層傳遞，<span style="color: #7aa2f7">改善深層模型訓練</span> |
  | **⑤ 正確初始化權重** | <span style="color: #7aa2f7">權重初始化 (Weight Initialization)</span> | 起跑前先把初始參數設得「**剛剛好（不大不小）**」，避免一開始就卡死或暴衝 |

* **3. 換個更聰明的領航員——4 大常見優化器 (Optimizer) 白話對照表**：
  | 優化器名稱 | 圖片原著比喻（秒懂畫面！） | 它是怎麼帶路的？（白話運作方式） | 考試必記特色 |
  | :--- | :--- | :--- | :--- |
  | **SGD (隨機梯度下降)** | 🔦 **拿手電筒摸黑找路** | 每次更新**只看一筆資料**，往誤差變小的方向慢慢前進 | <span style="color: #e06c75">走得慢、易迷路</span>；但具<span style="color: #7aa2f7">隨機性，有機會跳出局部低谷</span> |
  | **Momentum (動量法)** | 🛹 **在滑板上滑行（加慣性）** | 不只看現在的梯度，還<span style="color: #7aa2f7">保留之前的方向，像加了「慣性」</span> | 下坡越滑越快，能衝過小障礙，<span style="color: #7aa2f7">避免卡在小凹陷裡</span> |
  | **RMSProp (均方根傳播)** | ⛰️ **看地形陡緩自動調步伐** | 依梯度變化調整學習率：<span style="color: #7aa2f7">變動大就慢慢來，變動小就加快腳步</span> | 適合應付<span style="color: #7aa2f7">地形不平均</span>的情況，避免單邊訓練太激烈 |
  | **Adam (自適應矩陣估計)** | 🛹🧭 **滑板上還帶著導航** | **結合 <span style="color: #7aa2f7">Momentum ＋ RMSProp</span> 優點**：保留歷史方向＋自動調速 | 萬用型選手，是<span style="color: #7aa2f7">大多數模型的預設值</span>（⚠️ <span style="color: #e06c75">LDA 絕非優化器</span>） |

* **4. 過擬合（太會背）vs 欠擬合（太笨學不來）診斷與解法對照表**：
  | 狀態與白話外號 | 訓練資料表現 | 新資料表現 | 原著白話症狀與成因 | 🎯 對症下藥解法 |
  | :--- | :---: | :---: | :--- | :--- |
  | **過擬合 (Overfitting)**：「<span style="color: #e06c75">太會背，學得太像</span>」 | <span style="color: #7aa2f7">很亮眼</span> | <span style="color: #e06c75">直接完蛋</span> | **把訓練資料當考古題死背**，<span style="color: #e06c75">失去推論能力</span>（生圖只會複製貼上） | 用 <span style="color: #7aa2f7">L1/L2 正則化、Dropout、資料增強、早停法</span>，提升<span style="color: #7aa2f7">概括能力</span> |
  | **欠擬合 (Underfitting)**：「<span style="color: #e06c75">模型太笨，學不來</span>」 | <span style="color: #e06c75">準確率不高</span> | <span style="color: #e06c75">很差</span> | <span style="color: #e06c75">模型太簡單、訓練不夠久</span>，連基本模式都學不起來（生圖模糊破碎） | <span style="color: #7aa2f7">增加模型複雜度</span>（<span style="color: #7aa2f7">加層數、加參數、加非線性單元</span>）＋ <span style="color: #7aa2f7">增加訓練時間</span> |

* **5. 防止「過擬合（太會背）」的 4 個實用工具**：
  1. **L1 / L2 正則化 (Regularization)**：控制模型學習時的參數，**不讓它把單一考題背得太死**。
  2. **Dropout（隨機丟棄）**：訓練時**隨機讓一部分神經元輪流休息**，逼大家都要學會通則、不能只靠少數死背王。
  3. **資料增強 (Data Augmentation)**：把圖片旋轉、翻面、裁切**變出更多花樣**，題目變多元就沒辦法死背。
  4. **早停法 (Early Stopping)**：看到模型開始死背時就**提早喊卡停止訓練**。

* **6. 一字之差超易混淆：正規化 (Normalization) vs 正則化 (Regularization)**（⚠️ **最好中英文一起記！**）：
  | 名稱（中英文） | 它在管什麼？ | 一句話白話區分 |
  | :--- | :--- | :--- |
  | **正規化 (Normalization)** | 處理「<span style="color: #7aa2f7">輸入資料的尺度</span>」 | 讓不同特徵（如身高、體重）落在**相近範圍**（例如縮到 $0 \sim 1$，詳見 1.13 節） |
  | **正則化 (Regularization)** | 處理「<span style="color: #7aa2f7">模型學習時的參數控制</span>」 | 像給模型戴緊箍咒控制參數，專門用來「<span style="color: #7aa2f7">防止過擬合（避免記太多）</span>」 |

<details>
<summary><b>💡 點擊展開白話案例：手電筒與導航滑板 (優化器) ＆ 背考古題 vs 模型太笨 (過擬合/欠擬合)（原著範例）</b></summary>

> 🧭 **1. 為什麼需要優化器？換個更聰明的領航員（原著比喻）**：
> 模型靠「反向傳播」算出誤差後，調整權重的方式就叫「**梯度下降**」。選對**優化器（模型訓練的總指揮）**就像換個更聰明的領航員：
> * **SGD**：像**自己拿著手電筒摸黑找路**，每次只看一筆資料，走得慢、很容易迷路，但因有隨機性有機會跳出局部低谷。
> * **Momentum（動量法）**：像**在滑板上滑行（加了慣性）**，保留之前的方向，下坡越滑越快，穿過小障礙與小凹陷毫不費力。
> * **RMSProp**：像**看地形調步伐**，變動大的地方慢慢來，變動小的地方加快腳步，適合應付地形不平均。
> * **Adam**：像**滑板上還帶著導航**（結合 Momentum 慣性 ＋ RMSProp 自動調速），不但有地圖還會依地形調速，是萬用型選手與大多數模型的預設值！
>
> 📚 **2. 過擬合（太會背，學得太像）vs 欠擬合（模型太笨，學不來）**：
> * **過擬合 (Overfitting)**：模型太會記，記到**把訓練資料當成考古題**，反而失去了推論能力——平時練考古題很亮眼，大考一收到新資料就完蛋，準確率直接跌落神壇！靠 **L1/L2 正則化**與 **Dropout** 阻止它死背。
> * **欠擬合 (Underfitting)**：模型太笨、太簡單或訓練不夠久，連基本模式都學不起來——直接幫它**加層數、加參數、加非線性單元（增加模型複雜度）**並**增加訓練時間**就能解決！
> * *(💡 補充：上述訓練技巧主要針對深度神經網路如 **MLP、CNN、RNN、GRU**，特別是在鑑別式 AI 中；而傳統 AI 如 **SVM 支援向量機、決策樹** 在特定任務依然有效，與深度學習**並存互補**。)*
</details>

* **1.11 考點精華標記**：
  * **🔥 考試常考**：
    1. **預訓練 4 大問題與 5 大技巧**：4 大問題為<span style="color: #e06c75">梯度消失、梯度爆炸、過擬合、欠擬合</span>；5 大技巧為<span style="color: #7aa2f7">選擇合適啟動函數、挑選適當優化器、批量正規化 (Batch Normalization)、引入跳接結構 (如 ResNet)、正確初始化權重</span>。
    2. **四大優化器（模型訓練總指揮）**：**SGD**（<span style="color: #7aa2f7">看單筆資料、手電筒摸黑找路</span>）、**Momentum**（<span style="color: #7aa2f7">加慣性、滑板衝過小凹陷</span>）、**RMSProp**（<span style="color: #7aa2f7">變動大慢慢來、變動小加快腳步</span>，應付不平均地形）、**Adam**（<span style="color: #7aa2f7">結合 Momentum + RMSProp 的導航滑板</span>，大多數模型預設值）。
    3. **過擬合 vs 欠擬合**：**過擬合（太會背）**＝訓練資料表現亮眼、新資料表現很差，用 <span style="color: #7aa2f7">L1/L2 正則化、Dropout、資料增強、早停法</span>解決；**欠擬合（太笨學不來）**＝訓練資料準確率就不高，用<span style="color: #7aa2f7">增加模型複雜度（加層數/加參數/加非線性單元）與訓練時間</span>解決。
  * **⚠️ 易錯陷阱**：
    1. **正規化 vs 正則化陷阱**：處理「輸入資料尺度」是<span style="color: #7aa2f7">正規化 (Normalization)</span>；處理「模型參數控制、防止過擬合」是<span style="color: #7aa2f7">正則化 (Regularization)</span>，兩者<span style="color: #e06c75">切勿混淆</span>！
    2. **優化器陷阱**：問何者「不是 AI 優化器」，必選 <span style="color: #e06c75">LDA (線性判別分析)</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.12 資料清理與 ETL
* **🔗 上下游串聯**：直接對應 1.9 節預訓練的「**Step 1 資料收集與處理**」——餵給 AI 教材前，必須先經過 ETL 清洗、填補缺失值與剔除異常值。
* **核心名詞全稱**：
  * **ETL（提取、轉換、載入, Extract, Transform, Load）**
  * **Data Warehouse（資料倉儲）** vs **Data Lake（資料湖）**
  * **BI（商業智慧, Business Intelligence）**、**IQR（四分位距, Interquartile Range）**
* **資料分析 4 大通用流程**：`① 收集資料 / 資料擷取 (Data Acquisition，第一步)` ➔ `② 數據清理 (Data Cleaning)` ➔ `③ 模型建構 / 分析 (Modeling)` ➔ `④ 視覺呈現與報告撰寫 (Visualization & Reporting，最後一步)`
* **ETL 資料處理 3 步曲**：`Extract（提取原始資料）` ➔ `Transform（清洗/轉換/整合）` ➔ `Load（載入目標倉儲）`（常搭配 **Hadoop / Spark** 大量平行運算）

* **數據分析 AI 應用與儲存架構速查表**：
  | 考點維度 | 核心項目 | 一句話白話秒懂（它到底是什麼？） | 考試易錯排除 |
  | :--- | :--- | :--- | :--- |
  | **數據分析常見 AI 應用** | <span style="color: #7aa2f7">自動化資料清洗</span> (Automated Data Cleaning) | AI 自動幫你抓異常、補漏填、刪重複，省下 70% 整理表格時間 | ❌ <span style="color: #e06c75">非網頁爬蟲設計、硬碟分割、語音壓縮</span> |
  | **GenAI (Generative AI, 生成式 AI) 改善傳統分析** | <span style="color: #7aa2f7">提供預測與模式探索</span> (Prediction & Pattern Discovery) | 不只看過去發生什麼，更能從海量資料挖出隱藏規律、預測未來趨勢 | ❌ <span style="color: #e06c75">非提升網速、減少輸入錯誤、優化排班</span> |
  | **資料倉儲 (Data Warehouse)** | <span style="color: #7aa2f7">高度結構化</span> | 像先畫好固定格子的 Excel 表格，商業智慧 (BI, Business Intelligence) 查報表速度極快 | 適合結構化報表，<span style="color: #e06c75">靈活性較低</span> |
  | **資料湖 (Data Lake)** | <span style="color: #7aa2f7">非結構化 / 原始數據</span> | 像大型倉庫，把文字、影音、紀錄檔原封不動先丟進去，要用時再整理 | <span style="color: #7aa2f7">靈活性極高</span>，適合大模型與多模態訓練 |

* **資料清理 (Data Cleaning) 核心選型表**：
  | 清理任務 | 細項情境 | 最佳處理方法 | 一句話白話判準與禁忌 |
  | :--- | :--- | :--- | :--- |
  | **缺失值填補 (Imputation)** | **連續數值型**（常態無極端值） | <span style="color: #7aa2f7">平均數</span> (Mean) | 只有極少數幾筆漏填時才能直接刪掉；若<span style="color: #e06c75">漏填達 1/3 直接刪會嚴重失真</span> |
  | **缺失值填補** | **數值有極端值 / 偏態分佈** | <span style="color: #7aa2f7">中位數</span> (Median) | 排正中間的數字，<span style="color: #7aa2f7">不怕被極端大或極端小的怪數字拉偏</span>（如郭台銘拉高平均薪資） |
  | **缺失值填補** | **類別型資料 (Categorical)**（如性別、職業） | <span style="color: #7aa2f7">眾數</span> (Mode) | 挑出現次數最多的那一類來補；⚠️ 文字類別<span style="color: #e06c75">絕對算不出平均數或中位數</span>！ |
  | **異常值檢測 (Outliers)** | 四分位距 (IQR, Interquartile Range) / 標準差 | <span style="color: #7aa2f7">盒鬚圖</span> (Box plot) / <span style="color: #7aa2f7">Z 分數</span> (Z-score) | Z 分數偏離平均<span style="color: #7aa2f7">超過 3 個標準差</span> ($|Z| > 3$)，就判定是異常怪咖 |
  | **標註一致性** | 多人標註標準不一 | <span style="color: #7aa2f7">統一標準 ＋ 交叉驗證</span> ＋ 自動比對 | 避免不同人貼的標籤標準不一，害模型<span style="color: #e06c75">越學越混亂</span> |

<details>
<summary><b>💡 點擊展開白話案例：廚房洗切料理 (ETL) 與找空白年齡 (原著範例)</b></summary>

> 🍳 **1. 食材洗切與 ETL 流水線**：
> 資料就像食材，要先採買提取（Extract）、洗切調味轉換（Transform），最後盛盤載入上桌（Load），並靠 Hadoop / Spark 這座大型中央廚房平行加速。
>
> 🗄️ **2. 資料倉儲 vs 資料湖**：
> **資料倉儲**像貼好標籤的整齊調味料架；**資料湖**像大型保鮮庫，生鮮蔬果先原封不動堆進去，要煮時再分類。
>
> 📝 **3. 缺失值補法與異常值怪咖**：
> 顧客年齡漏填用「平均數」；若有人亂填 150 歲極端值改用「中位數」；漏填性別（類別）則用出現最多的「眾數」。若平時日營收 3000 元某天暴衝到 50000 元（$|Z|>3$），就是必須檢查的異常值！
</details>

* **1.12 考點精華標記**：
  * **🔥 考試常考**：
    1. **流程首尾**：第一步是<span style="color: #7aa2f7">收集資料 / 資料擷取</span>；最後一步是<span style="color: #7aa2f7">視覺呈現與報告撰寫</span>。
    2. **數據分析常見 AI 應用**：<span style="color: #7aa2f7">自動化資料清洗</span>；**生成式 AI 改善分析**：<span style="color: #7aa2f7">提供預測與模式探索</span>。
    3. **缺失值填補口訣**：一般數值用<span style="color: #7aa2f7">平均數</span>、有極端值用<span style="color: #7aa2f7">中位數</span>、類別型用<span style="color: #7aa2f7">眾數</span>。
    4. **異常值判準**：<span style="color: #7aa2f7">盒鬚圖</span> 或 $|Z| > 3$（<span style="color: #7aa2f7">超過 3 個標準差</span>）。
  * **⚠️ 易錯陷阱**：<span style="color: #e06c75">缺失達 1/3 時切勿直接刪除</span>；<span style="color: #e06c75">類別型資料切勿用平均數填補</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.13 正規化 vs 標準化
* **🔗 上下游串聯**：承接 1.12 節資料清理——把缺失與異常值洗乾淨後，進模型前最後一步就是將不同單位的數值「縮放到相近尺度（正規化 / 標準化）」，避免大數字霸佔權重。
* **核心名詞全稱**：
  * **Normalization（正規化 / Min-Max Scaling 最小最大值縮放）**
  * **Standardization（標準化 / Z-score Normalization Z 分數標準化）**
  * **Regularization（正則化，詳見 1.11 節）**
* **一字之差秒殺口訣**：`正規化 Normalization（管輸入資料尺度：把數字按比例壓縮到 0~1）` vs `正則化 Regularization（管模型參數控制：加懲罰機制防止死背過擬合）`

* **正規化 (Min-Max) vs 標準化 (Z-score) 核心對照表**：
  | 比較維度 | 正規化 (Normalization / Min-Max) | 標準化 (Standardization / Z-score) |
  | :--- | :--- | :--- |
  | **核心口訣** | 「<span style="color: #7aa2f7">數值壓縮</span>」（按比例縮到 $0 \sim 1$） | 「<span style="color: #7aa2f7">分佈調整</span>」（看偏離平均幾個標準差） |
  | **轉換後範圍** | <span style="color: #7aa2f7">全部壓縮到 $0 \sim 1$ 之間</span>（如把身高、體重都換成 $0 \sim 1$ 分） | <span style="color: #7aa2f7">平均值變 $\mu=0$、標準差變 $\sigma=1$</span>（有正有負，<span style="color: #e06c75">沒有固定上下限</span>） |
  | **計算公式** | $\frac{X - \min}{\max - \min}$ | $\frac{X - \mu}{\sigma}$ |
  | **適用資料分佈** | <span style="color: #7aa2f7">非常態分佈</span>、最大與最小值差距極大時 | <span style="color: #7aa2f7">接近常態分佈</span>（中間多、兩端少） |
  | **常用演算法** | <span style="color: #7aa2f7">深度學習、神經網路</span>（喜歡 $0 \sim 1$ 的穩定輸入） | <span style="color: #7aa2f7">K-NN (K-Nearest Neighbors)、SVM (Support Vector Machine)、邏輯迴歸</span>（需要算座標距離或梯度） |

<details>
<summary><b>💡 點擊展開白話案例：公平起跑線與溫度轉攝氏（原著範例）</b></summary>

> 🏃 **特徵縮放的公平起跑線**：
> 若把身高（180 公分）和體重（70 公斤）直接丟進模型，大數值會霸佔權重。**正規化**像把所有消費金額按比例壓縮到 $0 \sim 1$ 的尺規上；**標準化**則像算學校考試的「Z 分數」，看出該筆資料偏離全班平均幾個標準差。
</details>

* **1.13 考點精華標記**：
  * **🔥 考試常考**：
    1. **正規化**：壓縮至 <span style="color: #7aa2f7">$0 \sim 1$</span>，適合<span style="color: #7aa2f7">非常態分佈</span>與<span style="color: #7aa2f7">神經網路</span>。
    2. **標準化**：轉為<span style="color: #7aa2f7">平均 0、標準差 1</span>，適合<span style="color: #7aa2f7">常態分佈</span>與 <span style="color: #7aa2f7">K-NN / SVM</span>。
  * **⚠️ 易錯陷阱**：考題問「在損失函數加懲罰項防止過擬合」，答案是<span style="color: #7aa2f7">正則化</span> (Regularization)，<span style="color: #e06c75">絕非正規化</span> (Normalization)！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.14 五大學習範式與經典演算法
* **🔗 上下游串聯**：承接 **1.13 資料正規化與標準化**完成特徵縮放後，便依資料「有無標籤」進入**五大學習範式**與經典演算法訓練（其中**自監督學習**即為 **1.9 / 2.1 LLM 預訓練**的核心，**神經網路優化器**則對應 **1.11 梯度下降領航員**）。
* **標籤光譜速記**：`監督式（全有標籤）` ➔ `半監督（少標籤＋多無標籤）` ➔ `自監督（無人工標籤，資料自產標籤）` ➔ `非監督（全無標籤）` ｜ `強化學習（靠環境獎懲回饋）`

* **機器學習五大學習範式速查表（含英文全稱）**：
  | 學習範式（英文全稱） | 標籤狀態 | 一句話白話秒懂（怎麼學？） | 典型任務與代表模型 |
  | :--- | :--- | :--- | :--- |
  | **監督式學習 (Supervised Learning)** | <span style="color: #7aa2f7">有人工標籤</span> | 像做有附標準答案的題庫，給 AI 看「題目＋正確答案」配對練習 | 分類 (Classification)、迴歸 (Regression)（<span style="color: #7aa2f7">鑑別式 AI 最常用</span>） |
  | **非監督式學習 (Unsupervised Learning)** | <span style="color: #7aa2f7">無標籤</span> | 不給標準答案，讓 AI 自己觀察資料長相，把相似的聚在一起 | <span style="color: #7aa2f7">分群 (Clustering: K-Means)</span>、<span style="color: #7aa2f7">降維 (PCA, Principal Component Analysis / t-SNE, t-Distributed Stochastic Neighbor Embedding)</span>、**VAE (Variational Autoencoder)** |
  | **自監督學習 (Self-Supervised Learning)** | <span style="color: #7aa2f7">無人工標籤（資料自產）</span> | 不用人工標註，把文章遮住幾個字或遮最後一字，讓資料「自己生出考題與答案」 | <span style="color: #7aa2f7">LLM (Large Language Model) 預訓練核心</span>（**GPT 玩文字接龍、BERT 玩克漏字**） |
  | **半監督式學習 (Semi-Supervised Learning)** | <span style="color: #7aa2f7">少量標籤＋大量無標籤</span> | 只請專家標註少數幾張圖，搭配海量沒標籤的圖一起學，省下高額人工費 | 醫療影像辨識、語音辨識 |
  | **強化學習 (RL, Reinforcement Learning)** | <span style="color: #7aa2f7">無標籤（靠獎懲回饋）</span> | 像打電動或訓練寵物，做對給獎勵加分、做錯扣分，摸索出最高分策略 | 下圍棋 (AlphaGo)、電玩 AI、機器人控制 |

* **經典演算法與優化器 (Optimizer) 配對速查表（含英文全稱）**：
  | 類別歸屬 | 演算法 / 元件名稱（英文全稱） | 一句話白話秒懂與必考關鍵字 |
  | :--- | :--- | :--- |
  | **監督式（分類）** | **邏輯迴歸 (Logistic Regression)** | 名字雖叫迴歸，其實是做「<span style="color: #7aa2f7">是非題二元分類</span>」（如判斷是不是垃圾信，輸出 $0 \sim 1$ 機率） |
  | **監督式（分類）** | **支援向量機 (SVM, Support Vector Machine)** | 在兩群資料中間畫出一條「<span style="color: #7aa2f7">最寬、最清楚的分界線</span>」（最大化分類邊界超平面） |
  | **監督式（分類/迴歸）** | **決策樹 (Decision Tree) / 隨機森林 (Random Forest)** | 像玩心理測驗走樹狀分支做判斷；隨機森林則是找<span style="color: #7aa2f7">好多棵樹一起投票</span>，防單一棵樹死背 |
  | **監督式（分類/迴歸）** | **K-最近鄰 (KNN, K-Nearest Neighbors)** | 看離你最近的 $K$ 個鄰居大多數是什麼身分，你就跟他們同類（<span style="color: #7aa2f7">鄰居有標籤！</span>） |
  | **監督式（特徵降維）** | **線性判別分析 (LDA, Linear Discriminant Analysis)** | 把資料壓縮降維，讓「不同類分得越開、同一類聚得越近」（⚠️ <span style="color: #e06c75">絕非神經網路優化器！</span>） |
  | **非監督式（分群）** | **K-均值分群 (K-Means Clustering)** | 沒給標籤，自動把距離相近的資料分成 $K$ 小隊（如把客群自動分 3 類，<span style="color: #7aa2f7">無標籤！</span>） |
  | **非監督式（降維）** | **主成分分析 (PCA, Principal Component Analysis)** | 把好幾百個複雜特徵<span style="color: #7aa2f7">濃縮成幾個最重要的核心指標</span>（線性降維），保留最多原始資訊 |
  | **非監督式（視覺化）** | **t-SNE (t-Distributed Stochastic Neighbor Embedding)** | 把高維度複雜資料壓平畫在 <span style="color: #7aa2f7">2D / 3D 平面圖上</span>，方便人類肉眼觀察分群 |
  | **神經網路優化器** | **SGD (隨機梯度下降, Stochastic Gradient Descent)**、**Momentum (動量法)**、**RMSProp (Root Mean Square Propagation)**、**Adam (Adaptive Moment Estimation)** | 模型訓練的「<span style="color: #7aa2f7">總指揮 / 領航員</span>」，負責順著梯度方向更新權重（<span style="color: #7aa2f7">Adam 結合 Momentum+RMSProp 為萬用預設值</span>，詳見 1.11 節） |

<details>
<summary><b>💡 點擊展開白話案例：垃圾信過濾、下棋試錯與接話高手（原著範例精煉）</b></summary>

> 📧 **監督式（垃圾信）**：給 AI 1000 封標好「正常/垃圾」的信，讓它學會看到「中獎」就分類為垃圾信。
> 🤖 **強化學習（學走路/下棋）**：像小孩打電動，走對加分、走錯扣分，靠獎懲摸索出最高分策略。
> 💬 **自監督（GPT vs BERT）**：不用人工標註，把文章最後一字遮住練接龍（GPT），或挖空中間字練克漏字（BERT），自己生出答案來練功！
</details>

* **1.14 考點精華標記**：
  * **🔥 考試常考**：
    1. **常見 AI 優化器**：<span style="color: #7aa2f7">SGD、Momentum、RMSProp、Adam</span>（詳見 1.11 節）。
    2. **自監督學習代表**：**GPT** 與 **BERT** 的海量預訓練；**非監督降維視覺化**：<span style="color: #7aa2f7">PCA、t-SNE</span>。
  * **⚠️ 易錯陷阱**：
    1. **優化器陷阱**：問何者「不是 AI 優化器」，必選 <span style="color: #e06c75">LDA</span>（LDA 是降維/分類演算法）！
    2. **雙 K 與迴歸陷阱**：<span style="color: #7aa2f7">KNN 是監督式分類</span>，<span style="color: #7aa2f7">K-Means 是非監督式分群</span>；<span style="color: #e06c75">邏輯迴歸是分類器，非數值迴歸</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.15 模型評估指標 F1-score R-squared 與 BLEU
* **🔗 上下游串聯**：承接 **1.14 監督式與自監督學習**訓練出的各類模型，最後一步必須針對「**分類、迴歸、文本生成**」三大任務選對專屬考官指標進行成效驗收（考題最愛跨任務亂配對誘答！）。
* **核心名詞全稱**：
  * **F1-score（F1 分數）**：**Precision（精確率 / 查準率）** 與 **Recall（召回率 / 查全率）** 之 **Harmonic Mean（調和平均數）**
  * **R-square ($R^2$, Coefficient of Determination 決定係數)**、**Accuracy（準確率）**、**Class Imbalance（資料不平衡）**
  * **BLEU（雙語評估替補指標, Bilingual Evaluation Understudy）**、**ROC-AUC（接收者操作特徵曲線下面積, Area Under the Receiver Operating Characteristic Curve）**
* **三大任務指標總配對流**：`① 分類任務（資料不平衡） ➔ F1-score（調和平均）/ ROC-AUC` ｜ `② 迴歸任務（連續數值預測） ➔ R-square (R² 決定係數)` ｜ `③ 文本生成（聊天機器人/機器翻譯） ➔ BLEU`

* **一、 三大任務評估指標（F1-score vs R-square vs BLEU）核心對照表**：
  | 評估指標 | 所屬任務類型 | 核心公式 / 運算機制（白話秒懂） | 關鍵優勢與必考排除陷阱 |
  | :--- | :--- | :--- | :--- |
  | **F1-score** | **<span style="color: #7aa2f7">分類任務</span>** (Classification) | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$（精確率與召回率的<span style="color: #7aa2f7">調和平均數</span>） | 只要精確率或召回率一邊考太爛，總分就極低；為「<span style="color: #7aa2f7">資料不平衡 (Class Imbalance) 時的最佳分類指標</span>」（❌ <span style="color: #e06c75">非算術平均或幾何平均</span>） |
  | **R-square ($R^2$)** | **<span style="color: #7aa2f7">迴歸任務</span>** (Regression) | 看預測出來的連續數字跟真實答案有多接近（決定係數） | 衡量「<span style="color: #7aa2f7">連續數值預測（如猜房價/猜銷量）準不準</span>」；介於 $0 \sim 1$，越接近 1 越準（❌ <span style="color: #e06c75">不可用於是非分類題</span>） |
  | **BLEU** | **<span style="color: #7aa2f7">文本生成任務</span>** (Text Generation) | ① <span style="color: #7aa2f7">N-gram 詞彙重疊度比對</span>（看 AI 寫的句子跟標準答案有幾個詞一樣）＋ ② <span style="color: #7aa2f7">簡短懲罰 (Brevity Penalty)</span>（防 AI 只寫一兩字偷懶） | 評估「<span style="color: #7aa2f7">聊天機器人 (Chatbot) 回答品質</span>」與「<span style="color: #7aa2f7">機器翻譯 (Machine Translation) 準確度</span>」核心演算法指標（❌ 排除：<span style="color: #e06c75">ROC-AUC、對話次數、用戶滿意度</span>） |

* **二、 精確率 (Precision) vs 召回率 (Recall) 1 行秒懂**：
  * **精確率 (Precision)**：AI 抓出來的壞人裡面，真的有幾個是壞人？（寧缺毋濫，<span style="color: #7aa2f7">防誤抓好人</span>）。
  * **召回率 (Recall)**：所有真正的壞人裡面，AI 成功抓到了幾個？（天羅地網，<span style="color: #7aa2f7">防漏掉壞人</span>）。

<details>
<summary><b>💡 點擊展開白話案例：1% 詐騙郵件的準確率騙局 vs 改英文翻譯作文的 BLEU 老師</b></summary>

> 🚨 **1. 1% 詐騙郵件的準確率騙局 (F1-score)**：
> 1,000 封信裡只有 10 封詐騙信 (1%)。笨模型把 1,000 封全猜「正常」，準確率 (Accuracy) 高達 99%，卻漏掉全部詐騙信（召回率 0%）！改用 **F1-score（調和平均）**，只要召回率是 0，F1 就直接歸零，完美戳破假象。
>
> 📝 **2. 改英文翻譯作文的關鍵字比對法 (BLEU)**：
> **BLEU** 像英文老師拿標準答案逐行比對你的翻譯作文，數有幾個單字、片語吻合（N-gram 匹配）；若你投機只寫一個字交卷，老師就啟動「簡短懲罰」扣分。
</details>

* **1.15 考點精華標記**：
  * **🔥 考試常考**：
    1. **資料不平衡最佳分類指標**：<span style="color: #7aa2f7">F1-score</span>（精確率與召回率的<span style="color: #7aa2f7">調和平均數</span>）。
    2. **迴歸數值預測首選指標**：<span style="color: #7aa2f7">R-square ($R^2$)</span>。
    3. **聊天機器人 / 機器翻譯生成評估指標**：<span style="color: #7aa2f7">BLEU</span>。
  * **⚠️ 易錯陷阱**：
    1. F1-score 是<span style="color: #7aa2f7">調和平均數</span>（<span style="color: #e06c75">非算術/幾何平均</span>）；評估房價/銷量預測選 <span style="color: #7aa2f7">$R^2$</span>，<span style="color: #e06c75">不可選 F1-score</span>！
    2. 評估聊天機器人或翻譯效能必選 <span style="color: #7aa2f7">BLEU</span>，切勿選業務指標（<span style="color: #e06c75">對話次數、用戶滿意度</span>）或分類指標（<span style="color: #e06c75">ROC-AUC</span>）！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 1.16 AIGC 核心概念
* **核心名詞全稱**：
  * **PGC（專業生產內容, Professional Generated Content）**
  * **UGC（用戶生產內容, User Generated Content）**
  * **AIGC（人工智慧生成內容, Artificial Intelligence Generated Content）**
  * **OCR（光學字元辨識, Optical Character Recognition）**
* **內容生產三代演進**：`PGC（專業生產 / Web 1.0）` ➔ `UGC（用戶生產 / Web 2.0）` ➔ `AIGC（AI 演算法生產 / Web 3.0）`

* **PGC vs UGC vs AIGC 演進對照表**：
  | 生產模式 | 英文全稱 | 生產主體 | 代表平台與核心特徵 |
  | :--- | :--- | :--- | :--- |
  | **PGC** | Professional Generated Content | <span style="color: #7aa2f7">專業團隊 / 專家</span> | 傳統媒體、新聞門戶；品質高但<span style="color: #e06c75">產能低、成本高</span> |
  | **UGC** | User Generated Content | <span style="color: #7aa2f7">一般網友大眾</span> | YouTube、IG、FB；產量大互動高，但<span style="color: #e06c75">品質參差</span> |
  | **AIGC** | **Artificial Intelligence Generated Content** | <span style="color: #7aa2f7">生成式 AI 演算法</span> | ChatGPT、Midjourney、Sora；<span style="color: #7aa2f7">高效率、跨模態海量原創</span> |

* **四大易混淆概念辨析表**：
  | 概念名稱 | 本質定位 | 輸出結果 | 是否為 AIGC 生成應用？ |
  | :--- | :--- | :--- | :--- |
  | **AIGC** | 利用深度學習生成多模態內容之應用總稱 | 全新文本、影像、音訊、影片 | ✅ <span style="color: #7aa2f7">是（核心本體）</span> |
  | **CNN (卷積神經網路, Convolutional Neural Network)** | 鑑別式影像特徵萃取與分類模型 | 類別標籤或機率 | ❌ <span style="color: #e06c75">否（屬判別式模型）</span> |
  | **OCR (光學字元辨識, Optical Character Recognition)** | 將圖片文字轉為數位文字之識別技術 | 辨識出之既有文字 | ❌ <span style="color: #e06c75">否（屬鑑別識別技術）</span> |
  | **NLP (自然語言處理, Natural Language Processing)** | 涵蓋語言理解 (NLU) 與生成 (NLG) 的學科領域 | 理解分析或文本生成 | ⚠️ 為學科領域，<span style="color: #e06c75">非單指生成應用</span> |

<details>
<summary><b>💡 點擊展開白話案例：從報社印刷廠到 AI 夢工廠（內容生產演進比喻）</b></summary>

> 🏭 **PGC ➔ UGC ➔ AIGC**：
> **PGC** 是專業報社寫書；**UGC** 是全民拿手機發 IG 限動；**AIGC** 是給 AI 一句話，幾秒內自動寫小說、畫名畫、拍短片！而 CNN / OCR 就像「驗鈔機」，只負責辨識真假，不會無中生有印出新鈔票。
</details>

* **1.16 考點精華標記**：
  * **🔥 考試常考**：
    1. **AIGC 定義**：利用<span style="color: #7aa2f7">深度學習技術生成各種形式內容和資料</span>的應用（支援模型：<span style="color: #7aa2f7">GAN、VAE、Diffusion、Transformer</span>）。
  * **⚠️ 易錯陷阱**：<span style="color: #e06c75">CNN</span> 與 <span style="color: #e06c75">OCR</span> 皆屬於「鑑別/識別技術」，<span style="color: #e06c75">絕非 AIGC 生成式模型或應用</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

## 第二部分 生成式AI能力強化

### 2.1 大語言模型 (LLM) 原理 微調與應用模式
* **🔗 上下游串聯**：將第一部分 **1.5 Transformer 架構**、**1.9 自監督預訓練（產出通才 Base Model）** 與 **1.10 損失函數＋反向傳播** 整合落地，透過本節的「**微調 (Fine-tuning) ＋ RLHF 行為對齊**」把通才調教成 ChatGPT 等專業助手，並衍生出下一節 **2.2 提示工程 (Prompt Engineering)** 的免重訓引導技術。
* **核心名詞全稱**：
  * **LLM（大語言模型, Large Language Model）**、**Base Model（基礎模型 / 預訓練通才模型）**
  * **Pre-training（預訓練）**、**Self-Supervised Learning（自監督學習）**、**Fine-tuning（微調）**
  * **Alignment（行為對齊）**、**RLHF（人類回饋強化學習, Reinforcement Learning from Human Feedback）**
  * **Reduction（縮減）**、**Transform（轉換）**、**Expansion（擴展）**
* **LLM 養成與強化一行流**：`底層結構（類神經網路 ＋ Transformer）` ➔ `① 預訓練 Pre-training（自監督讀千萬篇文章，靠梯度下降＋反向傳播 ➔ 產出「通才」基礎模型 Base Model）` ➔ `② 微調 Fine-tuning（角色扮演 ＋ 任務訓練 ＋ RLHF 行為對齊 ➔ 蛻變為「專才」如 ChatGPT）`

* **一、 大語言模型原理：預訓練（通才）vs 微調（專才）速查表**：
  | 階段 / 維度 | 核心機制與訓練方式 | 產出成果與核心定位（白話秒懂） |
  | :--- | :--- | :--- |
  | **底層結構** | **<span style="color: #7aa2f7">類神經網路</span> ＋ <span style="color: #7aa2f7">Transformer 架構</span>** | 透過自注意力機制平行處理上下文語意關聯 |
  | **1. 預訓練 (Pre-training)** | 屬「<span style="color: #7aa2f7">自監督學習</span> (Self-Supervised Learning)」，讀上千萬篇文章自學；先用<span style="color: #7aa2f7">梯度下降</span>找最佳解、再靠<span style="color: #7aa2f7">反向傳播</span>修正權重 | 產出「<span style="color: #7aa2f7">基礎模型 (Base Model)</span>」——博學但只會接話的「**通才**」（<span style="color: #e06c75">尚不懂扮演專業角色或遵守安全規範</span>） |
  | **2. 微調 (Fine-tuning)** | 在基礎模型上進行專業特訓，涵蓋「<span style="color: #7aa2f7">角色扮演、任務訓練、行為對齊</span>」三大目標 | 將通才調教成客服助理、數學老師或法律顧問等「<span style="color: #7aa2f7">專才</span>」（如 **ChatGPT**） |

* **二、 微調 (Fine-tuning) 的 3 大核心目標**：
  * **(a) <span style="color: #7aa2f7">角色扮演（角色訓練）</span>**：讓模型學會扮演特定身分與語氣（如客服助理、醫療諮詢專員、法律助理）。
  * **(b) <span style="color: #7aa2f7">任務訓練</span>**：強化特定工作技能，讓模型在特定任務更熟練（如寫報告、寫程式、數學推理）。
  * **(c) <span style="color: #7aa2f7">行為對齊 (Alignment)</span>**：讓模型回答貼近人類價值與安全期待，避免胡言亂語、<span style="color: #e06c75">歧視或洩密</span>（例如 **ChatGPT** 先用 GPT-3.5 加上 **<span style="color: #7aa2f7">人類回饋強化學習 (RLHF)</span>**，讓回答更有邏輯、更禮貌且不會脫序失控）。

* **三、 LLM 三大應用模式（縮減、轉換、擴展）速查表**：
  | 應用模式 | 資訊量變化（輸入 ➔ 輸出） | 一句話白話秒懂與實務案例 |
  | :--- | :--- | :--- |
  | **1. <span style="color: #7aa2f7">縮減 (Reduction)</span>** | **多 ➔ 少**（長文變短） | **去蕪存菁抓重點**：長篇會議紀錄摘要、萬字財報重點提煉、文章關鍵字萃取 |
  | **2. <span style="color: #7aa2f7">轉換 (Transform)</span>** | **A 形式 ➔ B 形式**（換包裝） | **換語言或換格式**：多國語言翻譯、口語改寫成正式公文、文法糾錯、自然語言轉程式碼 |
  | **3. <span style="color: #7aa2f7">擴展 (Expansion)</span>** | **少 ➔ 多**（短指令變長文） | **無中生有發散創作**：輸入一句主題自動擴寫成完整企劃書、行銷文案、客服回信或故事腳本 |

* **四、 主流 LLM 產品與科技巨頭生態系速查表（統計至 2025）**：
  | 公司 | 主打產品 / 模型 | 核心策略特色與必考題眼 |
  | :--- | :--- | :--- |
  | **OpenAI** | **ChatGPT**、**GPTs**、**DALL·E 3**、**Sora** | 即時互動、全能多模態（文/圖/音/影）、<span style="color: #7aa2f7">平臺化（GPTs 客製生態系）</span> |
  | **Google** | **Gemini 1.5 系列 (Pro / Flash)** | 搜尋整合、手機端導入（Pixel + Android）、<span style="color: #7aa2f7">多模態加強（影片、文件解析）</span> |
  | **Anthropic** | **Claude 3 系列 (Opus / Sonnet / Haiku)** | <span style="color: #7aa2f7">符合人類價值觀、安全性高</span>、支援超長上下文（達 <span style="color: #7aa2f7">200K tokens</span>） |
  | **Meta** | **Meta AI**、**LLaMA 3 (Large Language Model Meta AI)** | 主打「<span style="color: #7aa2f7">開源推廣</span>」標竿、社群導向整合（Meta App 內建 AI 助手） |
  | **xAI (Elon Musk)** | **Grok** | 結合 **X（原 Twitter）社群**、強調<span style="color: #7aa2f7">言論自由與即時資訊</span> |
  | **Amazon (AWS)** | **Bedrock**、**CodeWhisperer** | 雲端整合、<span style="color: #7aa2f7">B2B 工具型 AI、企業級部署</span>（與 AWS 深度結合） |
  | **Apple** | **Apple Intelligence** | Foundation Models 框架，強調「<span style="color: #7aa2f7">離線運算、隱私保護</span>」與多語言支援 |
  | **NVIDIA** | **H100 / B100**、**NIMs (NVIDIA Inference Microservices) 推論平臺** | 提供 LLM 訓練與部署核心基礎，為各家 AI 模型的「<span style="color: #7aa2f7">GPU 心臟</span>」 |

<details>
<summary><b>💡 點擊展開白話案例：大學通識畢業生 (Base Model) vs 企業入職特訓 (Fine-tuning)（原著概念精煉）</b></summary>

> 🎓 **預訓練 vs 微調**：
> **預訓練 (Pre-training)** 就像讓學生自己泡在圖書館讀完上千萬本書（自監督學習），畢業時成了博學多聞的「**通才（基礎模型 Base Model）**」；**微調 (Fine-tuning)** 則是進公司後的「職前特訓」，教他穿上制服說話（**角色扮演**）、熟練寫報告寫程式（**任務訓練**），並透過人類主管打分數調教（**RLHF 行為對齊**），讓他講話有禮貌、守規矩不失控，正式成為像 **ChatGPT** 一樣的專業助手！
</details>

* **2.1 考點精華標記**：
  * **🔥 考試常考**：
    1. **預訓練 vs 微調**：預訓練屬<span style="color: #7aa2f7">自監督學習</span>，產出通才型的「<span style="color: #7aa2f7">基礎模型 (Base Model)</span>」；微調包含<span style="color: #7aa2f7">角色扮演、任務訓練、行為對齊 (Alignment)</span> 三大目標。
    2. **ChatGPT 行為對齊關鍵**：以 GPT-3.5 結合 **<span style="color: #7aa2f7">人類回饋強化學習 (RLHF)</span>**，確保回答合乎人類價值觀與安全。
    3. **LLM 三大應用模式**：<span style="color: #7aa2f7">縮減 (Reduction，摘要提煉)、轉換 (Transform，翻譯改寫)、擴展 (Expansion，文案擴寫)</span>。
  * **⚠️ 易錯陷阱**：剛完成預訓練的 Base Model 只是<span style="color: #7aa2f7">通才</span>，<span style="color: #e06c75">尚未具備專業角色扮演與安全對齊能力</span>，必須經過「<span style="color: #7aa2f7">微調 (Fine-tuning)</span>」與「<span style="color: #7aa2f7">RLHF</span>」才能成為專業可靠的對話助手！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 2.2 提示工程 In-Context Learning 與思維樹 (ToT)
* **🔗 上下游串聯**：承接 **2.1 微調完成的 LLM**，若不想花大錢重新訓練改權重，只需在推論階段運用本節的「**提示工程 (Prompt Engineering)**」與「**上下文學習 (In-Context Learning)**」即時引導模型；面對高難度邏輯問題時，更可從單步直覺 (Zero-shot) 升級為「**思維鏈 (CoT) ➔ 思維樹 (ToT)**」多路徑探索。
* **核心名詞全稱**：
  * **Prompt Engineering（提示工程）**、**ICL（上下文學習, In-Context Learning）**
  * **Zero-shot Prompting（零樣本提示）**、**Few-shot Prompting（少樣本提示）**
  * **CoT（思維鏈, Chain-of-Thought）**、**ToT（思維樹, Tree-of-Thought）**、**Backtracking（回溯搜尋）**
* **Prompt 引導與推理升級一行流**：`基礎設計（明確任務需求＋熟悉詞彙，忌冗長與抽象）` ➔ `上下文學習 ICL（給 Few-shot 範例，免改權重即時調整回答）` ➔ `推理升級：Zero-shot（一步給答案） ➔ 思維鏈 CoT（單線逐步推理） ➔ 思維樹 ToT（樹狀多路徑探索＋回溯搜尋）`

* **一、 提示工程 (Prompt Engineering) 正反向原則與上下文學習 (ICL) 速查表**：
  | 考點類別 | ✅ 正確做法 / 核心優勢（必背正解） | ❌ 錯誤迷思 / 非協助技巧（必選陷阱） |
  | :--- | :--- | :--- |
  | **Prompt 設計必要考量** | ① <span style="color: #7aa2f7">適當表達任務需求</span>、② <span style="color: #7aa2f7">使用模型熟悉的單詞</span>、③ <span style="color: #7aa2f7">嘗試不同的 Prompt 方式</span> | ❌ <span style="color: #e06c75">確保 Prompt 長度足夠長</span>（過長會浪費 Token、稀釋注意力並引發幻覺） |
  | **Prompt 優化協助技巧** | ① <span style="color: #7aa2f7">測試不同的 Prompt 組合和格式</span>、② <span style="color: #7aa2f7">累積正確的回答和訂正錯誤</span>、③ <span style="color: #7aa2f7">監控和調整提問</span> | ❌ <span style="color: #e06c75">開放式的抽象描述</span>（缺乏具體目標與格式約束，導致輸出發散離題） |
  | **上下文學習 (ICL, In-Context Learning)** | <span style="color: #7aa2f7">無需重新訓練或改模型權重</span>，僅在 Prompt 提供指令或<span style="color: #7aa2f7">少數範例 (Few-shot)</span>，最大優勢為「**<span style="color: #7aa2f7">能根據上下文即時調整回答</span>**」 | ❌ <span style="color: #e06c75">並非不需要訓練資料、並非不需要語言模型、並非只適用圖像</span> |

* **二、 進階推理框架：思維樹 (ToT, Tree-of-Thought) 四大核心特點速查表**：
  | 核心特點 | 一句話白話秒懂（它怎麼思考？） |
  | :--- | :--- |
  | **1. 分解複雜問題** | 把大難題拆成好幾個<span style="color: #7aa2f7">中間小步驟</span>（思維節點），一步一步慢慢想（❌ <span style="color: #e06c75">絕非僅依賴單一步驟解答！</span>） |
  | **2. 促進逐步思考** | 每走一步就同時想出<span style="color: #7aa2f7">好幾個候選方案</span>來比較，不急著一步下結論 |
  | **3. 多路徑探索與回溯** | 像走迷宮展開<span style="color: #7aa2f7">樹狀多條路線</span>，發現走進死胡同還能自動「<span style="color: #7aa2f7">退回上一步</span> (回溯 Backtracking)」換路走 |
  | **4. 提高解決問題準確性** | 先比較各條路線好壞再做決定（<span style="color: #7aa2f7">全局評估搜索</span>），大幅提高複雜邏輯與規劃準確率 |

<details>
<summary><b>💡 點擊展開白話案例：點餐講重點、看便利貼即時調頭的秘書 (ICL) 與西洋棋大師 (ToT)</b></summary>

> 🗣️ **1. 點餐講重點（忌冗長與抽象）**：
> 點餐說「大麥克不要酸黃瓜」店員秒懂；若廢話講 10 分鐘（追求長度足夠長）或說「給我一份有靈魂的餐點」（開放式抽象描述），店員只會一頭霧水！
>
> 🔄 **2. In-Context Learning（看便利貼即時調頭的秘書）**：
> 不用把秘書送去進修三個月（重新訓練微調），只要開會時遞給他一張便利貼（Prompt 上下文範例），他就能**當場即時調整回答語氣與格式**！
>
> ♟️ **3. ToT 西洋棋大師比喻**：
> **單步解答 (Zero-shot)** 像新手看一眼就亂下一子；**CoT (思維鏈)** 像只算一條路線走到黑；**ToT (思維樹)** 則像西洋棋大師，腦中同時展開 A、B、C 三條分支路線，發現走 A 第三步會死棋，立刻「**回溯**」改走 B 路線！
</details>

* **2.2 考點精華標記**：
  * **🔥 考試常考**：
    1. **In-Context Learning 最大優勢**：<span style="color: #7aa2f7">能根據上下文即時調整回答</span>。
    2. **Prompt 非必要考量**：<span style="color: #e06c75">確保 Prompt 長度足夠長</span>；**非協助技巧**：<span style="color: #e06c75">開放式的抽象描述</span>。
    3. **思維樹 (ToT) 特點**：<span style="color: #7aa2f7">分解複雜問題、促進逐步思考、提高解決問題的準確性</span>。
  * **⚠️ 易錯陷阱**：
    1. Prompt 講求<span style="color: #7aa2f7">精簡、具體與結構化</span>，切勿被「<span style="color: #e06c75">長度足夠長</span>」或「<span style="color: #e06c75">開放式抽象描述</span>」誤導！
    2. 問何者「**不是**」ToT 特點，必選「<span style="color: #e06c75">僅依賴單一步驟解答</span>」！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 2.3 程式邏輯理解 語法分析器
* **核心名詞全稱**：
  * **Parser（語法分析器）**、**AST（抽象語法樹, Abstract Syntax Tree）**、**GNN（圖神經網路, Graph Neural Network）**
* **程式碼解析強化流**：`原始程式碼文本` ➔ `語法分析器 (Parser)` ➔ `抽象語法樹 (AST, Abstract Syntax Tree)` ➔ `強化 AI 理解程式碼上下文邏輯`

* **程式理解技術對照表**：
  | 技術名稱 | 一句話白話秒懂（它到底在做什麼？） | 考試選型判斷 |
  | :--- | :--- | :--- |
  | **語法分析器 (Parser)** | 像文法老師把程式碼畫成「<span style="color: #7aa2f7">樹狀家譜圖 (抽象語法樹 AST, Abstract Syntax Tree)</span>」，幫 AI 看懂誰呼叫誰、前後邏輯與管轄範圍 | ✅ 「<span style="color: #7aa2f7">強化 AI 理解程式上下文邏輯</span>」唯一正解 |
  | **圖神經網路 (GNN, Graph Neural Network) / 語意轉換器 / 語音模型** | 處理人際網路圖 / 一般白話文轉換 / 語音訊號處理 | ❌ <span style="color: #e06c75">非拆解程式碼語法結構</span>之核心技術 |

<details>
<summary><b>💡 點擊展開白話案例：文法分析老師與樹狀架構圖</b></summary>

> 🌲 **語法分析器 (Parser) 的家譜樹**：
> 語法分析器就像嚴謹的文法老師，把一萬行程式碼畫成一張「樹狀家譜圖 (AST)」，標明誰是父函數、誰是子變數，讓 AI 瞬間看懂深層程式邏輯！
</details>

* **2.3 考點精華標記**：
  * **🔥 考試常考**：可強化 AI 理解程式上下文邏輯的技術為「<span style="color: #7aa2f7">語法分析器</span> (Parser)」。
  * **⚠️ 易錯陷阱**：切勿誤選<span style="color: #e06c75">圖神經網路、語意轉換器或語音模型</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

## 第三部分 生成式AI應用技能

### 3.1 影片生成工具與模型
* **核心名詞全稱**：
  * **Text-to-Video（文字生成影片）**、**Image-to-Video（圖片生成影片）**
  * **Space-Time U-Net（時空 U-Net 架構，Google Lumiere 核心技術）**
* **主流 AI 影片生成陣營**：`Runway (Gen-1 ~ Gen-4)` ＋ `Sora (OpenAI)` ＋ `Pika (Pika Labs)` ＋ `Google Lumiere / Veo` ＋ `FlexClip (PearlMountain)`（⚠️ **DALL·E 是文字生成圖像 Text-to-Image，絕非影片工具！**）

* **影片生成工具速查表**：
  | 開發商 | 代表工具 / 模型 | 技術基礎 | 核心受眾、應用與關鍵考點 |
  | :--- | :--- | :--- | :--- |
  | **Runway ML** | **Runway (Gen-1 ~ Gen-4)** | <span style="color: #7aa2f7">改良版 Diffusion</span> | 創作者影片草圖、剪輯特效；操作簡易生成快，專門協助「<span style="color: #7aa2f7">非專業創作者</span>」 |
  | **OpenAI** | **Sora** | <span style="color: #7aa2f7">Diffusion + Transformer</span> | 文生成影片（語意驅動敘事）；<span style="color: #7aa2f7">畫質佳、敘事強、世界理解力高</span>（60 秒高連貫） |
  | **Pika Labs** | **Pika** | <span style="color: #7aa2f7">動畫導向 Diffusion + 多模態控制</span> | 原創角色動畫、社群短片；<span style="color: #7aa2f7">風格自由、互動性高、創作流程友善</span> |
  | **Google / DeepMind** | **Google Lumiere / Veo** | <span style="color: #7aa2f7">Space-Time U-Net</span> (Lumiere) | 高畫質敘事影片；<span style="color: #7aa2f7">畫面穩定、邏輯一致、細節表現極佳</span> |
  | **PearlMountain** | **FlexClip** | 雲端 SaaS 平台 | 線上輕量化 <span style="color: #7aa2f7">AI 影片生成與剪輯</span>平台（支援文字轉影片、自動字幕） |

* **3.1 考點精華標記**：
  * **🔥 考試常考**：
    1. **Runway ML 核心受眾**：協助「<span style="color: #7aa2f7">非專業創作者</span>」（<span style="color: #e06c75">非專業程式設計師、非硬體工程師</span>）；代表模型為 <span style="color: #7aa2f7">Gen-2 / Gen-4</span>。
  * **⚠️ 易錯陷阱**：考題問何者「**不是**」影片生成工具，必選 <span style="color: #e06c75">DALL·E</span>（DALL·E 為<span style="color: #e06c75">靜態圖像生成工具</span>）！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 3.2 資料視覺化設計原則
* **核心名詞全稱**：
  * **Data Visualization（資料視覺化）**、**Style Consistency（風格一致性）**、**Information Overload（資訊過載 / 資訊過多）**
* **視覺化黃金準則**：`統一顏色與標籤格式（保持風格一致）` ＋ `簡單配色突出重點` ＋ `嚴禁資訊過多（避免資訊過載）`

* **資料視覺化正反向考點對照表**：
  | 審題方向 | 考題提問方式 | 必選標準答案 | 干擾項排除 |
  | :--- | :--- | :--- | :--- |
  | **正面原則** | 保持視覺化**風格一致性**應使用何種元素？ | <span style="color: #7aa2f7">統一的顏色和標籤格式</span> | ❌ <span style="color: #e06c75">多種顏色、不同字體、隨機資料源</span> |
  | **負面防範** | 進行資料視覺化時**應注意避免的問題**？ | <span style="color: #e06c75">資訊過多（資訊過載）</span> | ✅ 使用簡單顏色、突出關鍵資訊、數據穩定（此三項為正面做法） |

<details>
<summary><b>💡 點擊展開白話案例：品牌制服的一致性美感 vs 萬聖節百家布拼貼（補充範例）</b></summary>

> 👔 **品牌制服 vs 百家布**：
> 整套報表用「統一的顏色和標籤格式」就像穿整齊制服，一眼看懂重點；若每頁五顏六色、字體亂跳，或塞滿密密麻麻的字（資訊過多），就會像萬聖節百家布一樣讓人眼花撩亂。
</details>

* **3.2 考點精華標記**：
  * **🔥 考試常考**：保持風格一致性選「<span style="color: #7aa2f7">統一的顏色和標籤格式</span>」；應避免的問題選「<span style="color: #e06c75">資訊過多</span>」。
  * **⚠️ 易錯陷阱**：注意題幹問「<span style="color: #7aa2f7">應使用（正面）</span>」還是「<span style="color: #e06c75">應避免（負面）</span>」，<span style="color: #e06c75">切勿看錯正反向</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 3.3 商業文案與簡報生成
* **核心名詞全稱**：
  * **NLG（自然語言生成, Natural Language Generation）**
  * **Automated Proofreading Tools（自動校對工具）**
* **商業生成 5 大場景秒配表**：
  | 商業應用場景 | 核心技術 / 最佳實務解答 | 易錯干擾項排除 |
  | :--- | :--- | :--- |
  | **1. 個人化市場營銷內容** | <span style="color: #7aa2f7">自然語言生成 (NLG, Natural Language Generation)</span>（動態生成客製文案與推廣信） | ❌ <span style="color: #e06c75">非傳統固定模板群發</span> |
  | **2. AI 簡報避免內容重複** | <span style="color: #7aa2f7">定期更新數據源</span> (Regularly Update Data Sources) | ❌ <span style="color: #e06c75">非固定提示詞、非減少頁數</span> |
  | **3. 運用 AI 確保簡報準確性** | <span style="color: #7aa2f7">使用自動校對工具</span> (Automated Proofreading Tools) | ❌ 切勿選「<span style="color: #e06c75">手動檢查</span>」或「<span style="color: #e06c75">請他人幫忙</span>」（非運用 AI） |
  | **4. AI 撰寫產品描述優勢** | <span style="color: #7aa2f7">撰寫引人注目的產品特點</span> (Catchy Product Highlights) | ❌ <span style="color: #e06c75">技術規格需工程師核對、法律合規需法務把關</span> |
  | **5. 生成式 AI 對法律領域影響** | <span style="color: #7aa2f7">提供法律文書和合約的自動生成</span>（起草初稿） | ❌ <span style="color: #e06c75">非增加律師負擔、非減少法律教育、非提高審判速度</span> |

<details>
<summary><b>💡 點擊展開白話案例：行銷金筆、簡報活水與 AI 法律助理（補充範例）</b></summary>

> ✍️ **行銷金筆 (NLG)** 幫不同客戶量身寫文案；📊 **簡報活水**靠「定期更新數據源」防重複、靠「AI 自動校對工具」抓錯字；⚖️ **AI 法律助理**負責一鍵起草合約與訴狀初稿，減輕重複文書勞動。
</details>

* **3.3 考點精華標記**：
  * **🔥 考試常考**：個人化行銷靠 <span style="color: #7aa2f7">NLG</span>；簡報防重複靠<span style="color: #7aa2f7">定期更新數據源</span>；AI 確保簡報準確靠<span style="color: #7aa2f7">使用自動校對工具</span>；產品描述靠<span style="color: #7aa2f7">撰寫引人注目的產品特點</span>；法律影響為<span style="color: #7aa2f7">提供法律文書和合約的自動生成</span>。
  * **⚠️ 易錯陷阱**：問「如何運用 AI 確保準確性」，千萬別選「<span style="color: #e06c75">人工手動檢查</span>」！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 3.4 AI 學習平台 Kaggle
* **Kaggle 核心定位與實踐途徑表**：
  | 維度 | 核心內容與考點結論 |
  | :--- | :--- |
  | **平台定位** | Google 旗下全球最大免費資料科學與 AI 競賽實作社群（<span style="color: #7aa2f7">最適合學習 AI 工具實作與應用之平台</span>） |
  | **3 大實踐途徑** | ① <span style="color: #7aa2f7">參加公開競賽</span> (Competitions)、② <span style="color: #7aa2f7">學習開源優秀解決方案</span> (Code / Notebooks)、③ <span style="color: #7aa2f7">論壇交流切磋</span> (Discussions) |
  | **干擾項排除** | ❌ <span style="color: #e06c75">YouTube Shorts / IG Reels / LINE 群組</span>（碎片娛樂通訊，無法寫 code 實作）；❌ <span style="color: #e06c75">免費訂閱優惠會員</span>（Kaggle 本身免費開放，此為虛構干擾項！） |

* **3.4 考點精華標記**：
  * **🔥 考試常考**：最適合學習 AI 工具實作與應用的平台為 <span style="color: #7aa2f7">Kaggle</span>。
  * **⚠️ 易錯陷阱**：問何者「**不是**」Kaggle 提升技能方式，必選「<span style="color: #e06c75">免費訂閱優惠會員</span>」！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 3.5 自動化工具 Zapier
* **核心名詞全稱**：
  * **No-Code Automation（無程式碼自動化）**、**Trigger（觸發條件）**、**Action（執行動作）**
  * **API（應用程式介面, Application Programming Interface）**、**SaaS（軟體即服務, Software as a Service）**
* **Zapier 運作與邊界流**：`觸發條件 (Trigger) ➔ 跨平台執行動作 (Action：傳通知/存檔案/同步表格)` ｜ ⚠️ **致命限制：`無法完整理解語意`**

* **Zapier 能與不能對照表**：
  | 能力分類 | 具體任務項目 | 原因解析 |
  | :--- | :--- | :--- |
  | **✅ 擅長任務** | <span style="color: #7aa2f7">傳送通知</span> (Slack/Email)、<span style="color: #7aa2f7">儲存檔案</span> (Drive/Dropbox)、<span style="color: #7aa2f7">同步表格</span> (Sheets) | 基於固定規則與 API (Application Programming Interface) 管線自動觸發執行 |
  | **❌ 無法完成** | <span style="color: #e06c75">完整理解語意</span>（必考答案！） | 僅為規則串接管線，本身<span style="color: #e06c75">不具備 LLM (Large Language Model) 的自然語言理解</span> (NLU, Natural Language Understanding) 能力 |

<details>
<summary><b>💡 點擊展開白話案例：辦公室的自動齒輪傳送帶 vs 讀懂文意的秘書</b></summary>

> ⚙️ **Zapier（不識字的自動傳送帶）**：
> 只要信箱收到附件（Trigger），傳送帶就自動把檔案存進雲端並發通知（Action）。但傳送帶本身是機械規則，根本「不識字」——**無法完整理解合約裡寫了什麼語意**！
</details>

* **3.5 考點精華標記**：
  * **🔥 考試常考**：Zapier 擅長<span style="color: #7aa2f7">傳送通知、儲存檔案、同步表格</span>；無法完成「<span style="color: #e06c75">完整理解語意</span>」。
  * **⚠️ 易錯陷阱**：問 Zapier 無法完成哪項任務，必選「<span style="color: #e06c75">完整理解語意</span>」！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 3.6 音樂創作 AIVA
* **核心名詞全稱**：
  * **AIVA（人工智慧虛擬藝術家, Artificial Intelligence Virtual Artist）**
* **AI 歌曲創作黃金組合**：`GPT（寫歌詞 / 韻腳 / 結構）` ＋ `AIVA（自動作曲 / 編曲 / 配樂）` ➔ `完整原創歌曲`

* **3.6 考點精華標記**：
  * **🔥 考試常考**：常與 <span style="color: #7aa2f7">GPT</span> 結合使用來<span style="color: #7aa2f7">創作歌曲</span>的 AI 音樂工具為 <span style="color: #7aa2f7">AIVA</span>。
  * **⚠️ 易錯陷阱**：切勿選 <span style="color: #e06c75">Hugging Face</span>（開源模型庫）、<span style="color: #e06c75">Figma</span>（UI/UX 介面設計, User Interface / User Experience）或 <span style="color: #e06c75">Canva</span>（平面排版）！

<details>
<summary><b>💡 點擊展開白話案例：作詞家 GPT 與 作曲大師 AIVA 的錄音室合作</b></summary>

> 🎵 **作詞家 GPT ＋ 作曲家 AIVA**：GPT 負責寫出押韻感人的歌詞，交給坐在鋼琴前的 AIVA 自動譜出旋律與交響伴奏，兩者合璧即成完整歌曲！
</details>

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 3.7 程式生成 Copilot
* **核心名詞全稱**：
  * **GitHub Copilot（AI 結對編程助手, AI Pair Programmer）**
  * **IDE（整合開發環境, Integrated Development Environment）**
  * **HTML（超文本標記語言, HyperText Markup Language）**、**CSS（階層式樣式表, Cascading Style Sheets）**
* **網頁前端生成流**：`自然語言註解 / 描述` ➔ `GitHub Copilot (嵌入 VS Code 等 IDE)` ➔ `自動生成 HTML 結構與 CSS 樣式代碼`

* **3.7 考點精華標記**：
  * **🔥 考試常考**：在網頁開發中，常用來生成 <span style="color: #7aa2f7">HTML 與 CSS 結構</span>的 AI 工具為 <span style="color: #7aa2f7">Copilot</span>（GitHub / Microsoft）。
  * **⚠️ 易錯陷阱**：切勿選 <span style="color: #e06c75">Bard (Gemini)</span>、<span style="color: #e06c75">Claude</span>（通用對話模型）或 <span style="color: #e06c75">Notion</span>（筆記軟體）！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

## 第四部分 生成式AI倫理法律

### 4.1 深度偽造 (Deepfake)
* **核心名詞全稱**：
  * **Deepfake（深度偽造，結合 Deep Learning 與 Fake）**
  * **GAN（生成對抗網路, Generative Adversarial Network）**
* **技術根源與風險流**：`基於 GAN 等生成技術（換臉/擬聲）` ➔ `三大社會風險：① 身分盜用詐騙 (Identity Theft)、② 造假影片與假新聞 (Fake News)、③ 操弄政治與破壞選舉公平`

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 4.2 歐盟 AI 法案 AI 治理與臺灣 TAIDE 計畫
* **核心名詞全稱**：
  * **EU AI Act（歐盟人工智慧法案, European Union Artificial Intelligence Act）**
  * **Trustworthy AI（可信賴人工智慧）**、**Accountability（問責制）**、**Safety Assessment（安全性評估）**
  * **TAIDE（可信任生成式 AI 對話引擎, Trustworthy AI Dialog Engine）**、**NSTC（國家科學及技術委員會 / 國科會, National Science and Technology Council）**、**ML（機器學習, Machine Learning）**
  * **Rollback（版本還原）**、**Kill Switch（緊急停機開關 / 人工介入）**
  * **NPC（非玩家角色, Non-Player Character）**、**IT（資訊科技, Information Technology）**
* **歐盟風險 4 級金字塔**：`① 不可接受風險 (Unacceptable Risk：完全禁止)` ➔ `② 高風險 (High Risk：強制嚴格規範)` ➔ `③ 有限風險 (Limited Risk：基本資訊揭露)` ➔ `④ 最低風險 (Minimal Risk：免受規範)`
* **AI 全生命週期治理流**：`設計 ➔ 部署 ➔ 使用 ➔ 維護`（⚠️ **全週期持續管控，絕非部署完就結束！**）

* **歐盟《AI 法案 (EU AI Act)》四級風險必背配對表**：
  | 風險等級 | 管制強度 | 核心規範要求 | 必背代表應用場景（考題直接配對！） |
  | :--- | :--- | :--- | :--- |
  | **1. 不可接受風險** | <span style="color: #e06c75">完全禁止</span> (Prohibited) | 威脅人類安全、生存權與基本尊嚴 | <span style="color: #e06c75">社會信用評分系統</span> (Social Scoring)、認知行為操縱 |
  | **2. 高風險** | <span style="color: #7aa2f7">強制嚴格規範</span> | 攸關性命、健康、升學就業重大權益，需嚴格審查 | <span style="color: #7aa2f7">醫療診斷 AI</span>、<span style="color: #7aa2f7">招聘與求職篩選系統</span> (Recruitment)、司法執法 |
  | **3. 有限風險** | <span style="color: #7aa2f7">基本資訊揭露</span> | 履行透明度義務，<span style="color: #7aa2f7">主動告知使用者正在與 AI 互動</span> | <span style="color: #7aa2f7">聊天機器人</span> (Chatbot)、Deepfake 標註 |
  | **4. 最低風險** | <span style="color: #7aa2f7">免受規範</span> | 日常應用無重大侵害疑慮，自願性自律 | <span style="color: #7aa2f7">影片推薦系統</span>、<span style="color: #7aa2f7">電玩遊戲角色自動回應</span> (NPC, Non-Player Character)、垃圾信過濾 |

* **可信賴 AI 四大原則、治理機制與臺灣 TAIDE 計畫速查表**：
  | 治理維度 | 核心原則 / 機制（含英文） | 關鍵考點結論與常見誘答排除 |
  | :--- | :--- | :--- |
  | **可信賴 AI 4 原則** | <span style="color: #7aa2f7">公平性 (Fairness)、透明性 (Transparency)、安全性 (Safety)、可解釋性 (Explainability)</span> | 當 <span style="color: #e06c75">AI 模型偏誤導致特定族群受害</span>時，屬於「<span style="color: #7aa2f7">公平性 (Fairness)</span>」面向！ |
  | **倫理準則問責與評估** | <span style="color: #7aa2f7">問責制</span> (Accountability) ＆ <span style="color: #7aa2f7">安全性評估</span> (Safety Assessment) | 問責制體現為「<span style="color: #7aa2f7">政府機構設立生成式 AI 監管機制</span>」；建議進行「<span style="color: #7aa2f7">安全性評估</span>」 |
  | **AI 風險 3 大來源** | <span style="color: #7aa2f7">資料面向、模型面向、使用面向</span> | <span style="color: #e06c75">資料偏差</span> (Data Bias) ➔ <span style="color: #e06c75">模型黑箱/幻覺</span> (Hallucination) ➔ <span style="color: #e06c75">濫用/Deepfake/侵權</span> |
  | **組織權責分工** | <span style="color: #7aa2f7">跨部門全員參與</span> | 涵蓋資訊 (IT, Information Technology)、法務、業務與決策者，<span style="color: #e06c75">絕非僅由 IT 部門單獨負責</span>！ |
  | **異常應對 4 機制** | 目標：出錯時能有效控管與修正 | ① <span style="color: #7aa2f7">即時監控</span>、② <span style="color: #7aa2f7">錯誤回報</span>、③ <span style="color: #7aa2f7">版本還原</span> (Rollback)、④ <span style="color: #7aa2f7">人工介入</span> (Kill Switch 緊急停機) |
  | **臺灣治理精神** | AI 自律規範（金融/公部門） | 秉持「<span style="color: #7aa2f7">審慎、自律、資訊揭露</span>」三大原則 |
  | **臺灣 TAIDE 計畫** | **可信任生成式 AI 對話引擎** (Trustworthy AI Dialog Engine) | 由 <span style="color: #7aa2f7">國科會 (NSTC)</span> 主導，基於開源模型融入臺灣繁體中文、文化與法規，核心技術本質為<span style="color: #7aa2f7">機器學習 (ML, Machine Learning)</span>；❌ 切勿誤選<span style="color: #e06c75">虛擬實境 (VR, Virtual Reality)、區塊鏈或量子計算</span> |

<details>
<summary><b>💡 點擊展開白話案例：歐盟四級紅綠燈、飛機緊急停機鈕與臺灣在地大腦 TAIDE（原著範例精煉）</b></summary>

> 🚦 **1. 歐盟四級紅綠燈**：
> * **紅燈（不可接受/禁止）**：像違禁品全面禁止（**社會信用評分**）。
> * **黃燈（高風險/嚴管）**：攸關人命與飯碗，必須嚴格安檢（**醫療診斷、AI 招聘篩選**）。
> * **藍燈（有限風險/揭露）**：像食品標示成分，必須誠實說「我是 AI」（**聊天機器人**）。
> * **綠燈（最低風險/免管）**：日常娛樂無害（**影片推薦系統、電玩 NPC**）。
>
> 🛑 **2. 飛機緊急停機鈕 (Kill Switch)**：
> 治理不是要求自動駕駛「100% 永不出錯」，而是遇到亂流失靈時，機長能按下緊急停機鈕 (Kill Switch) 手動接管、退回穩定版本！
>
> 🇹🇼 **3. 臺灣可信賴 AI 代表實踐——TAIDE 核心價值比喻**：
> 國外模型常把臺灣的「計程車」說成「出租車」。國科會主導的 **TAIDE（可信任生成式 AI 對話引擎）** 就像在臺灣本土長大的語言學家，利用「**機器學習 (ML)**」技術大量吸收繁體中文與臺灣文化法規，打造值得信賴的本土對話引擎。
</details>

* **4.2 考點精華標記**：
  * **🔥 考試常考**：
    1. **歐盟四級風險配對**：禁止＝<span style="color: #e06c75">社會信用評分</span>；高風險＝<span style="color: #7aa2f7">醫療診斷 / 招聘系統</span>；有限風險＝<span style="color: #7aa2f7">聊天機器人</span>；最低風險＝<span style="color: #7aa2f7">影片推薦 / 遊戲自動回應</span>。
    2. **特定族群因偏誤受害**：屬於「<span style="color: #7aa2f7">公平性 (Fairness)</span>」面向。
    3. **歐盟問責制體現**：<span style="color: #7aa2f7">政府機構設立生成式 AI 監管機制</span>；**建議評估類型**：<span style="color: #7aa2f7">安全性評估</span>。
    4. **臺灣 TAIDE 計畫**：主導單位為<span style="color: #7aa2f7">國科會 (NSTC)</span>，技術核心本質為<span style="color: #7aa2f7">機器學習 (ML, Machine Learning)</span>。
  * **⚠️ 易錯陷阱**：
    1. 聊天機器人是<span style="color: #7aa2f7">有限風險</span>（<span style="color: #e06c75">非高風險</span>）；影片推薦是<span style="color: #7aa2f7">最低風險</span>（<span style="color: #e06c75">非有限風險</span>）。
    2. 族群受害選<span style="color: #7aa2f7">公平性</span>（<span style="color: #e06c75">切勿選問責性、隱私性或合法性</span>）。
    3. AI 治理<span style="color: #e06c75">絕非部署完就結束</span>，且<span style="color: #e06c75">絕非僅由 IT 部門負責</span>！
    4. **TAIDE** 的技術本質是<span style="color: #7aa2f7">機器學習</span>，<span style="color: #e06c75">切勿誤選為虛擬實境 (VR)、區塊鏈或量子計算</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 4.3 資料治理三大目標與資安防護
* **🔗 上下游串聯**：承接 **4.2 AI 風險三大來源中的「資料面向」**，若要從源頭杜絕訓練資料偏見、外洩與侵權，必須先落實本節的**資料治理三大目標**與**加密防護**，並進一步遵循 **4.4 全球隱私法規**。
* **核心名詞全稱**：
  * **Data Governance（資料治理）**、**GIGO（垃圾進、垃圾出, Garbage In, Garbage Out）**
  * **Symmetric Encryption（對稱加密）** vs **Asymmetric Encryption（非對稱加密）**
  * **Access Control（存取權限控制）**、**Social Engineering（社交工程）**
* **資料治理 3 大目標**：`① 保證資料品質 (Quality：完整性＋準確性)` ＋ `② 確保透明度與合規 (Transparency：記錄來源授權＋可解釋)` ＋ `③ 維持資料一致性 (Consistency：標準統一穩定輸出)`

* **資安加密與威脅速查表**：
  | 資安維度 | 技術 / 威脅名稱 | 一句話白話秒懂（怎麼運作？） |
  | :--- | :--- | :--- |
  | **資料加密** | **對稱加密 (Symmetric Encryption)** | 上鎖（加密）跟開鎖（解密）都用「<span style="color: #7aa2f7">同一把金鑰</span>」（速度快，適合鎖自己硬碟裡的大批檔案） |
  | **資料加密** | **非對稱加密 (Asymmetric Encryption)** | 分成「<span style="color: #7aa2f7">公開金鑰</span>（像郵筒投信口，大家都能鎖）＋ <span style="color: #7aa2f7">私密金鑰</span>（像郵差鑰匙，只有本人能開）」，適合網路安全傳輸 |
  | **權限管理** | **最小權限原則 (Access Control)** | 依職務只給剛好的權限，嚴格限制「<span style="color: #7aa2f7">誰能看、誰能改、誰能刪</span>」 |
  | **人為威脅** | **社交工程 (Social Engineering)** | 不攻破電腦技術，而是利用<span style="color: #e06c75">人性弱點與騙術</span>（如釣魚信、假冒主管）騙你主動交出密碼個資 |

<details>
<summary><b>💡 點擊展開白話案例：資料治理如同買保險與一把鑰匙 vs 郵筒投信（原著範例）</b></summary>

> 🛡️ **資料治理像買保險**：不管好資料，就會引發資安外洩、版權挨告、模型胡言亂語與巨額罰款。
> 🔑 **對稱 vs 非對稱加密**：**對稱加密**像一般保險箱，鎖門開門用「同一把鑰匙」；**非對稱加密**像街邊「郵筒」，路人都能用「公鑰（投信口）」把信投進去，但只有郵差拿「私鑰（後門鑰匙）」能打開拿信！
</details>

* **4.3 考點精華標記**：
  * **🔥 考試常考**：
    1. **資料治理三大目標**：<span style="color: #7aa2f7">品質（完整/準確）、透明（合規/信任）、一致性（穩定表現）</span>。
    2. **對稱 vs 非對稱加密**：<span style="color: #7aa2f7">同一把金鑰</span> vs <span style="color: #7aa2f7">公鑰＋私鑰</span>。
  * **⚠️ 易錯陷阱**：AI 產出偏見與侵權的主因常源於「<span style="color: #e06c75">訓練資料治理不當</span>」，非單純演算法數學問題！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 4.4 隱私法規 GDPR HIPAA 與個資法
* **🔗 上下游串聯**：承接 **4.3 資料治理與加密技術**，當企業蒐集個資或醫療數據訓練 AI 時，技術上的「加密與刪除權限」即直接對應本節 **GDPR（被遺忘權）**、**HIPAA（加密保護病患資料）** 與 **臺灣個資法** 的強制合規要求。
* **核心名詞全稱**：
  * **GDPR（歐盟一般資料保護規則, General Data Protection Regulation）**、**Right to be Forgotten（被遺忘權）**
  * **HIPAA（美國健康保險可攜性和責任法案, Health Insurance Portability and Accountability Act）**、**PHI（受保護健康資訊, Protected Health Information）**
  * **PDPA（個人資料保護法 / 臺灣個資法, Personal Data Protection Act）**

* **全球三大隱私法規必背對照表**：
  | 法規名稱 | 管轄與適用領域 | 核心關鍵要求（必考！） | 違規最高罰則 / 條文 |
  | :--- | :--- | :--- | :--- |
  | **歐盟 GDPR (General Data Protection Regulation)** | 歐盟／全球最嚴個資標準 | <span style="color: #7aa2f7">明確同意</span> (Explicit Consent)、<span style="color: #7aa2f7">被遺忘權（刪除權）</span>、隱私設計 | 最高處<span style="color: #e06c75">全球年度營收 4%</span>（或 2,000 萬歐元） |
  | **美國 HIPAA (Health Insurance Portability and Accountability Act)** | 美國／<span style="color: #7aa2f7">醫療與健康資料</span> (PHI, Protected Health Information) | <span style="color: #7aa2f7">使用加密技術保護患者資料</span>、嚴格存取授權 | 鉅額民刑事罰款（⚠️ <span style="color: #e06c75">商業與非商業皆嚴格受管</span>） |
  | **臺灣個資法 (PDPA, Personal Data Protection Act)** | 臺灣公私部門個資 | <span style="color: #7aa2f7">第 5 條</span>（明確告知蒐集目的）<br><span style="color: #7aa2f7">第 11 條</span>（當事人查詢、停止利用、刪除權） | 行政罰鍰、限期改正、民刑事責任 |

<details>
<summary><b>💡 點擊展開白話案例：被遺忘權與病歷保險箱（法律生活比喻）</b></summary>

> 🧽 **GDPR 被遺忘權**：用戶有權要求平台徹底刪除自己的個資足跡，不能偷偷留著訓練 AI。
> 🏥 **HIPAA 病歷加密保險箱**：醫院處理病患健康資料，最重要的安全措施就是**「使用加密技術保護患者資料」**，絕不可公開或僅存紙本！
</details>

* **4.4 考點精華標記**：
  * **🔥 考試常考**：
    1. **GDPR**：<span style="color: #7aa2f7">明確同意、被遺忘權</span>、最高罰全球營收 <span style="color: #e06c75">4%</span>。
    2. **美國 HIPAA 處理健康資料關鍵措施**：<span style="color: #7aa2f7">使用加密技術保護患者資料</span>。
    3. **臺灣個資法**：<span style="color: #7aa2f7">第 5 條</span>（告知目的）、<span style="color: #7aa2f7">第 11 條</span>（查詢/停止/刪除權）。
  * **⚠️ 易錯陷阱**：HIPAA 考題切勿選「<span style="color: #e06c75">將患者資料公開</span>」、「<span style="color: #e06c75">保存於紙質檔案</span>」或「<span style="color: #e06c75">僅非商業用途受保護</span>」！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 4.5 模型可解釋性 LIME vs SHAP
* **🔗 上下游串聯**：落實 **4.2 可信賴 AI 四大原則中的「可解釋性 (Explainability)」**，為了解決 **1.3 深度學習神經網路的黑箱問題**，實務上便透過本節的 **LIME（單筆局部解釋）** 與 **SHAP（全局特徵貢獻）** 拆解 AI 的決策理由。
* **核心名詞全稱**：
  * **XAI（可解釋人工智慧, Explainable Artificial Intelligence）**
  * **LIME（局部可解釋模型無關解釋, Local Interpretable Model-agnostic Explanations）**
  * **SHAP（沙普利加性解釋, SHapley Additive exPlanations）**
  * **CC License（創用 CC 授權, Creative Commons License）**
* **可解釋性 XAI (Explainable AI) 雙雄秒配**：`LIME ➔ 只針對「單筆預測」做局部解釋 (Local)` ｜ `SHAP ➔ 算出「整體每個特徵各佔多少貢獻」(Global & Local，基於賽局理論)`

* **LIME vs SHAP 與開源共用原則表**：
  | 主題維度 | 工具 / 平台名稱 | 一句話白話秒懂（它怎麼解釋 AI 黑箱？） |
  | :--- | :--- | :--- |
  | **可解釋性工具** | **LIME (Local Interpretable Model-agnostic Explanations)** | 像拿放大鏡只看「<span style="color: #7aa2f7">單一筆預測結果</span>」做<span style="color: #7aa2f7">局部解釋</span> (Local，例如：為什麼張三這筆貸款沒過？) |
  | **可解釋性工具** | **SHAP (SHapley Additive exPlanations)** | 像拔河比賽算功勞（基於<span style="color: #7aa2f7">合作賽局理論</span>），算出「<span style="color: #7aa2f7">整體特徵貢獻度分配</span>」（如：收入佔 50%、年齡佔 20%） |
  | **模型與數據共用平台** | **Hugging Face ＆ Kaggle** | 全球最大開源模型與資料集共用樞紐；必須遵循 <span style="color: #7aa2f7">CC 授權 (創用 CC, Creative Commons License)</span> 並完整標註資料來源出處 |

<details>
<summary><b>💡 點擊展開白話案例：放大鏡看個人 (LIME) vs 拔河比賽貢獻度分帳 (SHAP)（補充範例）</b></summary>

> 🔍 **LIME（放大鏡看單人）**：行員拿放大鏡看張三的申請表，解釋「張三這筆為何沒過」。
> ⚖️ **SHAP（拔河比賽分獎金）**：用賽局理論算出全體決策中，「收入、年齡、信用分數」各自出了幾分力（整體特徵貢獻度）。
</details>

* **4.5 考點精華標記**：
  * **🔥 考試常考**：<span style="color: #7aa2f7">LIME 管單一預測局部解釋</span>；<span style="color: #7aa2f7">SHAP 管整體特徵貢獻分配</span>；共用平台代表為 <span style="color: #7aa2f7">Hugging Face</span> 與 <span style="color: #7aa2f7">Kaggle</span>。
  * **⚠️ 易錯陷阱**：LIME（<span style="color: #7aa2f7">局部單筆</span>）與 SHAP（<span style="color: #7aa2f7">整體特徵貢獻</span>）<span style="color: #e06c75">切勿記反</span>！

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

## 第五部分 生成式AI工具與開發商總覽

> 💡 **5 秒速查指南**：全量收錄本筆記與題庫所有工具，遵循**「工具 ➔ 開發商 ➔ 架構 ➔ 定位 ➔ 案例 ➔ 考點 ➔ 陷阱」**極簡關鍵字對照。

---

### 5.1 文本與對話語言模型 (LLMs)

| 工具 / 模型 | 開發商 | 授權 / 架構 | 核心定位與受眾 | 💡 具體應用案例 | 🔥 考試常考重點 | ⚠️ 易錯陷阱提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ChatGPT ＆ GPTs** | **OpenAI** | 閉源商業<br>(Decoder-only + <span style="color: #7aa2f7">RLHF</span>) | 通用對話大語言模型與客製化 AI 生態系平臺 | 撰寫行銷文案、智能客服、透過 GPTs 打造專屬領域助手 | 對話式互動；具備<span style="color: #7aa2f7">上下文學習 (ICL)</span>、以 GPT-3.5 + <span style="color: #7aa2f7">RLHF 行為對齊</span>、打造 <span style="color: #7aa2f7">GPTs 平臺化生態系</span> | 閉源服務；仍會產生「<span style="color: #e06c75">幻覺</span> (Hallucination)」 |
| **GPT-1 / 3 / 4V / 4o** | **OpenAI** | 閉源模型<br>(自迴歸 Decoder) | 業界標竿 LLM (Large Language Model) 底座（4V/4o 支援多模態） | 企業 API (Application Programming Interface) 串接、跨模態圖文音綜合推理 | <span style="color: #7aa2f7">單向文字接龍</span>；Few-shot 與多模態代表 | GPT-3 為純文字模型，<span style="color: #e06c75">無法直接生圖</span> |
| **Gemini 1.5 系列<br>(Pro / Flash，原 Bard)** | **Google**<br>(DeepMind) | 閉源商業<br>(原生多模態) | 搜尋整合、手機端導入 (Pixel + Android)、跨文字影音代碼理解 | 上傳 1 小時影片或長篇 PDF 文件即時解析與檢索 | <span style="color: #7aa2f7">原生多模態架構</span>；<span style="color: #7aa2f7">搜尋與 Android 手機端整合</span>；Bard 已更名為 Gemini | 通用對話模型，<span style="color: #e06c75">非 IDE (Integrated Development Environment) 內建網頁代碼外掛</span> |
| **Claude 3 系列<br>(Opus / Sonnet / Haiku)** | **Anthropic** | 閉源商業<br>(Constitutional AI) | 符合人類價值觀、高安全性、支援 <span style="color: #7aa2f7">200K tokens 超長上下文</span> | 大型代碼庫重構、長篇法規合約逐條審閱 | Anthropic 開發；<span style="color: #7aa2f7">憲政 AI 安全對齊</span>、<span style="color: #7aa2f7">200K 超長上下文</span> | 通用語言模型，<span style="color: #e06c75">非 IDE 專用代碼外掛</span> |
| **Meta AI ＆ LLaMA 3** | **Meta** | **開源權重**<br>(Large Language Model Meta AI) | 全球最具代表性之開源大型語言模型與社群內建助手 | 企業免費下載 LLaMA 3 權重於本地微調部署；FB/IG 社群互動 | 主打「<span style="color: #7aa2f7">開源推廣</span>」標竿、<span style="color: #7aa2f7">社群導向整合</span>（Meta App 內建 AI 助手） | 屬開源權重模型，與閉源的 GPT-4 / Claude 路線不同 |
| **Grok** | **xAI**<br>(Elon Musk) | 閉源商業 / 部分開源<br>(Transformer) | 深度結合 **X（原 Twitter）社群**之即時資訊對話模型 | 即時擷取 X 平台全球熱門貼文與突發新聞進行分析問答 | 結合 <span style="color: #7aa2f7">X (原 Twitter) 社群</span>、強調<span style="color: #7aa2f7">言論自由與即時資訊</span> | 開發商為 Elon Musk 創立的 <span style="color: #7aa2f7">xAI</span> |
| **Apple Intelligence** | **Apple** | 閉源系統級整合<br>(Foundation Models 框架) | 蘋果生態系個人化智慧系統，強調裝置端運算與隱私 | iPhone / Mac 離線郵件摘要、Siri 跨 App 操作、通知重點整理 | 強調「<span style="color: #7aa2f7">離線運算、隱私保護</span>」與多語言支援 | 優先在終端設備離線處理，保護個人隱私 |
| **Perplexity** | **Perplexity AI** | 閉源服務<br>(LLM + 即時搜尋) | 對話式搜尋引擎，提供即時檢索與引用來源 | 搜尋最新科技動態並產出附真實網址之報告 | 專長<span style="color: #7aa2f7">資訊檢索與摘要</span>；<span style="color: #e06c75">非生圖/生影片工具</span> | 常作為「生成電影風格圖片」之<span style="color: #e06c75">干擾誘答項</span>！ |
| **TAIDE** | **臺灣國科會**<br>(NSTC, National Science and Technology Council) | 開源繁中模型<br>(基於 LLaMA 微調) | 臺灣在地文化、繁體中文與法規可信賴模型 (Trustworthy AI Dialog Engine) | 公部門公文擬稿、臺灣在地法規與用語問答 | 技術本質為<span style="color: #7aa2f7">機器學習 (ML, Machine Learning)</span>；國科會主導 | <span style="color: #e06c75">切勿誤選為 VR (Virtual Reality)、區塊鏈或量子計算</span>！ |
| **BERT** | **Google**<br>(2018) | 開源模型<br>(雙向 Encoder) | NLU (Natural Language Understanding) 語言理解標竿，專攻分類與語意表徵 (Bidirectional Encoder Representations from Transformers) | Google 搜尋長句理解、金融留言正負情緒分類 | <span style="color: #e06c75">⚠️ 鑑別式 AI</span>、<span style="color: #7aa2f7">雙向 MLM (Masked Language Model) 猜詞 ＋ NSP (Next Sentence Prediction)</span> | 屬於「<span style="color: #e06c75">鑑別式 AI</span>」，<span style="color: #e06c75">絕非生成式 AI</span>！ |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 5.2 圖像生成與編輯平台 (Image Generation)

| 工具 / 模型 | 開發商 | 授權 / 架構 | 核心定位與受眾 | 💡 具體應用案例 | 🔥 考試常考重點 | ⚠️ 易錯陷阱提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DALL·E 系列 ＆ CLIP** | **OpenAI** | 閉源商業<br>(Diffusion + CLIP, Contrastive Language-Image Pre-training) | 敘事式文字生圖 (Text-to-Image) 工具 | **生成電影風格圖片**、遊戲概念海報插畫 | 常用於生成<span style="color: #7aa2f7">電影風格圖片</span>；OpenAI 畫圖工具 | 是靜態畫圖工具，<span style="color: #e06c75">絕非影片生成工具</span>！ |
| **Midjourney** | **Midjourney** | 閉源獨立平台<br>(自研 Diffusion) | 頂級藝術美感與光影質感生圖平台 | 時尚雜誌概念封面、奇幻場景氛圍渲染 | <span style="color: #7aa2f7">畫面風格穩定、藝術美感極強</span>；閉源平台 | <span style="color: #e06c75">無開源權重</span>，無法免費下載至本地部署 |
| **Stable Diffusion** | **Stability AI**<br>(2022) | **開源免費**<br>(潛在擴散 LDM, Latent Diffusion Model) | 全球代表性開源生圖底座，支援自由微調 | 本地端顯卡部署，訓練企業專屬 LoRA (Low-Rank Adaptation) 模型 | <span style="color: #7aa2f7">開源、免費、可自由修改與本地部署</span> | 開發商是 <span style="color: #7aa2f7">Stability AI</span>（<span style="color: #e06c75">非 OpenAI/Google</span>） |
| **DreamStudio** | **Stability AI** | 閉源網頁<br>(LDM, Latent Diffusion Model) | Stability AI 官方簡化版網頁操作介面 | 免裝環境，新手直接在瀏覽器輸入文字生圖 | 官方介面、操作簡易、<span style="color: #7aa2f7">適合初學者入門</span> | 為網頁介面，進階本地微調多直接用 SD (Stable Diffusion) |
| **Leonardo AI** | **Leonardo.Ai** | 閉源平台<br>(微調 SD, Stable Diffusion) | 遊戲美術、概念設計與幻想風格創作平台 | 快速生成 RPG (Role-Playing Game) 遊戲角色立繪、道具與貼圖 | 擅長<span style="color: #7aa2f7">概念圖與幻想風格</span>，內建多微調模型 | 底層源於 SD，但平台本身為商業服務 |
| **Adobe Firefly** | **Adobe** | 商業閉源<br>(自研 Diffusion) | 專為商業設計打造之無版權疑慮生圖引擎 | Photoshop「生成填色」擴展商業廣告海報 | <span style="color: #7aa2f7">合法授權素材訓練，商用版權安全無虞</span> | 考「合法素材來源、商用安全」唯一首選！ |
| **Canva / Figma** | **Canva / Figma** | SaaS (Software as a Service) 平台 | 平面圖文排版 (Canva) 與 UI/UX (User Interface / User Experience) 設計 (Figma) | 製作社群宣傳圖卡；繪製 App 介面原型 | 視覺排版與介面工具，<span style="color: #e06c75">非 AI 作曲或寫碼工具</span> | 常在「結合 GPT 創作歌曲 (AIVA)」題當<span style="color: #e06c75">干擾項</span> |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 5.3 影片生成與剪輯工具 (Video Generation)

| 工具 / 模型 | 開發商 | 授權 / 架構 | 核心定位與受眾 | 💡 具體應用案例 | 🔥 考試常考重點 | ⚠️ 易錯陷阱提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sora** | **OpenAI**<br>(2024) | 閉源模型<br>(<span style="color: #7aa2f7">Diffusion + Transformer</span>) | 文生成影片（語意驅動敘事），長達 60 秒高連貫影片 | 一段文字生成好萊塢級動態街景或奇幻短片 | <span style="color: #7aa2f7">Diffusion 當畫圖工人、Transformer 當導演編劇（引導模組/條件控制器）</span>；畫質佳、敘事強、世界理解力高 | 是正港<span style="color: #7aa2f7">影片生成</span>工具（勿與 DALL·E 混淆） |
| **Runway<br>(Gen-1 ~ Gen-4)** | **Runway ML** | 閉源網頁<br>(<span style="color: #7aa2f7">改良版 Diffusion</span>) | 創作者影片草圖、剪輯特效，賦能**非專業創作者** | 自媒體文字生影片、動態筆刷局部特效合成 | 目標客群為「<span style="color: #7aa2f7">非專業創作者</span>」；操作簡易、生成快速、商業已落地 | 受眾選<span style="color: #7aa2f7">非專業創作者</span>（<span style="color: #e06c75">非程式設計師/硬體師</span>） |
| **Pika** | **Pika Labs** | 閉源平台<br>(<span style="color: #7aa2f7">動畫導向 Diffusion + 多模態控制</span>) | 原創角色動畫、社群短片生成平台 | 將靜態動漫人像一鍵生成眨眼微風吹拂短片 | <span style="color: #7aa2f7">風格自由、互動性高、創作流程友善</span> | 屬影片生成工具，非純靜態繪圖工具 |
| **Google Lumiere / Veo** | **Google / DeepMind** | 閉源模型<br>(<span style="color: #7aa2f7">Space-Time U-Net</span>) | 高畫質敘事影片與自然語言超高清影片生成 | 輸入分鏡腳本一次生成整段時空連貫影視片段 | Lumiere 採用 <span style="color: #7aa2f7">Space-Time U-Net</span>；<span style="color: #7aa2f7">畫面穩定、邏輯一致、細節表現極佳</span> | 屬影片生成工具 |
| **FlexClip** | **PearlMountain** | 閉源雲端 SaaS (Software as a Service) | 線上輕量化 AI 影片製作、文字轉影片平台 | 輸入行銷文案自動配音、上字幕生成宣傳短片 | 代表性<span style="color: #7aa2f7">線上 AI 影片生成與剪輯工具</span> | 屬<span style="color: #7aa2f7">影片生成與剪輯</span>工具（<span style="color: #e06c75">DALL·E 才不是</span>） |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 5.4 音樂與語音生成工具 (Audio and Music)

| 工具 / 模型 | 開發商 | 授權 / 架構 | 核心定位與受眾 | 💡 具體應用案例 | 🔥 考試常考重點 | ⚠️ 易錯陷阱提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AIVA (Artificial Intelligence Virtual Artist)** | **AIVA Technologies** | 專利深度學習引擎 | AI 自動作曲與編曲平台（古典/流行/配樂） | **GPT 寫歌詞 ＋ AIVA 自動譜曲編曲**成完整歌 | <span style="color: #7aa2f7">常與 GPT 結合使用來創作歌曲的工具</span> | 勿選 <span style="color: #e06c75">Hugging Face、Figma 或 Canva</span>！ |
| **Google Assistant / Siri** | **Google / Apple** | 閉源語音助理<br>(早期基於 **RNN, Recurrent Neural Network** 序列模型) | 智慧語音辨識 (Speech Recognition) 與語音指令理解助理 | 手機接收使用者連續語音指令，記住前詞並理解整句話意圖 | <span style="color: #7aa2f7">RNN 處理時間序列與語音辨識</span>之經典應用代表 | 依賴時序依賴性理解語音，<span style="color: #e06c75">非空間影像辨識 (CNN, Convolutional Neural Network)</span> |
| **YouTube 自動字幕 / Google 翻譯** | **Google** | 閉源雲端服務<br>(早期基於 **RNN / Seq2Seq, Sequence-to-Sequence**) | 自動字幕生成 (Automatic Subtitling) 與跨語言機器翻譯 | YouTube 影片即時語音轉錄字幕；Google 翻譯依前後語境轉換語言 | <span style="color: #7aa2f7">RNN 記憶機制與時序依賴性</span>之經典應用案例 | 傳統純 RNN 遇超長句子易發生「<span style="color: #e06c75">梯度消失</span>」 |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 5.5 程式輔助與工作流程自動化 (Coding and Automation)

| 工具 / 模型 | 開發商 | 授權 / 架構 | 核心定位與受眾 | 💡 具體應用案例 | 🔥 考試常考重點 | ⚠️ 易錯陷阱提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Copilot** | **GitHub / Microsoft** | 閉源商業外掛<br>(嵌入 VS Code 等 IDE, Integrated Development Environment) | 軟體工程師專用 AI 結對編程助手 (AI Pair Programmer) | 前端開發依註解**自動生成 HTML (HyperText Markup Language) 與 CSS (Cascading Style Sheets) 結構** | 網頁開發生成 <span style="color: #7aa2f7">HTML 與 CSS 結構</span>首選 <span style="color: #7aa2f7">Copilot</span> | 勿選 <span style="color: #e06c75">Bard (Gemini)、Claude 或 Notion</span>！ |
| **CodeWhisperer** | **Amazon (AWS)** | 閉源雲端外掛<br>(AWS 生態系整合) | 企業級 B2B 程式碼即時生成與資安漏洞掃描助手 | 撰寫 AWS 雲端部署腳本與企業後端程式碼自動補全 | 結合 <span style="color: #7aa2f7">AWS 雲端整合、B2B 工具型 AI</span> | 與 Bedrock 同屬 Amazon 旗下企業級 AI 工具 |
| **Zapier** | **Zapier** | SaaS (Software as a Service) 自動化平台 | No-Code 跨平台工作流程自動化串接工具 | 收到 Gmail 附件自動存 Drive 並發 Slack 通知 | 擅長<span style="color: #7aa2f7">傳通知/存檔/同步表格</span>；<span style="color: #e06c75">無法完整理解語意</span> | 問 Zapier 無法完成何事，必選「<span style="color: #e06c75">完整理解語意</span>」 |
| **Notion / Notion AI** | **Notion Labs** | SaaS 協作軟體 | 多合一筆記、知識庫與文書整理軟體 | 團隊撰寫專案文件、會議紀錄整理擴寫 | 筆記知識庫工具，<span style="color: #e06c75">非生圖或 IDE 代碼生成工具</span> | 常作為生圖 (DALL·E) 或寫碼 (Copilot) 的<span style="color: #e06c75">干擾項</span> |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 5.6 資料科學與開源社群平台 (Platforms and Communities)

| 工具 / 模型 | 開發商 | 授權 / 架構 | 核心定位與受眾 | 💡 具體應用案例 | 🔥 考試常考重點 | ⚠️ 易錯陷阱提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Kaggle** | **Google**<br>(Alphabet) | 免費雲端社群 | 全球最大資料科學競賽與動手實作平台 | 在免費雲端 Notebook 載入公開數據訓練模型 | <span style="color: #7aa2f7">最適合學習 AI 工具實作之平台</span>（競賽/學方案/論壇） | 勿選 Shorts/Reels/LINE；<span style="color: #e06c75">無「免費訂閱優惠會員」</span> |
| **Hugging Face** | **Hugging Face Inc.** | 開源 AI 社群 | 全球最大開源模型與資料集共享樞紐 (AI 界 GitHub) | 下載開源模型權重微調，遵循 CC (Creative Commons) 授權商用 | 與 Kaggle 並列<span style="color: #7aa2f7">模型與數據共用</span>代表平台 | <span style="color: #e06c75">非音樂創作工具</span>（歌曲題勿選） |
| **Amazon Bedrock** | **Amazon (AWS)** | 商業雲端平臺<br>(AWS 託管服務) | 企業級 B2B 雲端基礎模型託管與安全部署平臺 | 企業透過單一 API 調用 Claude / LLaMA 等模型並結合 AWS 部署 | 主打「<span style="color: #7aa2f7">雲端整合、B2B 工具型 AI、企業級部署</span>」 | 屬企業雲端 AI 部署平臺，非單一開源社群 |
| **H100 / B100 ＆ NIMs** | **NVIDIA** | 硬體晶片與推論微服務<br>(NVIDIA Inference Microservices) | 提供全球 LLM 訓練與推論部署之核心算力基礎設施 | 資料中心建置千萬級參數大模型訓練叢集與高速推論服務 | 各家 AI 模型的「<span style="color: #7aa2f7">GPU 心臟</span>」；提供訓練與部署核心基礎 | 屬底層算力硬體與推論平臺，非終端聊天軟體 |
| **LIME (Local Interpretable Model-agnostic Explanations)** | **開源演算法** | 開源 Python 庫<br>(局部代理模型) | 模型可解釋性 (XAI, Explainable AI) 工具，專注**單一局部解釋** | 解釋特定客戶「張三」為何被銀行拒貸 | 著重<span style="color: #7aa2f7">單一預測局部解釋</span> (Local) | 管「局部單筆」，<span style="color: #e06c75">勿與全局貢獻的 SHAP 記反</span> |
| **SHAP (SHapley Additive exPlanations)** | **開源演算法** | 開源 Python 庫<br>(Shapley 賽局值) | 模型可解釋性 (XAI, Explainable AI) 工具，專注**整體特徵貢獻** | 計算全體核貸中收入、年齡、負債各佔多少權重 | 著重<span style="color: #7aa2f7">整體特徵貢獻分配</span>（基於賽局理論） | 管「整體特徵貢獻」，<span style="color: #e06c75">勿與 LIME 記反</span> |
| **Hadoop / Spark** | **Apache 基金會** | 開源分散式框架 | 海量大數據分散式儲存與平行運算框架 | 百億筆電商交易日誌執行 ETL (Extract, Transform, Load) 平行清洗轉換 | <span style="color: #7aa2f7">ETL 流程</span>常搭配之<span style="color: #7aa2f7">大量平行運算</span>平台 | 屬後端大數據基礎設施，非終端生成軟體 |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>

---

### 5.7 開發商 vs 代表工具速查矩陣

| 開發商 / 組織 | 旗下代表性模型與工具 | 關鍵考點 1 行秒殺口訣 |
| :--- | :--- | :--- |
| **OpenAI** | **ChatGPT**, **GPTs**, **GPT-1/3/4V/4o**, **CLIP & DALL·E 3**, **Sora** | ChatGPT (<span style="color: #7aa2f7">GPT-3.5 + RLHF 對齊</span>)、GPTs (<span style="color: #7aa2f7">平臺化生態系</span>)、GPT-3 (<span style="color: #e06c75">純文字不生圖</span>)、DALL·E (<span style="color: #7aa2f7">畫圖</span><span style="color: #e06c75">非影片</span>)、Sora (<span style="color: #7aa2f7">生影片：Diffusion + Transformer</span>) |
| **Google / DeepMind** | **Gemini 1.5 (Pro/Flash，原 Bard)**, **BERT**, **Lumiere / Veo**, **Kaggle**, **Assistant / 翻譯 / 字幕** | Gemini 1.5 (<span style="color: #7aa2f7">搜尋+手機 Pixel/Android 整合+長影音文件解析</span>)、BERT (<span style="color: #e06c75">鑑別式</span><span style="color: #7aa2f7">雙向理解</span>)、Lumiere (<span style="color: #7aa2f7">Space-Time U-Net</span>)、Kaggle (<span style="color: #7aa2f7">實作平台</span>) |
| **Anthropic** | **Claude 3 系列 (Opus / Sonnet / Haiku)** | <span style="color: #7aa2f7">Constitutional AI</span> 符合人類價值觀、高安全性、支援 <span style="color: #7aa2f7">200K tokens 超長上下文</span> |
| **Meta** | **Meta AI**, **LLaMA 3 (Large Language Model Meta AI)** | 主打「<span style="color: #7aa2f7">開源推廣</span>」標竿、社群導向整合（Meta App 內建 AI 助手） |
| **xAI (Elon Musk)** | **Grok** | 結合 **X（原 Twitter）社群**、強調<span style="color: #7aa2f7">言論自由與即時資訊</span> |
| **Amazon (AWS)** | **Bedrock**, **CodeWhisperer** | 雲端整合、<span style="color: #7aa2f7">B2B 工具型 AI、企業級部署</span>（與 AWS 深度結合） |
| **Apple** | **Apple Intelligence**, **Siri** | Apple Intelligence (<span style="color: #7aa2f7">Foundation Models 框架、離線運算、隱私保護</span>)；Siri (早期基於 <span style="color: #7aa2f7">RNN</span> 語音辨識) |
| **NVIDIA** | **H100 / B100 GPU**, **NIMs 推論平臺** | 提供 LLM 訓練與部署核心基礎，為各家 AI 模型的「<span style="color: #7aa2f7">GPU 心臟</span>」 |
| **Stability AI** | **Stable Diffusion**, **DreamStudio** | Stable Diffusion (<span style="color: #7aa2f7">開源免費可本地部署</span>)、DreamStudio (<span style="color: #7aa2f7">官方入門介面</span>) |
| **Runway ML** | **Runway (Gen-1 ~ Gen-4)** | <span style="color: #7aa2f7">改良版 Diffusion</span> 影片生成，專門協助「<span style="color: #7aa2f7">非專業創作者</span>」 |
| **Adobe** | **Adobe Firefly** | 整合 Photoshop (PS)，主打「<span style="color: #7aa2f7">合法授權素材訓練、商用安全無侵權</span>」 |
| **Microsoft / GitHub** | **GitHub Copilot** | IDE (Integrated Development Environment) 內建助手，網頁開發生成 <span style="color: #7aa2f7">HTML 與 CSS 結構</span> |
| **AIVA Technologies** | **AIVA (Artificial Intelligence Virtual Artist)** | AI 作曲編曲，<span style="color: #7aa2f7">常與 GPT 結合創作歌曲</span> |
| **Pika Labs** | **Pika** | <span style="color: #7aa2f7">動畫導向 Diffusion + 多模態控制</span>，原創角色動畫與社群短片 |
| **PearlMountain** | **FlexClip** | 線上 <span style="color: #7aa2f7">AI 影片生成與剪輯</span>平台（文字轉短片） |
| **Leonardo.Ai** | **Leonardo AI** | 微調 SD (Stable Diffusion)，擅長遊戲美術與<span style="color: #7aa2f7">概念圖/幻想風格</span> |
| **臺灣國科會 (NSTC)** | **TAIDE (Trustworthy AI Dialog Engine)** | 臺灣繁中本土大模型，核心技術為<span style="color: #7aa2f7">機器學習 (ML, Machine Learning)</span> |
| **Zapier** | **Zapier** | No-Code 自動化串接，致命限制為「<span style="color: #e06c75">無法完整理解語意</span>」 |
| **Notion Labs** | **Notion / Notion AI** | 筆記知識庫軟體，<span style="color: #e06c75">非生圖或 IDE 寫碼工具</span> |
| **Hugging Face Inc.** | **Hugging Face** | 全球<span style="color: #7aa2f7">開源模型與資料集共用</span>社群樞紐 |
| **Apache 基金會** | **Hadoop, Spark** | 大數據平行運算，<span style="color: #7aa2f7">ETL (Extract, Transform, Load) 資料處理</span>經典搭配平台 |

<div align="right"><sub><a href="#top" style="text-decoration: none; color: #8b949e;">↑ Top</a></sub></div>
