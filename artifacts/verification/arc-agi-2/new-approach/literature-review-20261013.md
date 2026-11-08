# ARC-AGI-2 文獻與研究成果回顧（2026-10-13，使用者要求「找有沒有直接或較大幫助的」）

兩個研究子代理（ARC 方法、L4 推論加速）＋本對話抽查來源；與 Claude Science CSR-arc-agi-2-001 互相對照。

## 一句話
- **沒有任何公開、已驗證、能在 4×L4／12 h 跑、ARC-AGI-2 上大幅勝過 NVARC 的方法**；2026 前段隊伍（70–88%）的做法都沒公開。
- 對我們真正有直接幫助的是三類：
  1. 推論配置（瓶頸不是頻寬）；
  2. 程式合成通道的「取樣＋修正」配方與選法；
  3. 一個現成的 ARC 程式合成小模型（SOAR）。

## 直接或較大幫助（依預期效益排序）
1. **推論配置（已在跑 P1 掃描，Kaggle `arc-2026-p1-qwen38-sweep-v1`）**
   - 依據：P0 每串流 17.4 tok/s，MTP 每步約 2.3 token，換算約 133 ms／步。權重頻寬下限只有約 25–40 ms，所以瓶頸是每步固定開銷，例如排程、TP=4 的 PCIe all-reduce、kernel。
   - 我們沿用公開 notebook 的 `--no-async-scheduling`，把非同步排程關掉了。
   - 可試的項目（推論，未量）：
     - 開回非同步排程；
     - 併發 16 → 32：KV 最高只用 67%；
     - TP2×2 副本：少一半 all-reduce；
     - MTP 3。
   - 預期合計 1.5–2.5×。
   - 來源：HF incoai/Qwen3.8-27B-DFlash2 卡上 H200 的併發數據，以及 P0 實測。
2. **程式合成通道用「取樣＋修正」，並以 SOAR 的票數規則選前 2**
   - 依據：SOAR（arXiv 2507.14172，github flowersteam/SOAR）在 ARC-AGI-1 上：
     - Sample&Refine 勝過純取樣，例如 Qwen2.5-Coder-14B 29.67 → 34.83%；
     - 兩輪測試時迭代約 +5%；
     - 票數規則：count + 1000×train_acc。
   - 我們的 P0b 已是「取樣 2＋修正 1」，選法要改成這個規則。
   - 注意：沒有 ARC-AGI-2 數字。
3. **SOAR 微調模型當快速程式撰寫器**
   - 模型：julien31/Soar-qwen-7b／14b／32b，**Apache-2.0**，Qwen2.5-Coder 基底。已查 HF：7b 有 7.6B 參數。
   - 優點：
     - 非推理型 coder，每個程式只要數百到數千 token，比 Qwen3.8 xhigh 每次 6.5 萬少一兩個數量級。
     - 7B 可在單張 L4 跑（不需 TP，沒有 all-reduce 開銷），能每題取樣幾十到上百個程式。
   - ARC-AGI-1：五模型集成 52%。ARC-AGI-2 未知，可能很低。
   - 試法：Colab 上用 keys30 量驗證通過率與精確度，再與 NVARC 做聯集。
4. **轉導端的過濾與評分（小，未必可轉移）**
   - 來源：arXiv 2603.06590（LongT5 200M＋TTT，Kaggle 27.08%）。
   - 內部集消融：符號過濾（輸出尺寸／顏色合理性）−14，traversal 增強 −6，對稱性評分 −4。都不是 ARC-AGI-2 數字，也沒有程式碼。
   - 可以試一個「不合理候選過濾器」接在 NVARC 後面，成本小。
5. **NVARC 端的推論加速**
   - 來源：ARChitects 2025 writeup。投機解碼讓搜尋快約 4.7×；前綴快取讓評分快約 5.8×，因此可把評分增強從 8 提高到 32。
   - 要先確認我們的 NVARC 流程是否已有這些。準確率增益不明，可能小。

## 沒用或有風險
- TinyLM（2605.01154）：21.7% 是不加 TTT 的版本；TTT／PoE 都是 0%。無可借用。
- ARCANA（2607.09059）：宣稱 32.5%，無程式碼，內容多處矛盾。**不可據以規劃**。
- NVIDIA Nemotron-SFT／RL-ARC-AGI-v1：資料集存在。`python_inductive` 子集正是「寫 transform」格式，可當提示與評測參考。但授權標「pending-legal-review」、「research and development only」，**不拿來訓練**，否則會影響開源得獎資格。
- 需要 H100 預訓練的方法（ARChitects 擴散、MindsAI 預訓練）、閉源 API 方法（Pang、Berman、Poetiq、Land）：不適用。
- DFlash2 草稿模型：H200 上比 MTP 快約 1.3–1.4×，但 vLLM 支援還在未合併的 PR，離線 wheel 裝不了。暫不碰。
- 反例：有團隊的 DSL 搜尋中，42 題裡有 3 題「通過所有範例但測試答錯」→ 只靠範例驗證的程式，精確度有上限。P0b 會量這個。

## 對計畫的影響
- P1（Kaggle，進行中）量配置加速 → 若達 2× 以上，Qwen3.8 程式通道的時間預算會寬很多。
- P0b（Colab G4，進行中）量程式模式的 token 與精確度。
- 下一個候選實驗：**SOAR-7B／14B 程式撰寫器**（Colab，小時級），讓程式通道不依賴 27B 推理模型。
- 選法採 SOAR 票數規則：通過全部範例的程式 → attempt_1，NVARC 第一名 → attempt_2。
