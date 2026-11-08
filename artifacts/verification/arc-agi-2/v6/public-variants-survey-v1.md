# 公開 NVARC 變體對照（2026-10-04，子代理整理，來源見 runs/arc-agi-2/v6/external-src/）

- 跑次變異：同一份 stock NVARC 重跑五次得分 29.7／31.8／26.9／32.2／31.4（luxluxshan 自述），單次 LB 差異 ±3pp 內不足以證明優劣。
- 時間：沒有任何證據顯示全部任務能在 12h 內完成；Program 078 在約 9h04 後有 15 個空輸出、17 個單一候選；固定 cap 800 s 時 24 題中 2 題未到達。
- 數學等價：failed-in-aimo（33.89 母版）、original-kg、perfpatch（mikelou1，自稱母版 33.89，他人回報其本身 31.81）。

## M3 單一改動候選（依優先序）
1. perfpatch（GPU 端 logsumexp／gather，與作者版數學等價；可多完成任務）。
2. cheap-first 排序（依估計工作量由小到大，讓 deadline 截掉最貴的尾端）。
3. 自適應每題上限 max(floor, min(cap, 剩餘時間×4／待處理數))。
4. 缺漏／單一候選補跑：只補 attempt_2，Primary top1 不變（單調不降分）。
5. 半成本＋第二 seed 池化（luxluxshan，引用 LB 35.97，第三手說法）。
- 其他：固定 PYTHONHASHSEED／crc32 seed，讓 A/B 可比較；TRM 混合上限最高，但需要另外掛載資料與 GPU。
