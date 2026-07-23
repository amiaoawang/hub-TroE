核心设计
历史只保留 Q&A，不保留 ReAct 中间步骤 —— 每轮结束后只把 (用户问题, Final Answer) 写入历史，下一轮注入到 system 与本轮 user 之间。Thought/Action/Observation 不进历史，避免上下文膨胀和模型被中间格式干扰。

改动清单
react_manual.py — run() 增加 history: list | None = None 参数，拼接到 system 之后；System Prompt 增加多轮指代消解规则。

react_function_calling.py — 同样的 history 注入逻辑。

serve.py — 新增内存会话存储 SESSIONS，QueryRequest 加 session_id（不传则服务端用 uuid 生成）；_stream_react 启动前读历史注入，结束后写回本轮 Q&A；新增 POST /reset 端点清空会话；历史裁剪到最近 10 条消息（约 5 轮），成对裁剪保证始终以 user 开头。

agent.py — 新增 --interactive / -i 多轮 REPL 模式，输入 reset/新对话 清空历史，exit/quit 退出。

index.html — 多轮 UI：每轮以 turn-block 分组（含 Turn 序号、问题、历史轮次标记），不再清空旧轮次；新增「✨ 新对话」按钮和会话指示器（显示历史轮数）；session_id 由服务端首次响应分配并在后续请求复用。

验证
所有 Python 文件 py_compile 通过
历史裁剪逻辑单测：7 轮后正确裁剪到 5 轮，丢弃最早 2 轮，角色始终 user/assistant 交替
evaluate.py 调用 react_run(question, max_steps=...) 不传 history，因默认 None 完全兼容
试用方式
Web：uvicorn serve:app，连续提问（如先问"茅台和五粮液2023年毛利率差多少"，再问"那它俩近一年股价涨跌幅谁更好"），观察第二轮如何引用第一轮结论
CLI：python agent.py --interactive -i
