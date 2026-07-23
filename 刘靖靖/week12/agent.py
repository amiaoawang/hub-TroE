"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --question "..." --max_steps 8
  python agent.py --interactive                  # 多轮对话 REPL，保留上下文

环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen-max，可换 deepseek-v3 等
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

# 多轮 REPL 的退出指令
EXIT_COMMANDS = {"exit", "quit", "q", "退出"}
RESET_COMMANDS = {"reset", "clear", "新对话", "重新开始"}


def run_interactive(mode: str, max_steps: int):
    """
    多轮对话 REPL：在终端里持续提问，跨轮次保留上下文。

    每轮结束后，把 (用户问题, Final Answer) 追加到 history，
    下一轮注入到 ReAct 的 system 与 user 之间。
    输入 reset/clear/新对话 清空历史开始新会话；输入 exit/quit 退出。
    """
    if mode == "manual":
        from react_manual import run
    else:
        from react_function_calling import run

    print("\n" + "=" * 60)
    print(f"多轮对话模式  |  实现: {mode}  |  模型见各模块日志")
    print(f"输入 {sorted(EXIT_COMMANDS)} 退出；输入 {sorted(RESET_COMMANDS)} 开始新对话")
    print("=" * 60)

    history: list = []
    turn = 0

    while True:
        try:
            question = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("再见 👋")
            return
        if question.lower() in RESET_COMMANDS:
            history = []
            turn = 0
            print("🔄 已清空对话历史，开始新会话")
            continue

        turn += 1
        print(f"\n─── 第 {turn} 轮（历史 {len(history) // 2} 轮）───")

        final_answer = ""
        for step_data in run(question, max_steps=max_steps, history=history):
            stype = step_data["type"]
            if stype == "final":
                final_answer = step_data.get("answer", "")
            elif stype in ("error", "max_steps"):
                final_answer = step_data.get("answer", "")

        # 写回历史，供下一轮使用
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": final_answer})
        # 简单裁剪，保留最近 10 条消息
        if len(history) > 10:
            history = history[-10:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="进入多轮对话 REPL，跨轮次保留上下文",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.mode, args.max_steps)
    else:
        if args.mode == "manual":
            from react_manual import run_and_print
        else:
            from react_function_calling import run_and_print
        run_and_print(args.question, args.max_steps)
