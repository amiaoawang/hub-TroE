"""评估模块：LLM-as-judge 三维度评分。

维度：
  - faithfulness       回答是否忠于资料、有无编造
  - relevance          回答是否切题、完整
  - context_precision  检索到的资料是否对问题有用

边界处理：
- LLM 返回非 JSON → safe_json_loads 兜底
- 评分解析失败 → 用 -1 标记，不影响整体流程
- 评估集为空 → 平均分返回 0
"""
from __future__ import annotations

from typing import Any, Dict, List

from config import LLMConfig
from generator import Generator
from utils import get_logger, safe_json_loads, truncate

logger = get_logger()


# 三个维度的 prompt 模板
EVAL_FAITHFULNESS = """请判断回答是否完全基于参考资料，是否存在编造或与资料冲突。

【参考资料】
{context}

【问题】
{question}

【回答】
{answer}

只输出 JSON，格式：{{"score": 0到10的整数, "reason": "30字以内说明"}}
"""

EVAL_RELEVANCE = """请判断回答是否切题、是否完整回答了用户问题。

【问题】
{question}

【回答】
{answer}

只输出 JSON，格式：{{"score": 0到10的整数, "reason": "30字以内说明"}}
"""

EVAL_CONTEXT = """请判断检索到的资料是否对回答问题有用。

【问题】
{question}

【资料】
{context}

只输出 JSON，格式：{{"score": 0到10的整数, "reason": "30字以内说明"}}
"""


class Evaluator:
    """LLM-as-judge 评估器。"""

    def __init__(self, cfg: LLMConfig):
        # 评估直接复用生成器，但强制 json 输出
        self.generator = Generator(cfg)
        self.cfg = cfg

    def _judge(self, prompt: str) -> Dict[str, Any]:
        try:
            # 走生成器内部，但要求 json_object
            from openai import OpenAI
            client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key,
                            timeout=self.cfg.timeout)
            resp = client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"评估 LLM 调用失败: {e}")
            return {"score": -1, "reason": f"调用失败: {e}"}

        data = safe_json_loads(text, default=None)
        if not isinstance(data, dict):
            return {"score": -1, "reason": "解析失败"}
        score = data.get("score", -1)
        try:
            score = int(score)
            if score < 0 or score > 10:
                score = -1
        except (TypeError, ValueError):
            score = -1
        return {"score": score, "reason": str(data.get("reason", ""))}

    def evaluate(self, question: str, contexts: List[Dict], answer: str) -> Dict[str, Any]:
        """对单条问答打三个维度的分。"""
        context_str = "\n\n".join(truncate(c["text"], 1000) for c in contexts) or "(无资料)"

        faith = self._judge(EVAL_FAITHFULNESS.format(
            context=context_str, question=question, answer=answer))
        rel = self._judge(EVAL_RELEVANCE.format(question=question, answer=answer))
        ctx = self._judge(EVAL_CONTEXT.format(question=question, context=context_str))

        return {
            "faithfulness": faith["score"],
            "faithfulness_reason": faith["reason"],
            "relevance": rel["score"],
            "relevance_reason": rel["reason"],
            "context_precision": ctx["score"],
            "context_precision_reason": ctx["reason"],
            "retrieval_distances": [round(float(c["score"]), 4) for c in contexts],
            "retrieved_pages": [c["page"] for c in contexts],
        }


def avg_score(results: List[Dict], key: str) -> float:
    """计算某个维度的平均分，过滤掉 -1（解析失败）。"""
    valid = [r["eval"][key] for r in results
             if isinstance(r.get("eval", {}), dict) and r["eval"].get(key, -1) >= 0]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def summarize(results: List[Dict]) -> Dict[str, float]:
    """汇总平均分。"""
    if not results:
        return {"faithfulness": 0.0, "relevance": 0.0, "context_precision": 0.0}
    return {
        "faithfulness": round(avg_score(results, "faithfulness"), 2),
        "relevance": round(avg_score(results, "relevance"), 2),
        "context_precision": round(avg_score(results, "context_precision"), 2),
    }
