"""RAG 评测脚本：把"感觉变好了"变成"命中率从 62% 涨到 88%"。

三种指标，分层定位问题：
1. 检索命中率 —— 期望来源是否出现在 top-k 里（低 → 检索/切块的问题）；
2. 关键词覆盖率 —— 期望的关键信息是否出现在检索文本/答案里（低 → 生成的问题）；
3. 拒答正确率 —— 该说"不知道"的时候有没有瞎编（低 → 幻觉的问题）。

用法：
    python eval/run_eval.py                  # 只测检索（便宜，只有嵌入调用）
    python eval/run_eval.py --k 2            # 指定检索条数（语料少时建议 2）
    python eval/run_eval.py --mode full      # 端到端，每题一次生成，会产生费用
    python eval/run_eval.py --mode full --limit 3   # 先拿 3 道题试水
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config_data as config  # noqa: E402
from providers import get_embeddings  # noqa: E402

REFUSAL_HINTS = ("未找到", "没有找到", "未提及", "不包含", "未提供", "无法回答", "不清楚", "没有相关")


def _norm(text: str) -> str:
    """归一化：全角转半角 + 去掉所有空白。

    这样 `7 天` / `7天` / `7　天` 会被视为同一个关键点，
    避免"答案正确却被判错"——评测脚本的口径错误比系统错误更隐蔽。
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text)


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_retriever(k: int):
    from vector_stores import VectorStoreService

    service = VectorStoreService(get_embeddings())
    return service.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 4, 10), "lambda_mult": config.mmr_lambda},
    )


def keyword_groups(keywords: list[str]) -> list[str]:
    """把同义写法合并成一个关键点（去重后每个元素就是一个必须命中的点）。"""
    groups: list[str] = []
    for kw in keywords:
        normalized = _norm(kw)
        if normalized and normalized not in groups:
            groups.append(normalized)
    return groups


def keyword_hits(text: str, keywords: list[str]) -> tuple[int, int]:
    groups = keyword_groups(keywords)
    if not groups:
        return 0, 0
    normalized_text = _norm(text)
    return sum(1 for g in groups if g in normalized_text), len(groups)


def run_retrieval(cases: list[dict], k: int) -> None:
    retriever = build_retriever(k)
    hits = eligible = 0
    kw_total = kw_hit = 0

    print(f"{'题号':<6}{'类型':<12}{'检索命中':<10}{'关键词':<10}备注")
    print("-" * 70)

    for case in cases:
        if case["type"] == "multi_turn":
            print(f"{case['id']:<6}{case['type']:<12}{'—':<10}{'—':<10}需 --mode full（依赖问题改写）")
            continue

        docs = retriever.invoke(case["question"])
        sources = {(d.metadata or {}).get("source") for d in docs}
        text = "\n".join(d.page_content for d in docs)

        hit = "—"
        if case["expected_source"]:
            eligible += 1
            if case["expected_source"] in sources:
                hits += 1
                hit = "✅"
            else:
                hit = "❌"

        got, total = keyword_hits(text, case["expected_keywords"])
        kw_hit += got
        kw_total += total
        kw_str = f"{got}/{total}" if total else "—"
        note = f"召回来源：{', '.join(sorted(s for s in sources if s))}"
        print(f"{case['id']:<6}{case['type']:<12}{hit:<10}{kw_str:<10}{note}")

    print("-" * 70)
    print(f"检索命中率：{hits}/{eligible} = {hits / eligible:.0%}" if eligible else "无检索类样本")
    print(f"关键词覆盖率：{kw_hit}/{kw_total} = {kw_ratio(kw_hit, kw_total)}")


def kw_ratio(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "—"


def run_full(cases: list[dict], k: int) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from RAG import RagService
    from file_history_store import get_history

    service = RagService()
    service.vector_service.vector_store  # 触发一次初始化，避免首题计时失真

    ok = refuse_ok = negative = 0
    kw_total = kw_hit = 0

    print(f"{'题号':<6}{'类型':<12}{'判定':<8}{'关键词':<8}耗时   回答摘要")
    print("-" * 90)

    for index, case in enumerate(cases):
        session_id = f"eval_{case['id']}_{index}"
        history = get_history(session_id)
        history.clear()
        for i, msg in enumerate(case.get("history") or []):
            history.add_messages([HumanMessage(content=msg) if i % 2 == 0 else AIMessage(content=msg)])

        started = time.time()
        try:
            answer = service.chain.invoke(
                {"input": case["question"]}, {"configurable": {"session_id": session_id}}
            )
        except Exception as exc:
            print(f"{case['id']:<6}{case['type']:<12}{'💥':<8}{'—':<8}0.0s  调用失败：{exc}")
            history.clear()
            continue
        elapsed = time.time() - started
        history.clear()

        got, total = keyword_hits(answer, case["expected_keywords"])

        if case["type"] == "refuse":
            negative += 1
            passed = any(hint in answer for hint in REFUSAL_HINTS)
            refuse_ok += int(passed)
            verdict = "✅ 拒答" if passed else "❌ 编造"
            kw_str = "—"
        else:
            kw_total += total
            kw_hit += got
            passed = (got == total) if total else True
            ok += int(passed)
            verdict = "✅" if passed else "❌"
            kw_str = f"{got}/{total}" if total else "—"

        summary = answer.replace("\n", " ")[:38]
        print(f"{case['id']:<6}{case['type']:<12}{verdict:<8}{kw_str:<8}{elapsed:.1f}s  {summary}")

    fact_total = len(cases) - negative
    print("-" * 90)
    print(f"事实题通过：{ok}/{fact_total} = {kw_ratio(ok, fact_total)}")
    print(f"拒答题通过：{refuse_ok}/{negative} = {kw_ratio(refuse_ok, negative)}")
    print(f"整体通过率：{ok + refuse_ok}/{len(cases)} = {kw_ratio(ok + refuse_ok, len(cases))}")
    print(f"关键点覆盖率：{kw_hit}/{kw_total} = {kw_ratio(kw_hit, kw_total)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--file", default=str(Path(__file__).parent / "questions.jsonl"))
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--k", type=int, default=2, help="检索条数，默认 2（语料少时更有区分度）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0 表示全部")
    args = parser.parse_args()

    cases = load_cases(Path(args.file))
    if args.limit:
        cases = cases[: args.limit]

    print(f"题目数：{len(cases)} | 模式：{args.mode} | 检索条数 k={args.k}\n")
    if args.mode == "retrieval":
        run_retrieval(cases, args.k)
    else:
        run_full(cases, args.k)


if __name__ == "__main__":
    main()
