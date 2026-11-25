"""一个 episode 的状态：预算、已展示商品、状态码（DESIGN.md §3；specs/training.md §2）。

状态流转：

- ``submit`` 成功（参数格式正确）→ ``ok``
- 参数格式错误、或驱动方报告模型没有给出合法的工具调用 → ``format_error``
- search / view 用满 ``max_tool_calls`` 次后，环境提示“必须提交”，再给一轮机会；
  这一轮里仍调用 search / view，或没有提交 → ``budget_exhausted``
- 驱动方报告整条上下文超限 → ``truncated``

终态一旦确定就不再改变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from searchtune.config import EnvConfig
from searchtune.data.pool import Catalog, Pool
from searchtune.env import tools
from searchtune.env.pool_index import PoolIndex
from searchtune.eval.metrics import normalize_submission


@dataclass
class EpisodeResult:
    submitted: list[str]
    status: str  # ok | format_error | budget_exhausted | truncated；未结束时为 "running"
    n_search: int
    n_view: int
    n_dropped_ids: int = 0  # submit 时被丢弃的重复或池外 id 数
    searched_queries: list[str] = field(default_factory=list)
    shown_ids: list[str] = field(default_factory=list)  # search 结果里出现过的商品（按首次出现）
    viewed_ids: list[str] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return self.n_search + self.n_view


class Episode:
    def __init__(self, pool: Pool, catalog: Catalog, env_cfg: EnvConfig | None = None):
        self._pool = pool  # 持有标签，但不通过任何方法暴露给 Agent
        self._catalog = catalog
        self.cfg = env_cfg or EnvConfig()
        self._index = PoolIndex(pool, catalog)
        self._n_search = 0
        self._n_view = 0
        self._submitted: list[str] = []
        self._dropped = 0
        self._status = "running"
        self._queries: list[str] = []
        self._shown: list[str] = []
        self._viewed: list[str] = []
        self._last_results: list[str] = []
        self._must_submit = False

    # ---- 状态 -------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self._status != "running"

    @property
    def must_submit(self) -> bool:
        return self._must_submit and not self.done

    @property
    def last_results(self) -> list[str]:
        """最近一次 search 返回的 product_id（供 B0、L1 等非 LLM 驱动使用，不是给 Agent 的接口）。"""
        return list(self._last_results)

    @property
    def result(self) -> EpisodeResult:
        return EpisodeResult(
            submitted=list(self._submitted),
            status=self._status,
            n_search=self._n_search,
            n_view=self._n_view,
            n_dropped_ids=self._dropped,
            searched_queries=list(self._queries),
            shown_ids=list(self._shown),
            viewed_ids=list(self._viewed),
        )

    def _finish(self, status: str) -> None:
        if not self.done:
            self._status = status

    # ---- 驱动方通知 -------------------------------------------------------
    def mark_format_error(self) -> None:
        """模型没有给出可解析的工具调用，或工具参数不合法。"""
        self._finish("format_error")

    def mark_truncated(self) -> None:
        """整条上下文超过 ``max_context_tokens``。"""
        self._finish("truncated")

    def end_without_submit(self) -> None:
        """模型停止调用工具但没有 submit。预算已用尽则记预算耗尽，否则记格式错误。"""
        self._finish("budget_exhausted" if self._must_submit else "format_error")

    # ---- 预算 -------------------------------------------------------------
    def _gate(self) -> str | None:
        """search / view 的前置检查；返回非 None 表示该调用被拒绝。"""
        if self.done:
            return tools.FINISHED
        if self._n_search + self._n_view >= self.cfg.max_tool_calls:
            self._finish("budget_exhausted")
            return tools.OVER_BUDGET
        return None

    def _after_call(self, text: str) -> str:
        if self._n_search + self._n_view >= self.cfg.max_tool_calls:
            self._must_submit = True
            return f"{text}\n{tools.BUDGET_NOTICE}"
        return text

    # ---- 工具 -------------------------------------------------------------
    def search(self, query: str) -> str:
        blocked = self._gate()
        if blocked:
            return blocked
        self._n_search += 1
        if not isinstance(query, str) or not query.strip():
            self._last_results = []
            return self._after_call(tools.EMPTY_QUERY)
        self._queries.append(query)
        ids = self._index.search(query, self.cfg.search_top_k)
        self._last_results = ids
        for pid in ids:
            if pid not in self._shown:
                self._shown.append(pid)
        text = tools.format_search([self._catalog.get(p) for p in ids], self.cfg)
        return self._after_call(text)

    def view(self, product_id: str) -> str:
        blocked = self._gate()
        if blocked:
            return blocked
        self._n_view += 1
        if not isinstance(product_id, str) or product_id not in self._pool:
            return self._after_call(tools.UNKNOWN_PRODUCT)
        if product_id not in self._viewed:
            self._viewed.append(product_id)
        return self._after_call(tools.format_view(self._catalog.get(product_id), self.cfg))

    def submit(self, product_ids: list[str]) -> None:
        if self.done:
            return
        if not isinstance(product_ids, list) or not all(isinstance(p, str) for p in product_ids):
            self.mark_format_error()
            return
        self._submitted, self._dropped = normalize_submission(product_ids, self._pool, self.cfg.submit_top_k)
        self._finish("ok")
