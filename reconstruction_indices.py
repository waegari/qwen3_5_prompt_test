"""
tree/forest에 인덱스를 부여

- depth 1 (루트): "0", "1", "2", ...
- depth 2: "0_0", "0_1", "1_0", ...
- depth 3: "0_0_0", "0_1_1", ...
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

Node = dict[str, Any]
Forest = Sequence[Node]


def assign_indices(
    nodes: Optional[Forest],
    prefix: Optional[Sequence[int]] = None,
) -> None:
    """
    nodes: 상위 노드 리스트
    prefix: 상위 경로 (0-based 정수), None이면 루트 레벨
    """
    if not nodes:
        return

    base: List[int] = list(prefix) if prefix is not None else []

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        path = base + [i]
        node["index"] = "_".join(str(p) for p in path)

        sub = node.get("subitems")
        if sub:
            assign_indices(sub, path)


def assign_indices_from_reconstruction(
    data: Union[Forest, dict[str, Any]],
) -> None:
    if isinstance(data, dict):
        recon = data.get("reconstruction")
        if isinstance(recon, list):
            assign_indices(recon)
        return
    assign_indices(data)
