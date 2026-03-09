from __future__ import annotations

from typing import List, Optional, Tuple

from .node import Node


class NodeScheduler:
    # Greedy: prefer an idle model group whose size exactly matches required parallelism
    def select_nodes(self, nodes: List[Node], required: int) -> Optional[List[int]]:
        if required <= 0:
            return None

        # Build groups by model_id among idle nodes
        group_to_indices: dict[int, List[int]] = {}
        for node in nodes:
            if node.is_idle():
                group_to_indices.setdefault(node.model_id, []).append(node.index)

        # Prefer group size == required and model_id != 0
        for model_id, indices in group_to_indices.items():
            if model_id != 0 and len(indices) == required:
                return indices.copy()

        # Otherwise: prefer idle nodes with model_id == 0 first, then other idle nodes by index
        idle_zero = sorted([n.index for n in nodes if n.is_idle() and n.model_id == 0])
        idle_nonzero = sorted([n.index for n in nodes if n.is_idle() and n.model_id != 0])
        chosen = idle_zero[:required]
        if len(chosen) < required:
            needed = required - len(chosen)
            chosen.extend(idle_nonzero[:needed])
        if len(chosen) != required:
            print(f"Node scheduling failed despite legality check, chosen={chosen}, required={required}")
        return chosen if len(chosen) == required else None


