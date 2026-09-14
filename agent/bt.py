"""A tiny hand-written behavior tree (BT) library.

Nothing here knows about Battlesnake. A behavior tree is a tree of nodes.
Each tick, we run the root, and every node reports SUCCESS or FAILURE to its
parent. Composite nodes use those results to decide which children to run:

- Sequence runs its children in order and fails as soon as one fails.
  Read it as "do this AND then this AND then this".
- Selector runs its children in order and succeeds as soon as one succeeds.
  Read it as "try this, OR ELSE this, OR ELSE this".

Leaves do the actual work:

- Condition asks a yes/no question and changes nothing.
- Action does something (usually writes to the blackboard) and reports
  whether it managed to.

The blackboard is whatever object the caller passes to tick(). It's how
nodes share information within one tick.

Every tick also fills in a trace: the names of the nodes that decided the
result, from the root down. For a Sequence that's the child that failed or,
if all succeeded, the last one; for a Selector it's the child that succeeded
or, if all failed, the last one. That shows *why* the tree did what it did.

Full BT libraries also have a RUNNING status, for actions that take several
ticks to finish. A Battlesnake turn is always one fresh tick, so we leave it
out. Nodes keep no state between ticks, so one tree can serve many requests
at once.
"""

from enum import Enum
from typing import Any, Callable


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class Node:
    def __init__(self, name: str):
        self.name = name

    def tick(self, blackboard: Any, trace: list[str]) -> Status:
        """Run this node. Append the deciding path to `trace`."""
        raise NotImplementedError


class Sequence(Node):
    """Run children in order; fail at the first failure, else succeed."""

    def __init__(self, name: str, children: list[Node]):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard: Any, trace: list[str]) -> Status:
        child_trace: list[str] = []
        for child in self.children:
            child_trace = []
            if child.tick(blackboard, child_trace) is Status.FAILURE:
                trace += [self.name, *child_trace]
                return Status.FAILURE
        trace += [self.name, *child_trace]
        return Status.SUCCESS


class Selector(Node):
    """Run children in order; succeed at the first success, else fail."""

    def __init__(self, name: str, children: list[Node]):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard: Any, trace: list[str]) -> Status:
        child_trace: list[str] = []
        for child in self.children:
            child_trace = []
            if child.tick(blackboard, child_trace) is Status.SUCCESS:
                trace += [self.name, *child_trace]
                return Status.SUCCESS
        trace += [self.name, *child_trace]
        return Status.FAILURE


class Condition(Node):
    """A yes/no question about the blackboard. Must not change it."""

    def __init__(self, name: str, check: Callable[[Any], bool]):
        super().__init__(name)
        self.check = check

    def tick(self, blackboard: Any, trace: list[str]) -> Status:
        trace.append(self.name)
        return Status.SUCCESS if self.check(blackboard) else Status.FAILURE


class Action(Node):
    """Do something with the blackboard; return True if it worked."""

    def __init__(self, name: str, act: Callable[[Any], bool]):
        super().__init__(name)
        self.act = act

    def tick(self, blackboard: Any, trace: list[str]) -> Status:
        trace.append(self.name)
        return Status.SUCCESS if self.act(blackboard) else Status.FAILURE
