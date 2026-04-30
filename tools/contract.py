"""
contract.py
-----------
ToolContract Protocol 

"""

from __future__ import annotations

from typing import Awaitable, Type
from pydantic import BaseModel
from typing import runtime_checkable, Protocol


@runtime_checkable
class ToolContract(Protocol):
    input_schema:  Type[BaseModel]
    output_schema: Type[BaseModel]

    @classmethod
    def schema(cls) -> dict: ...

    def execute(self, inputs: BaseModel) -> Awaitable[BaseModel]: ...
