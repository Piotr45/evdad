from pathlib import Path
from typing import Union

PathStr = Union[str, Path]


# class PathStr:
#     def __init__(self, value: Union[str, Path]):
#         if not isinstance(value, (str, Path)):
#             raise TypeError("PathStr must be initialized with a string or a Path object.")
#         self._value = Path(value) if isinstance(value, str) else value
#
#     def __str__(self):
#         return str(self._value)
#
#     def __repr__(self):
#         return f"PathStr({repr(self._value)})"
#
#     def __eq__(self, other):
#         if isinstance(other, (PathStr, Path, str)):
#             return self._value == Path(other)
#         return False
#
#     def as_path(self) -> Path:
#         """Returns the value as a Path object."""
#         return self._value
#
#     def as_str(self) -> str:
#         """Returns the value as a string."""
#         return str(self._value)
#
#     def __fspath__(self):
#         """Allows the object to be used in file system operations."""
#         return str(self._value)
