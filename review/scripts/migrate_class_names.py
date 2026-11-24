"""One-off migration: per-dtype class names -> structure classes + dtype (RESTRUCTURE 3.1 item 4).

Rewrites Python files in place using AST positions (formatting and comments survive):
  FloatMatrix(n)               -> DenseMatrix(n, dtype="float64")
  FloatMatrix(rows, cols)      -> DenseMatrix(rows, cols, dtype="float64")
  FloatMatrix(array_like)      -> matrix(array_like, dtype="float64")   (vectors: vector(...))
  isinstance(x, FloatMatrix)   -> (isinstance(x, DenseMatrix) and x.dtype == "float64")
  TriangularBitMatrix.random   -> TriangularMatrix.random
  from pycauset import Old     -> from pycauset import <structure classes used>
The prefix (pc., pycauset., or bare) is kept. Leftovers are reported for manual fixing.

Usage: python review/scripts/migrate_class_names.py FILE...
"""
from __future__ import annotations

import ast
import re
import sys

OLD = {
    "FloatMatrix": ("DenseMatrix", "float64"), "Float32Matrix": ("DenseMatrix", "float32"),
    "Float16Matrix": ("DenseMatrix", "float16"), "IntegerMatrix": ("DenseMatrix", "int32"),
    "Int8Matrix": ("DenseMatrix", "int8"), "Int16Matrix": ("DenseMatrix", "int16"),
    "Int64Matrix": ("DenseMatrix", "int64"), "UInt8Matrix": ("DenseMatrix", "uint8"),
    "UInt16Matrix": ("DenseMatrix", "uint16"), "UInt32Matrix": ("DenseMatrix", "uint32"),
    "UInt64Matrix": ("DenseMatrix", "uint64"), "ComplexFloat16Matrix": ("DenseMatrix", "complex_float16"),
    "ComplexFloat32Matrix": ("DenseMatrix", "complex_float32"), "ComplexFloat64Matrix": ("DenseMatrix", "complex_float64"),
    "DenseBitMatrix": ("DenseMatrix", "bool"), "TriangularBitMatrix": ("TriangularMatrix", "bool"),
    "TriangularFloatMatrix": ("TriangularMatrix", "float64"), "TriangularIntegerMatrix": ("TriangularMatrix", "int32"),
    "FloatVector": ("DenseVector", "float64"), "Float32Vector": ("DenseVector", "float32"),
    "Float16Vector": ("DenseVector", "float16"), "IntegerVector": ("DenseVector", "int32"),
    "Int8Vector": ("DenseVector", "int8"), "Int16Vector": ("DenseVector", "int16"),
    "Int64Vector": ("DenseVector", "int64"), "UInt8Vector": ("DenseVector", "uint8"),
    "UInt16Vector": ("DenseVector", "uint16"), "UInt32Vector": ("DenseVector", "uint32"),
    "UInt64Vector": ("DenseVector", "uint64"), "BitVector": ("DenseVector", "bool"),
    "ComplexFloat16Vector": ("DenseVector", "complex_float16"),
    "ComplexFloat32Vector": ("DenseVector", "complex_float32"),
    "ComplexFloat64Vector": ("DenseVector", "complex_float64"),
}
# Square-only structures take n; DiagonalMatrix/SymmetricMatrix/AntiSymmetricMatrix keep their
# names (default dtype float64 matches the old classes).
SIZE_NAMES = re.compile(r"^(n|N|m|M|k|size|rows|cols|dim|nn|n_\w+|\w*_n|\w*size\w*|\w*dim\w*|count|length)$")


def _name_of(func: ast.expr) -> tuple[str, str] | None:
    """(prefix, name) for Name / Attribute(pc|pycauset).Name."""
    if isinstance(func, ast.Name):
        return "", func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id + ".", func.attr
    return None


def _is_size_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.Name):
        return bool(SIZE_NAMES.match(node.id))
    if isinstance(node, ast.BinOp):
        return _is_size_expr(node.left) or _is_size_expr(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("int", "len"):
        return True
    if isinstance(node, ast.Attribute) and node.attr in ("n", "size", "rows", "cols"):
        return True
    return False


def migrate(src: str) -> tuple[str, list[str]]:
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    def pos(lineno: int, col: int) -> int:
        # col_offset is in UTF-8 bytes
        line = lines[lineno - 1]
        return offsets[lineno - 1] + len(line.encode()[:col].decode(errors="ignore"))

    edits: list[tuple[int, int, str]] = []
    leftovers: list[str] = []
    seg = lambda n: ast.get_source_segment(src, n)  # noqa: E731

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            nm = _name_of(node.func)
            # isinstance(x, Old) / isinstance(x, (Old, ...))
            if isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) == 2:
                target = node.args[1]
                elts = target.elts if isinstance(target, ast.Tuple) else [target]
                parts, touched = [], False
                for e in elts:
                    en = _name_of(e)
                    if en and en[1] in OLD:
                        cls, dt = OLD[en[1]]
                        parts.append(f'(isinstance({seg(node.args[0])}, {en[0]}{cls}) and {seg(node.args[0])}.dtype == "{dt}")')
                        touched = True
                    else:
                        parts.append(f"isinstance({seg(node.args[0])}, {seg(e)})")
                if touched:
                    edits.append((pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset),
                                  "(" + " or ".join(parts) + ")"))
                continue
            # assertIsInstance(x, Old)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "assertIsInstance" and len(node.args) >= 2:
                en = _name_of(node.args[1])
                if en and en[1] in OLD:
                    cls, dt = OLD[en[1]]
                    x = seg(node.args[0])
                    edits.append((pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset),
                                  f'{seg(node.func.value)}.assertTrue(isinstance({x}, {en[0]}{cls}) and {x}.dtype == "{dt}")'))
                continue
            if nm and nm[1] in OLD and not node.keywords:
                prefix, old = nm
                cls, dt = OLD[old]
                args = [seg(a) for a in node.args]
                if node.args and all(_is_size_expr(a) for a in node.args):
                    new = f'{prefix}{cls}({", ".join(args)}, dtype="{dt}")'
                elif len(node.args) == 1:
                    fn = "vector" if cls == "DenseVector" else "matrix"
                    pfx = prefix if prefix else "pycauset."
                    new = f'{pfx}{fn}({args[0]}, dtype="{dt}")'
                else:
                    leftovers.append(f"line {node.lineno}: {seg(node)}")
                    continue
                edits.append((pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset), new))
        elif isinstance(node, ast.Attribute) and node.attr in ("random",):
            en = _name_of(node.value)
            if en and en[1] in ("TriangularBitMatrix", "DenseBitMatrix"):
                cls = OLD[en[1]][0]
                edits.append((pos(node.value.lineno, node.value.col_offset), pos(node.value.end_lineno, node.value.end_col_offset),
                              f"{en[0]}{cls}"))

    # Apply non-overlapping edits from the end (outermost wins).
    edits.sort(key=lambda e: (e[0], -e[1]))
    kept: list[tuple[int, int, str]] = []
    for e in edits:
        if kept and e[0] < kept[-1][1]:
            continue
        kept.append(e)
    out = src
    for a, b, text in reversed(kept):
        out = out[:a] + text + out[b:]

    # from pycauset import ... Old ...
    def fix_import(m: re.Match) -> str:
        names = [x.strip() for x in m.group(2).split(",") if x.strip()]
        new: list[str] = []
        for x in names:
            base = x.split(" as ")[0].strip()
            repl = OLD[base][0] if base in OLD else x
            if repl not in new:
                new.append(repl)
        return f"{m.group(1)}{', '.join(new)}"

    out = re.sub(r"^(from pycauset import )([^\n(]+)$", fix_import, out, flags=re.M)
    for old in OLD:
        if re.search(rf"\b{old}\b", out):
            leftovers.append(f"remaining reference: {old}")
    return out, leftovers


if __name__ == "__main__":
    for path in sys.argv[1:]:
        src = open(path, encoding="utf-8").read()
        try:
            new, left = migrate(src)
        except SyntaxError as e:
            print(f"{path}: syntax error {e}")
            continue
        if new != src:
            open(path, "w", encoding="utf-8").write(new)
        for item in sorted(set(left)):
            print(f"{path}: {item}")
