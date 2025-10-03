"""Support for making mutations to source code."""

import contextlib
import difflib
import logging
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
import ast


import cosmic_ray.plugins
from cosmic_ray.ast import Visitor, get_ast
from cosmic_ray.testing import run_tests
from cosmic_ray.util import read_python_source, restore_contents
from cosmic_ray.work_item import MutationSpec, TestOutcome, WorkResult, WorkerOutcome
from typing import Optional
import json

log = logging.getLogger(__name__)


# pylint: disable=R0913
def mutate_and_test(mutations: Iterable[MutationSpec], test_command, timeout) -> WorkResult:
    """Apply a sequence of mutations, run thest tests, and reports the results.

    This is fundamentally the mutation(s)-and-test-run implementation at the heart of Cosmic Ray.

    There are three high-level ways that a worker can finish. First, it could fail exceptionally, meaning that some
    uncaught exception made its way from some part of the operation to terminate the function. This function will
    intercept all exceptions and return it in a non-exceptional structure.

    Second, the mutation machinery may determines that - for any of the mutations - there is no mutation to be made (e.g.
    the 'occurrence' is too high).  In this case there is no way to report a test result (i.e. killed, survived, or
    incompetent) so a special value is returned indicating that no mutation is possible.

    Finally, and hopefully normally, the worker will find that it can run a test. It will do so and report back the
    result - killed, survived, or incompetent - in a structured way.

    Args:
        mutations: An iterable of ``MutationSpec``\\s describing the mutations to make.
        test_command: The command to execute to run the tests
        timeout: The maximum amount of time (seconds) to let the tests run

    Returns:
        A ``WorkResult``.

    Raises:
        This will generally not raise any exceptions. Rather, exceptions will be reported using the 'exception'
        result-type in the return value.

    """
    try:
        with contextlib.ExitStack() as stack:
            file_changes: dict[Path, tuple[str, str]] = {}
            for mutation in mutations:
                operator_class = cosmic_ray.plugins.get_operator(mutation.operator_name)
                try:
                    operator_args = mutation.operator_args
                except AttributeError:
                    operator_args = {}
                operator = operator_class(**operator_args)

                (previous_code, mutated_code) = stack.enter_context(
                    use_mutation(mutation.module_path, operator, mutation.occurrence)
                )

                # If there's no mutated code, then no mutation was possible.
                if mutated_code is None:
                    return WorkResult(
                        worker_outcome=WorkerOutcome.NO_TEST,
                    )

                original_code, _ = file_changes.get(mutation.module_path, (previous_code, mutated_code))
                file_changes[mutation.module_path] = original_code, mutated_code

            test_outcome, output = run_tests(test_command, timeout)

            diffs = [
                _make_diff(original_code, mutated_code, module_path)
                for module_path, (original_code, mutated_code) in file_changes.items()
            ]

            result = WorkResult(
                output=output,
                diff="\n".join(chain(*diffs)),
                test_outcome=test_outcome,
                worker_outcome=WorkerOutcome.NORMAL,
            )

    except Exception:  # noqa # pylint: disable=broad-except
        return WorkResult(
            output=traceback.format_exc(), test_outcome=TestOutcome.INCOMPETENT, worker_outcome=WorkerOutcome.EXCEPTION
        )

    return result


@contextmanager
def use_mutation(module_path, operator, occurrence):
    """A context manager that applies a mutation for the duration of a with-block.

    This applies a mutation to a file on disk, and after the with-block it put the unmutated code
    back in place.

    Args:
        module_path: The path to the module to mutate.
        operator: The `Operator` instance to use.
        occurrence: The occurrence of the operator to apply.

    Yields:
        A `(unmutated-code, mutated-code)` tuple to the with-block. If there was no
        mutation performed, the `mutated-code` is `None`.
    """
    with restore_contents(module_path):
        original_code, mutated_code = apply_mutation(module_path, operator, occurrence)
        yield original_code, mutated_code


def apply_mutation(module_path, operator, occurrence):
    """Apply a specific mutation to a file on disk.

    Args:
        module_path: The path to the module to mutate.
        operator: The `operator` instance to use.
        occurrence: The occurrence of the operator to apply.

    Returns:
        A `(unmutated-code, mutated-code)` tuple to the with-block. If there was
        no mutation performed, the `mutated-code` is `None`.
    """
    return MutationVisitor.mutate_path(module_path, operator, occurrence)


def mutate_code(code, operator, occurrence):
    """Apply a specific mutation to a code string.

    Args:
        code: The code to mutate.
        operator: The `operator` instance to use.
        occurrence: The occurrence of the operator to apply.

    Returns:
        The mutated code, or None if no mutation was applied.
    """
    return MutationVisitor.mutate_code(code, operator, occurrence)


class MutationVisitor(Visitor):
    """Visitor that mutates a module with the specific occurrence of an operator.

    This will perform at most one mutation in a walk of an AST. If this performs
    a mutation as part of the walk, it will store the mutated node in the
    `mutant` attribute. If the walk does not result in any mutation, `mutant`
    will be `None`.

    Note that `mutant` is just the specifically mutated node. It will generally
    be a part of the larger AST which is returned from `walk()`.
    """

    @classmethod
    def mutate_code(cls, source, operator, occurence):
        ast = get_ast(source)
        visitor = cls(occurence, operator)
        mutated_ast = visitor.walk(ast)
        if not visitor.mutation_applied:
            return None
        return mutated_ast.get_code()

    @classmethod
    def mutate_path(cls, module_path, operator, occurrence):
        """Mutate a module in place on disk.

        Args:
            module_path (Path): The path to the module file.
            operator (Operator): The operator to apply.
            occurrence (int): The occurrence of the operator to apply.

        Returns:
            tuple[str, str|None]: The original code and the mutated code (or None)
        """
        log.info("Applying mutation: path=%s, op=%s, occurrence=%s", module_path, operator, occurrence)

        original_code = read_python_source(module_path)
        mutated_code = cls.mutate_code(original_code, operator, occurrence)

        if mutated_code is None:
            return original_code, None

        with module_path.open(mode="wt", encoding="utf-8") as handle:
            handle.write(mutated_code)
            handle.flush()

        return original_code, mutated_code

    def __init__(self, occurrence, operator):
        self.operator = operator
        self._occurrence = occurrence
        self._count = 0
        self._mutation_applied = False

    @property
    def mutation_applied(self):
        "Whether this visitor has applied a mutation."
        return self._mutation_applied

    def visit(self, node):
        for index, _ in enumerate(self.operator.mutation_positions(node)):
            if self._count == self._occurrence:
                self._mutation_applied = True
                node = self.operator.mutate(node, index)
            self._count += 1

        return node


def _make_diff(original_source, mutated_source, module_path):
    module_diff = ["--- mutation diff ---"]
    for line in difflib.unified_diff(
        original_source.split("\n"),
        mutated_source.split("\n"),
        fromfile="a" + str(module_path),
        tofile="b" + str(module_path),
        lineterm="",
    ):
        module_diff.append(line)
    return module_diff

def get_mutation(module_path, operator, occurrence) -> tuple[str, Optional[str]]:
    """Get a mutation without applying it to disk.
    
    Args:
        module_path: The path to the module to mutate.
        operator: The `operator` instance to use.
        occurrence: The occurrence of the operator to apply.

    Returns:
        A `(unmutated-code, mutated-code)` . If there was no mutation performed, 
        the `mutated-code` is `None`.
    """
    log.info("Getting mutation: path=%s, op=%s, occurrence=%s", module_path, operator, occurrence)
    original_code = read_python_source(module_path)
    mutated_code = MutationVisitor.mutate_code(original_code, operator, occurrence)
    return original_code, mutated_code

def get_mutations_only(mutations: Iterable[MutationSpec]) -> WorkResult:
    """Generate mutations without executing the test command.

    This collects the original and mutated source for each mutation spec and
    writes them to ``mutations.json`` in the current working directory.

    Returns a ``WorkResult`` whose ``worker_outcome`` is NORMAL if at least one
    mutation produced code, or NO_TEST if none did.
    """
    mutation_records = []

    try:
        for mutation in mutations:
            try:
                operator_class = cosmic_ray.plugins.get_operator(mutation.operator_name)
                try:
                    operator_args = mutation.operator_args
                except AttributeError:
                    operator_args = {}
                operator = operator_class(**operator_args)

                original_code, mutated_code = get_mutation(mutation.module_path, operator, mutation.occurrence)
                status = 'mutated' if mutated_code is not None else 'no_mutation'

                context_node = find_context(mutated_code, mutation.start_pos[0])

                if context_node:
                    node_source = ast.get_source_segment(mutated_code, context_node)
                else:
                    node_source = None

                diff_lines = _make_diff(original_code, mutated_code, mutation.module_path) if mutated_code else []
                mutation_records.append({
                    "module_path": str(mutation.module_path),
                    "operator_name": mutation.operator_name,
                    "occurrence": mutation.occurrence,
                    "start_pos": mutation.start_pos,
                    "end_pos": mutation.end_pos,
                    "operator_args": getattr(mutation, 'operator_args', {}),
                    "mutation": extract_mutated_line(mutated_code, mutation.start_pos, mutation.end_pos) if mutated_code else None,
                    "mutated_code": node_source,
                    "diff": "\n".join(diff_lines) if diff_lines else None,
                    "status": status,
                })
            except Exception:
                mutation_records.append({
                    "module_path": str(getattr(mutation, 'module_path', 'UNKNOWN')),
                    "operator_name": getattr(mutation, 'operator_name', 'UNKNOWN'),
                    "occurrence": getattr(mutation, 'occurrence', -1),
                    "start_pos": None,
                    "end_pos": None,
                    "operator_args": getattr(mutation, 'operator_args', {}),
                    "mutation": None,
                    "mutated_code": None,
                    "diff": None,
                    "status": 'error',
                    "error": traceback.format_exc(),
                })

        any_error = any(r.get('status') == 'error' for r in mutation_records)
        any_mutated = any(r.get('status') == 'mutated' for r in mutation_records)

        output_path = Path("mutations.json")
        existing: list = []
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        existing = data
                    else:
                        existing = [data]
            except Exception:
                log.warning("Failed reading existing mutations.json; starting fresh", exc_info=True)
                existing = []

        def _key(rec):
            return (
                rec.get('module_path'),
                rec.get('operator_name'),
                rec.get('occurrence'),
                tuple(rec.get('start_pos') or []),
                tuple(rec.get('end_pos') or []),
                rec.get('status'),
            )

        merged = { _key(r): r for r in existing }
        for rec in mutation_records:
            merged[_key(rec)] = rec
        final_records = list(merged.values())

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(final_records, fh, indent=2)
        log.info("Stored %d total mutation records (%d new) in %s", len(final_records), len(mutation_records), output_path.resolve())

        if any_error:
            worker_outcome = WorkerOutcome.EXCEPTION
            test_outcome = TestOutcome.INCOMPETENT
        elif any_mutated:
            worker_outcome = WorkerOutcome.NORMAL
            test_outcome = TestOutcome.NO_TEST
        else:
            worker_outcome = WorkerOutcome.NO_TEST
            test_outcome = TestOutcome.NO_TEST

        return WorkResult(
            worker_outcome=worker_outcome,
            test_outcome=test_outcome,
            output=f"Recorded {len(mutation_records)} records; total now {len(final_records)}"
        )
    except Exception:
        return WorkResult(
            output=traceback.format_exc(),
            test_outcome=TestOutcome.INCOMPETENT,
            worker_outcome=WorkerOutcome.EXCEPTION,
        )


def find_context(code: str, line: int):
    tree = ast.parse(code)

    def node_span(node):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            return node.lineno, node.end_lineno
        if hasattr(node, "lineno"):
            return node.lineno, node.lineno
        return None

    enclosing = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            span = node_span(node)
            if span and span[0] <= line <= span[1]:
                if enclosing is None or span[0] >= enclosing[1]:
                    enclosing = (node, span)
    return enclosing[0] if enclosing else None

def extract_mutated_line(code: str, start_pos: list[int], end_pos: list[int]) -> str:
    """
    Return the line where the mutation is located at.
    """
    if code is None:
        raise ValueError("Input 'code' cannot be None.")

    lines = code.splitlines() 
    start_line, start_col = start_pos
    start_idx = start_line - 1

    return lines[start_idx]