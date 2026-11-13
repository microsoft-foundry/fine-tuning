"""
Math grading utilities for RL training.

Includes math_normalize functionality that was dependency of grader.
"""

import contextlib
import importlib.util
import io
import logging
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, TypeVar, cast

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

logger = logging.getLogger(__name__)


T = TypeVar("T")

# ======================================================================
# Math Normalize Functions
# ======================================================================


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(str(answer))
    except Exception:
        return answer


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# ======================================================================
# Extract Boxed Functions
# ======================================================================


def extract_boxed(text: str) -> str:
    """
    Extract the context of the last \\boxed{...} in the text.
    Used for getting answers from hendrycks math
    """
    boxed_strs = []
    stack = []
    for ichar in range(len(text)):
        if text[ichar] == "{":
            stack.append(ichar)
        elif text[ichar] == "}":
            if len(stack) == 0:
                raise ValueError("Unmatched }")
            last_open_start = stack.pop()
            # check if start is preceded by \boxed
            if text[:last_open_start].endswith("\\boxed"):
                boxed_strs.append(text[last_open_start + 1 : ichar])
    if len(boxed_strs) > 0:
        return boxed_strs[-1]
    else:
        # maybe there's something like '\boxed 2' without curly braces
        match = re.search(r"\\boxed\s+([a-zA-Z0-9]+)", text)
        if match:
            return match.group(1)
        else:
            raise ValueError("No boxed strings found")


# ======================================================================
# Grader Functions
# ======================================================================

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except (ValueError, OverflowError):
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x_str: str) -> bool:
    try:
        x_str = _strip_properly_formatted_commas(x_str)
        x = float(x_str)
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _str_to_int(x_str: str) -> int:
    x_str = x_str.replace(",", "")
    x = float(x_str)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    return all(re.search(bad_regex, expr) is not None for bad_regex in BAD_REGEXES)


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    is_correct = False

    if (
        len(ground_truth_elems) > 1
        and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
        )
        or len(ground_truth_elems) != len(given_elems)
    ):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=True):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_math_verify(given_answer: str, ground_truth: str) -> bool:
    """
    Use the math_verify package to verify the answer.
    """
    from math_verify import parse, verify

    #   Make sure the answer is wrapped in $ if it already isn't otherwise it is not parsed correctly
    if not given_answer.startswith("$") and not given_answer.endswith("$"):
        given_answer = f"${given_answer}$"
    if not ground_truth.startswith("$") and not ground_truth.endswith("$"):
        ground_truth = f"${ground_truth}$"

    # The outer isolated worker owns the killable deadline. In particular,
    # math-verify's Windows timeout wrapper must not spawn nested processes
    # or silently convert a startup/grader failure into an empty parse.
    given_answer_parsed = parse(given_answer, parsing_timeout=None, raise_on_error=True)
    ground_truth_parsed = parse(ground_truth, parsing_timeout=None, raise_on_error=True)

    is_correct = verify(
        given_answer_parsed, ground_truth_parsed, timeout_seconds=None, raise_on_error=True
    )

    return is_correct


# ======================================================================
# Timeout Functions
# ======================================================================


# Define a custom exception for timeouts
class TimeoutException(Exception):
    pass


_MAX_WORKER_INPUT_BYTES = 1024 * 1024
# Capped files carry the request and response, not potentially blocking pipe
# writes. Stdio carries only a one-byte start handshake; grader output is
# discarded, so neither printed output nor a Queue feeder can grow IPC.
_MAX_WORKER_MESSAGE_BYTES = 4000
_WORKER_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
_WORKER_STARTUP_TIMEOUT_SECONDS = 30.0
_WORKER_TERMINATION_GRACE_SECONDS = 1.0


def _limit_worker_resources(timeout_seconds: float) -> None:
    """Apply Linux limits only in the disposable child, never in the server.

    Windows still has the parent-enforced wall timeout and bounded IPC, but
    does not offer these POSIX CPU/address-space limits through the stdlib.
    This process boundary is not a sandbox for executing untrusted Python.
    """
    if sys.platform != "linux":
        return

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # RLIMIT_CPU is cumulative from process creation. Imported dependencies
    # must not consume the expression's CPU allowance either.
    cpu_limit = math.ceil(usage.ru_utime + usage.ru_stime + timeout_seconds) + 1
    limits = (
        (resource.RLIMIT_AS, _WORKER_MEMORY_LIMIT_BYTES),
        (resource.RLIMIT_CPU, max(1, cpu_limit)),
        (resource.RLIMIT_CORE, 0),
    )
    for resource_id, requested in limits:
        soft, hard = resource.getrlimit(resource_id)
        finite_limits = [requested]
        finite_limits.extend(limit for limit in (soft, hard) if limit != resource.RLIM_INFINITY)
        limit = min(finite_limits)
        resource.setrlimit(resource_id, (limit, limit))


class _InvocationPickler(pickle.Pickler):
    """Record exact Python source origins without serializing the caller's runtime."""

    def __init__(self, stream: io.BytesIO):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.origins: dict[str, str] = {}
        self.roots: set[str] = set()

    def persistent_id(self, obj: Any) -> None:
        if isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType)):
            name = getattr(obj, "__module__", None)
            if name in ("__main__", "__mp_main__"):
                raise ValueError("Grading callables must live in an importable module, not __main__")
            module = sys.modules.get(name)
            origin = getattr(module, "__file__", None)
            if origin and Path(origin).suffix in (".py", ".pyw"):
                source = Path(origin).resolve()
                self.origins[name] = str(source)
                root = source.parent
                package = getattr(module, "__package__", "") or ""
                for _ in package.split(".") if package else ():
                    root = root.parent
                self.roots.add(str(root))
        return None


class _SourceUnpickler(pickle.Unpickler):
    def __init__(self, invocation: bytes, origins: dict[str, str]):
        super().__init__(io.BytesIO(invocation))
        self.origins = origins

    def find_class(self, module: str, name: str) -> Any:
        if module not in self.origins:
            return super().find_class(module, name)
        source = Path(self.origins[module])
        if source == Path(__file__).resolve():
            loaded = sys.modules[__name__]
            sys.modules[module] = loaded
        else:
            loaded = sys.modules.get(module)
            if loaded is None:
                spec = importlib.util.spec_from_file_location(module, source)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load grading callable from {source}")
                loaded = importlib.util.module_from_spec(spec)
                sys.modules[module] = loaded
                spec.loader.exec_module(loaded)
            elif Path(loaded.__file__).resolve() != source:
                raise ImportError(f"Grading callable source mismatch for {module}")
        value: Any = loaded
        for part in name.split("."):
            value = getattr(value, part)
        return value


def _worker_response(succeeded: bool, value: Any) -> bytes:
    try:
        payload = pickle.dumps((succeeded, value), protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > _MAX_WORKER_MESSAGE_BYTES:
            raise ValueError("grading worker response exceeds the IPC size limit")
    except Exception as exc:
        # Keep unpicklable exceptions visible, never turn them into timeouts.
        description = type(value).__name__[:128]
        if not succeeded:
            description += f": {str(value)[:256]}"
        error = RuntimeError(
            f"Cannot transfer grading worker {'result' if succeeded else 'exception'} "
            f"({description}): {str(exc)[:256]}"
        )
        payload = pickle.dumps((False, error), protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def _timeout_process_worker(directory: Path) -> None:
    """File entrypoint: no multiprocessing preparation or recipe __main__ replay."""
    try:
        with (directory / "request").open("rb") as stream:
            request = stream.read(_MAX_WORKER_INPUT_BYTES + 1)
        if len(request) > _MAX_WORKER_INPUT_BYTES:
            raise ValueError("grading worker input exceeds the IPC size limit")
        origins, roots, invocation, timeout_seconds = pickle.loads(request)
        _limit_worker_resources(_WORKER_STARTUP_TIMEOUT_SECONDS + timeout_seconds)
        # -I ignores ambient PYTHONPATH. Only explicitly recorded callable
        # source roots are added, and the grader itself is already loaded.
        sys.path[:0] = roots
        func, args, kwargs = _SourceUnpickler(invocation, origins).load()
        if func is grade_answer_math_verify:
            __import__("math_verify")
        _limit_worker_resources(timeout_seconds)
        (directory / "ready").touch()
        # The parent bounds this handshake with communicate(timeout=...).
        # Parent exit closes stdin, and any deadline kills this entire child.
        if sys.stdin.buffer.read(1) != b"G":
            raise RuntimeError("Math grading worker did not receive the start handshake")
        value = func(*args, **kwargs)
        succeeded = True
    except BaseException as exc:
        value = exc
        succeeded = False
    (directory / "response").write_bytes(_worker_response(succeeded, value))


def _stop_worker(process: subprocess.Popen) -> None:
    """Reap a worker with bounded waits, escalating past ignored SIGTERM."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_WORKER_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=_WORKER_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Could not terminate the timed-out math grading worker") from exc


def _read_worker_result(process: subprocess.Popen, directory: Path) -> Any:
    if process.returncode != 0:
        raise RuntimeError(f"Math grading worker exited with code {process.returncode}")
    try:
        with (directory / "response").open("rb") as stream:
            payload = stream.read(_MAX_WORKER_MESSAGE_BYTES + 1)
    except FileNotFoundError as exc:
        raise RuntimeError("Math grading worker exited without a result") from exc
    if len(payload) > _MAX_WORKER_MESSAGE_BYTES:
        raise RuntimeError("grading worker response exceeds the IPC size limit")
    try:
        succeeded, value = pickle.loads(payload)
        if not isinstance(succeeded, bool) or (not succeeded and not isinstance(value, BaseException)):
            raise ValueError("invalid result envelope")
    except Exception as exc:
        raise RuntimeError("Invalid math grading worker response") from exc
    if not succeeded:
        raise value
    return value


def _wait_worker_ready(process: subprocess.Popen, directory: Path, deadline: float) -> None:
    while not (directory / "ready").exists():
        if process.poll() is not None:
            _read_worker_result(process, directory)
            raise RuntimeError("Math grading worker exited without becoming ready")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Math grading worker initialization timed out")
        try:
            process.wait(timeout=min(0.01, remaining))
        except subprocess.TimeoutExpired:
            pass


def _worker_interpreter() -> tuple[str, dict[str, str] | None]:
    if sys.platform != "win32":
        return sys.executable, None

    import ctypes

    # Windows venv/Store executables can be redirectors. Killing the
    # redirector's Popen handle would leave the real grader running. Start
    # the actual current image, retaining venv resolution just as CPython's
    # multiprocessing launcher does, but without its __main__ preparation.
    image = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetModuleFileNameW(None, image, len(image))
    if not length or length >= len(image):
        raise OSError("Cannot locate the running Python interpreter image")
    environment = os.environ.copy()
    environment["__PYVENV_LAUNCHER__"] = sys.executable
    return image.value, environment


def run_with_timeout_signal(
    func: Callable[..., T],
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] | None = None,
    timeout_seconds: float = 5,
) -> T | None:
    """Run a picklable callable in a lightweight, disposable interpreter.

    Return its result, or None only on an expression wall-clock timeout.
    Initialization has a separate bounded deadline; startup, grader and IPC
    failures propagate. Cleanup adds at most two termination grace periods.
    Requests are capped at 1 MiB, responses at 4000 bytes, with no output queue.

    Execute this file with -I, never fork the server or replay its __main__.
    Callables must live in importable modules; explicit Python source origins
    support source checkouts without passing the training runtime to the child.
    """
    if not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite")
    if timeout_seconds <= 0:
        logger.warning("Function timed out after %s seconds.", timeout_seconds)
        return None

    stream = io.BytesIO()
    pickler = _InvocationPickler(stream)
    # Reduced callable objects can import this grader during initialization
    # even when their pickle contains only their own reconstruction function.
    pickler.persistent_id(grade_answer)
    pickler.dump((func, args, kwargs or {}))
    request = pickle.dumps(
        (pickler.origins, sorted(pickler.roots), stream.getvalue(), timeout_seconds),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    if len(request) > _MAX_WORKER_INPUT_BYTES:
        raise ValueError("grading worker input exceeds the IPC size limit")

    with tempfile.TemporaryDirectory(prefix="math-grading-") as temporary:
        directory = Path(temporary)
        (directory / "request").write_bytes(request)
        executable, environment = _worker_interpreter()
        startup_deadline = time.monotonic() + _WORKER_STARTUP_TIMEOUT_SECONDS
        process = subprocess.Popen(
            [executable, "-I", "-u", "-B", str(Path(__file__).resolve()),
             "--math-grading-worker", str(directory)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=environment,
        )
        try:
            _wait_worker_ready(process, directory, startup_deadline)
            try:
                process.communicate(input=b"G", timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                logger.warning("Function timed out after %s seconds.", timeout_seconds)
                return None
            return cast(T, _read_worker_result(process, directory))
        finally:
            _stop_worker(process)
            if process.stdin is not None:
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--math-grading-worker":
        raise SystemExit("This file is an internal math grading worker")
    _timeout_process_worker(Path(sys.argv[2]))
