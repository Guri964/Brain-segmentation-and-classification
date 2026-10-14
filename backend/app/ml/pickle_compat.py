"""Reading the ensemble's joblib artefacts, including pickles written by another environment.

A pickle stores the **module path its author's environment used**, not the class itself, so an
artefact can stop loading after a library renames one of its private modules.  The renames that
matter for these eight files:

* scikit-learn moved the gradient-boosting losses.  A file written by an older release asks for
  ``sklearn.ensemble._gb_losses.BinomialDeviance``, while a newer release keeps that class under
  ``sklearn._loss.loss.HalfBinomialLoss`` - and the old module no longer exists (measured here:
  ``import sklearn.ensemble._gb_losses`` raises, ``sklearn._loss.loss`` imports).  A file written
  where the class was importable under a bare name (``_loss.HalfBinomialLoss``) fails the same way,
  with ``ModuleNotFoundError: No module named '_loss'``;
* NumPy renamed ``numpy.core`` to ``numpy._core``, so a scaler written with NumPy 2 cannot be
  unpickled by NumPy 1 (and an older file, the other way round).

``joblib.load`` raises ``ModuleNotFoundError`` *before* any of this application's code runs, which
reads like a broken installation but is only a name change inside a healthy library.

What this module does
---------------------
It resolves those names against the **installed** library, and never guesses a value:

* a module that cannot be imported gets a temporary alias module, valid only while the file is being
  read, which resolves each attribute by name.  A plain module-path rename needs no further
  evidence: the class keeps its name, and for NumPy the rename resolves to the identical object
  (``numpy.core.multiarray._reconstruct is numpy._core.multiarray._reconstruct`` - True), which is
  the same approach scikit-learn's own unpickling shims take;
* the handful of *class* renames between scikit-learn releases (the gradient-boosting losses) are
  accepted only when the estimator's own declared ``loss`` agrees with the class and link that were
  substituted.  :func:`verify_estimator` performs that check and **refuses** the artefact when it
  fails, rather than handing the meta-learner a loss object the notebook never trained against;
* every substitution is recorded in ``report["substitutions"]`` and travels into the API payload and
  the audit report, so a reader can always see which class produced a number.

Nothing is silently repaired, the artefact on disk is never written to, and anything that cannot be
resolved raises :class:`~app.errors.ModelUnavailableError` naming the artefact, the missing module
and the candidates that were tried - so the ensemble reports ``incompatible`` with a reason instead
of a traceback.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import pickletools
import re
import sys
import threading
import types
import zipfile
import zlib
from pathlib import Path

from ..errors import ModelUnavailableError

#: Modules that no longer exist under that name, with the places their classes live now, in order.
MODULE_ALIASES: dict[str, tuple[str, ...]] = {
    # scikit-learn: gradient boosting, before the losses moved into sklearn._loss
    "sklearn.ensemble._gb_losses": (
        "sklearn._loss.loss",
        "sklearn._loss._loss",
        "sklearn.ensemble._gb",
    ),
    "sklearn.ensemble.gradient_boosting": ("sklearn.ensemble._gb", "sklearn.ensemble"),
    "sklearn.ensemble._gradient_boosting": ("sklearn.ensemble._gb",),
    "sklearn.ensemble.forest": ("sklearn.ensemble._forest",),
    "sklearn.tree.tree": ("sklearn.tree._tree",),
    "sklearn.preprocessing.data": ("sklearn.preprocessing._data",),
    # NumPy: numpy.core became numpy._core in NumPy 2.  Both directions are listed, because the
    # component that changed is not the last one, so toggling underscores does not find it.
    "numpy.core": ("numpy._core",),
    "numpy.core.multiarray": ("numpy._core.multiarray",),
    "numpy.core.numeric": ("numpy._core.numeric",),
    "numpy.core.umath": ("numpy._core.umath",),
    "numpy._core": ("numpy.core",),
    "numpy._core.multiarray": ("numpy.core.multiarray", "numpy.core._multiarray"),
    "numpy._core.numeric": ("numpy.core.numeric",),
    "numpy._core.umath": ("numpy.core.umath",),
}

#: The class names a gradient-boosting loss may resolve to.  Only these renames are accepted, and
#: only after :func:`verify_estimator` agrees with the estimator's own declared loss.
LOSS_CLASS_RENAMES: dict[str, str] = {
    "BinomialDeviance": "HalfBinomialLoss",
    "MultinomialDeviance": "HalfMultinomialLoss",
    "LeastSquaresError": "HalfSquaredError",
    "LeastAbsoluteError": "HalfAbsoluteError",
    "QuantileLossFunction": "PinballLoss",
    "HuberLossFunction": "HuberLoss",
}

#: Declared loss -> the class names that implement it, and the link they must be paired with.
#: (scikit-learn's ``_LOSSES`` table is keyed differently across releases, so this stays explicit;
#: a loss name that is not listed here is refused rather than guessed.)  The names cover the classes
#: this build installs *and* the names earlier releases used, because the check runs on an object
#: reconstructed from a file that may predate either.
LOSS_ALLOWED: dict[str, tuple[set[str], set[str]]] = {
    "log_loss": ({"HalfBinomialLoss", "CyHalfBinomialLoss", "BinomialDeviance"}, {"LogitLink"}),
    "deviance": ({"HalfBinomialLoss", "CyHalfBinomialLoss", "BinomialDeviance"}, {"LogitLink"}),
    "binomial_loss": ({"HalfBinomialLoss", "CyHalfBinomialLoss", "BinomialDeviance"}, {"LogitLink"}),
    "exponential": ({"ExponentialLoss", "CyExponentialLoss"}, {"LogLink", "HalfLogitLink"}),
    "exponential_loss": ({"ExponentialLoss", "CyExponentialLoss"}, {"LogLink", "HalfLogitLink"}),
    "squared_error": (
        {"HalfSquaredError", "CyHalfSquaredError", "LeastSquaresError"},
        {"IdentityLink"},
    ),
    "ls": ({"HalfSquaredError", "CyHalfSquaredError", "LeastSquaresError"}, {"IdentityLink"}),
    "absolute_error": (
        {"AbsoluteError", "CyAbsoluteError", "HalfAbsoluteError", "LeastAbsoluteError"},
        {"IdentityLink"},
    ),
    "lad": (
        {"AbsoluteError", "CyAbsoluteError", "HalfAbsoluteError", "LeastAbsoluteError"},
        {"IdentityLink"},
    ),
    "huber": ({"HuberLoss", "CyHuberLoss", "HuberLossFunction"}, {"IdentityLink"}),
    "huber_loss": ({"HuberLoss", "CyHuberLoss", "HuberLossFunction"}, {"IdentityLink"}),
    "quantile": ({"PinballLoss", "CyPinballLoss", "QuantileLossFunction"}, {"IdentityLink"}),
    "pinball_loss": ({"PinballLoss", "CyPinballLoss", "QuantileLossFunction"}, {"IdentityLink"}),
    "poisson_loss": ({"HalfPoissonLoss", "CyHalfPoissonLoss"}, {"LogLink"}),
    "gamma_loss": ({"HalfGammaLoss", "CyHalfGammaLoss"}, {"LogLink"}),
    "tweedie_loss": (
        {"HalfTweedieLoss", "CyHalfTweedieLoss", "HalfTweedieLossIdentity", "CyHalfTweedieLossIdentity"},
        {"LogLink", "IdentityLink"},
    ),
    "multinomial_loss": (
        {"HalfMultinomialLoss", "CyHalfMultinomialLoss", "MultinomialDeviance"},
        {"MultinomialLogit"},
    ),
    "multinomial_deviance": (
        {"HalfMultinomialLoss", "CyHalfMultinomialLoss", "MultinomialDeviance"},
        {"MultinomialLogit"},
    ),
}

#: Package roots a bare module name is expanded against (``_loss`` -> ``sklearn._loss``).
BARE_MODULE_ROOTS: tuple[str, ...] = (
    "sklearn",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn._loss",
    "sklearn.preprocessing",
    "sklearn.utils",
    "numpy",
    "numpy._core",
    "numpy.core",
    "scipy",
    "joblib",
)

_ALIAS_LOCK = threading.RLock()


def _importable(name: str) -> bool:
    """True when ``name`` can be imported (checked without importing it where possible)."""

    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):  # ValueError: parent package not importable
        return False


def _toggle_underscore(module: str) -> str:
    """``sklearn.ensemble.forest`` <-> ``sklearn.ensemble._forest`` (public vs private name)."""

    parts = module.split(".")
    last = parts[-1]
    parts[-1] = last[1:] if last.startswith("_") else "_" + last
    return ".".join(parts)


def candidate_modules(module: str) -> list[str]:
    """Every module path worth trying for ``module``, most likely first, without duplicates."""

    candidates: list[str] = [module]
    candidates.extend(MODULE_ALIASES.get(module, ()))
    if "." not in module:  # a bare name such as ``_loss``: look for it under every known root
        for root in BARE_MODULE_ROOTS:
            candidates.append(f"{root}.{module}")
            candidates.append(f"{root}.{module}.{module}")
        if "loss" in module:  # the gradient-boosting losses, where this build keeps them
            candidates.extend(("sklearn._loss.loss", "sklearn._loss._loss"))
    for base in list(candidates):  # the documented private-name change: last component only
        candidates.append(_toggle_underscore(base))
    seen: list[str] = []
    for name in candidates:
        if name not in seen:
            seen.append(name)
    return seen


def _aliased_names() -> frozenset[str]:
    """The module names currently served by an alias, so one alias never resolves through another."""

    return frozenset(
        name for name, module in list(sys.modules.items()) if isinstance(module, _AliasModule)
    )


class _AliasModule(types.ModuleType):
    """A stand-in for a module that no longer exists, resolving attributes by name.

    Registered in :data:`sys.modules` only while one file is being read, and only for a name that
    failed to import.  An attribute resolves from the first candidate module that has it - under the
    same name, under a documented scikit-learn loss rename, or (for a ``Cy`` Cython variant) under
    the Python implementation of the same loss.  Anything else raises ``AttributeError``, which the
    caller turns into a reported failure.
    """

    def __init__(self, requested: str, record: list[dict]) -> None:
        super().__init__(requested)
        self.__dict__["_requested"] = requested
        self.__dict__["_candidates"] = candidate_modules(requested)
        self.__dict__["_record"] = record

    def _names_to_try(self, attribute: str) -> list[tuple[str, bool]]:
        """The names to look for: the one asked for, then a documented loss rename, if any."""

        names = [(attribute, False)]
        renamed = LOSS_CLASS_RENAMES.get(attribute)
        if renamed:
            names.append((renamed, True))
        if attribute.startswith("Cy") and len(attribute) > 2:
            names.append((attribute[2:], True))
        return names

    def __getattr__(self, attribute: str) -> object:
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        resolving = self.__dict__.setdefault("_resolving", set())
        if attribute in resolving:  # a candidate that leads back here: refuse instead of looping
            raise AttributeError(attribute)
        resolving.add(attribute)
        aliases = _aliased_names()
        try:
            for module_name in self.__dict__["_candidates"]:
                # Never resolve through this alias itself (or another one): the requested name is
                # the first candidate, so considering it would recurse for ever.
                if module_name == self.__dict__["_requested"]:
                    continue
                try:
                    module = importlib.import_module(module_name)
                except Exception:  # noqa: BLE001 - a candidate that does not import is skipped
                    continue
                if isinstance(module, _AliasModule) or module_name in aliases:
                    continue
                for name, renamed in self._names_to_try(attribute):
                    if hasattr(module, name):
                        self.__dict__["_record"].append(
                            {
                                "requested": f"{self.__dict__['_requested']}.{attribute}",
                                "resolved": f"{module_name}.{name}",
                                "renamed_class": renamed,
                            }
                        )
                        return getattr(module, name)
        finally:
            resolving.discard(attribute)
        raise AttributeError(attribute)


def _is_worth_aliasing(module: str) -> bool:
    """True when an alias could plausibly resolve ``module`` (checked before touching sys.modules)."""

    if _importable(module):
        return False
    return any(_importable(name) for name in candidate_modules(module))


@contextlib.contextmanager
def _scoped_aliases(registered: list[str]):
    try:
        yield registered
    finally:
        for name in registered:
            sys.modules.pop(name, None)


def _aliases_for(path: Path, record: list[dict], registered: list[str]):
    """``joblib.load``, installing one temporary alias for each module it cannot import.

    A pickle stops at the first missing module, so a file that needs two renames needs two passes;
    the loop is bounded by the number of candidates so a cycle cannot spin.
    """

    import joblib

    for _ in range(8):
        try:
            return joblib.load(path)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            if missing in sys.modules or not _is_worth_aliasing(missing):
                raise
            sys.modules[missing] = _AliasModule(missing, record)
            registered.append(missing)
    raise ModelUnavailableError(
        f"{path.name} needs more module aliases than this loader will install; refusing to guess.",
        details={"artefact": path.name, "aliased_modules": list(registered)},
    )


def describe(path: str | Path, *, raw: bytes | None = None, limit: int = 4 << 20) -> dict:
    """What the file on disk actually is, read **without executing it**.

    Reports the container, the pickle protocol, and the ``module.name`` class paths the stream
    records, each with whether that name resolves in this environment and where the row came from.
    Two passes are used, and neither imports anything the file names:

    * the opcode walk, which for a joblib file stops at the first array buffer because joblib writes
      array bytes straight into the stream (``stopped_at_bytes`` says where it stopped);
    * a token pass over the stream text, which covers the rest of the file and both pickle framings
      (protocol 2 writes ``c<module>\\n<name>\\n``, protocol 4 writes the two strings back to back).

    A row's ``source`` says which pass produced it.  This is a description of the artefact, never a
    decision about it: the loader itself resolves names against the installed libraries.

    At most ``limit`` bytes are inspected, which only truncates a file far larger than these head
    artefacts; ``inspected_bytes`` says how much was read.
    """

    file = Path(path)
    raw = file.read_bytes() if raw is None else raw
    incoming: dict = {
        "artefact": file.name,
        "path": str(file),
        "bytes": len(raw),
        "container": _container(raw),
        "protocol": None,  # filled from whichever stream carries a pickle header
    }
    inspected = min(len(raw), limit)
    incoming["inspected_bytes"] = inspected
    payloads = _payloads(file, raw[:inspected], complete=inspected == len(raw))
    references: dict[tuple[str, str], str] = {}
    notes: list[str] = []
    for label, payload in payloads:
        if incoming["protocol"] is None and payload[:1] == b"\x80" and len(payload) > 1:
            incoming["protocol"] = payload[1]
        pairs, note = _references_from_opcodes(payload)
        notes.append(f"{label}: {note}")
        for pair in pairs:
            references.setdefault(pair, "opcodes")
        for pair in _references_from_tokens(payload):
            references.setdefault(pair, "tokens")
    rows = []
    for (module, name), source in references.items():
        row = {"module": module, "name": name, "resolvable": _importable(module), "source": source}
        if not row["resolvable"]:
            row["candidates"] = candidate_modules(module)[:6]
        rows.append(row)
    incoming["reading"] = notes
    incoming["references"] = rows
    incoming["unresolvable_modules"] = sorted({row["module"] for row in rows if not row["resolvable"]})
    return incoming


def _payloads(file: Path, raw: bytes, *, complete: bool) -> list[tuple[str, bytes]]:
    """The byte streams worth inspecting, each labelled with where it came from."""

    compressed = _zlib_streams(raw)
    # When the whole file is a compressed stream, the compressed bytes are not a stream of their own
    # and are not inspected as one.
    payloads: list[tuple[str, bytes]] = [] if compressed and compressed[0][0] == 0 else [("plain", raw)]
    for offset, decompressed in compressed:
        payloads.append((f"zlib stream at byte {offset}", decompressed))
    if raw[:4] == b"PK\x03\x04":  # a torch/keras archive keeps its pickle in a member file
        with contextlib.suppress(Exception):
            with zipfile.ZipFile(file) as archive:
                for member in archive.namelist():
                    if member.endswith(".pkl"):
                        payloads.append((f"archive member {member}", archive.read(member)))
    if not complete:
        payloads = payloads[:1]
    return payloads


def _zlib_streams(raw: bytes, window: int = 4096) -> list[tuple[int, bytes]]:
    """Decompress every zlib stream that starts in the first ``window`` bytes of the file.

    joblib's default is ``compress=3``, so the artefact usually *is* a zlib stream from byte 0; the
    header check (a valid CMF/FLG pair) says where one starts, and a truncated read is kept because
    what the file records is readable long before the end.
    """

    found: list[tuple[int, bytes]] = []
    for offset in range(min(len(raw) - 1, window)):
        if raw[offset] != 0x78 or (raw[offset] * 256 + raw[offset + 1]) % 31 != 0:
            continue
        try:
            data = zlib.decompressobj().decompress(raw[offset:])
        except zlib.error:
            continue
        if data:
            found.append((offset, data))
            break
    return found


def _container(raw: bytes) -> str:
    if raw[:4] == b"PK\x03\x04":
        return "zip-archive (torch/keras)"
    if raw[:6] == b"\x93NUMPY":
        return "numpy-npy"
    if len(raw) > 1 and raw[0] == 0x78 and (raw[0] * 256 + raw[1]) % 31 == 0:
        return "joblib, zlib-compressed"
    head = raw[:4096]
    if b"numpy_pickle" in head or b"joblib" in head:
        return "joblib, uncompressed"
    if raw[:1] in {b"\x80", b"c", b"(", b"}", b"]"}:
        return "pickle"
    return "unknown"


def _references_from_opcodes(payload: bytes) -> tuple[list[tuple[str, str]], str]:
    """``(module, name)`` for every reference the walked part of the stream records, plus a note.

    joblib writes array bytes inline, so the walk ends at the first array buffer; the pairs found up
    to that byte are still returned, because that is where the estimator's own class paths are.
    """

    pairs: list[tuple[str, str]] = []
    # ``STACK_GLOBAL`` pops the two names pushed immediately before it; only a memoising opcode may
    # sit between them (``BINGET``/``PUT`` restore a *different* stack value, so they clear the pair
    # - measured on a real joblib file, treating BINGET as transparent pairs two unrelated strings
    # and would report ``dtype.dtype`` as a class path).
    pending: list[str] = []
    transparent = {"PROTO", "FRAME", "MEMOIZE"}
    stopped = 0
    walked = 0
    try:
        for opcode, argument, position in pickletools.genops(payload):
            walked = position
            if opcode.name == "GLOBAL":
                module, _, name = str(argument).partition(" ")
                pairs.append((module, name))
                pending = []
            elif opcode.name == "STACK_GLOBAL":
                if len(pending) == 2:
                    pairs.append((pending[0], pending[1]))
                pending = []
            elif opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"}:
                pending = (pending + [str(argument)])[-2:]
            elif opcode.name not in transparent:
                pending = []
    except Exception as exc:  # noqa: BLE001 - joblib array framing ends the walk
        stopped = walked
        return pairs, (
            f"opcodes read up to byte {stopped} of {len(payload)} ({type(exc).__name__}: {exc}); "
            "joblib writes array bytes inline, so the rest of the stream is scanned as text"
        )
    return pairs, f"opcodes read the whole {len(payload)} byte stream"


#: Framings a class reference is written in: protocol 2 ``c<module>\n<name>\n``, protocol 4 the two
#: strings back to back.  Both are recovered by matching the module text, not by guessing.
_GLOBAL_LINE = re.compile(r"c([A-Za-z_][A-Za-z0-9_.]*)\n([A-Za-z_][A-Za-z0-9_]*)\n")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_REFERENCE_ROOTS = {"sklearn", "numpy", "scipy", "pandas", "joblib", "torch", "__main__"}


def _references_from_tokens(payload: bytes) -> list[tuple[str, str]]:
    """Class paths visible in the stream as text, in both pickle framings."""

    text = payload.decode("latin-1")
    pairs: list[tuple[str, str]] = [
        (match.group(1), match.group(2))
        for match in _GLOBAL_LINE.finditer(text)
        if _looks_like_a_reference(match.group(1))
    ]
    tokens = [match.group(0) for match in _TOKEN.finditer(text)]
    for first, second in zip(tokens, tokens[1:]):
        if "." in first and "." not in second and _looks_like_a_reference(first):
            pairs.append((first, second))
    return pairs


def _looks_like_a_reference(module: str) -> bool:
    """True for a plausible class-path module: a known library root, or a bare name that expands.

    A bare name matters here - an artefact written where a class was importable as ``_loss.X``
    records exactly that, and it is the shape of the reported failure - so it is accepted only when
    one of its expansions really exists in this environment.
    """

    if module.split(".")[0] in _REFERENCE_ROOTS:
        return True
    return "." not in module and any(_importable(name) for name in candidate_modules(module))


#: The loss *parameter* names scikit-learn renamed: 1.1 turned ``deviance`` into ``log_loss`` and
#: ``ls``/``lad`` into ``squared_error``/``absolute_error``.  The mapping is applied only to build the
#: installed library's own loss object; the estimator keeps the parameter value the file stored.
LOSS_PARAMETER_RENAMES: dict[str, str] = {
    "deviance": "log_loss",
    "ls": "squared_error",
    "lad": "absolute_error",
}


def repair_estimator(value: object, report: dict) -> None:
    """Make a loss object whole again, using the installed library's own construction of it.

    A pickle stores the *state* of the loss object, and a release that kept the loss in another
    module also shaped that state differently.  Measured on a genuine scikit-learn 1.3.2 artefact:
    the restored ``BinomialDeviance`` carries one attribute (``K``), while this release's
    ``HalfBinomialLoss`` sets nine in ``__init__`` and computes every probability through ``link`` -
    so the file loads and then fails at the first prediction with
    ``AttributeError: 'HalfBinomialLoss' object has no attribute 'link'``.

    The repair is structural, never guessed: when the restored object is missing attributes that the
    installed class defines, it is replaced by ``estimator._get_loss()`` - scikit-learn's own
    construction for the declared loss (its own table decides which class implements ``log_loss``
    for this number of classes).  What the file's own object carried is reported, and the fitted
    trees, the initial estimator and every parameter of the estimator itself are left untouched.

    Measured, on that same 1.3.2 artefact (200 trees, depth 3): after the repair the meta-learner
    returns the same labels and the same probabilities as the environment that wrote it, to within
    one unit in the last place (max abs difference 1.11e-16).  ``tests/test_final_ensemble.py``
    checks the same property with a file written here.
    """

    instance = getattr(value, "_loss", None)
    if instance is None or not hasattr(instance, "__dict__"):
        return
    cls = type(instance)
    try:
        complete = vars(cls())
    except Exception:  # noqa: BLE001 - cannot tell what a complete instance looks like
        return
    missing = sorted(set(complete) - set(vars(instance)))
    if not missing:
        return

    declared = getattr(value, "loss", None)
    factory = getattr(value, "_get_loss", None)
    if factory is None or not isinstance(declared, str):  # not a gradient-boosting estimator
        raise ModelUnavailableError(
            f"{report['artefact']} restores a {cls.__name__} that is missing "
            f"{', '.join(missing)}, and this build has no way to complete it: the object was written "
            "by a release that shaped it differently. Refusing the artefact instead of running it "
            "half-built.",
            details={"artefact": report["artefact"], "class": cls.__name__, "missing_attributes": missing},
        )
    installed = LOSS_PARAMETER_RENAMES.get(declared, declared)
    rebuilt = None
    failure: Exception | None = None
    try:
        if installed != declared:
            # the installed factory reads the public parameter, so it is given the name it knows
            value.loss = installed
        rebuilt = factory(sample_weight=None)
    except Exception as exc:  # noqa: BLE001 - reported as a refusal
        failure = exc
    finally:
        if installed != declared:
            value.loss = declared
    if rebuilt is None:
        raise ModelUnavailableError(
            f"{report['artefact']} restores a {cls.__name__} that is missing "
            f"{', '.join(missing)}, and the installed library cannot reconstruct the declared loss "
            f"'{declared}' ({type(failure).__name__}: {failure}). Refusing the artefact rather than "
            "running it with an incomplete loss.",
            details={
                "artefact": report["artefact"],
                "declared_loss": declared,
                "missing_attributes": missing,
                "reason": f"{type(failure).__name__}: {failure}",
            },
        )
    allowed = LOSS_ALLOWED.get(declared)
    if allowed is not None:
        classes, links = allowed
        link = getattr(rebuilt, "link", None)
        actual_link = type(link).__name__ if link is not None else None
        if type(rebuilt).__name__ not in classes or not (actual_link is None or actual_link in links):
            raise ModelUnavailableError(
                f"{report['artefact']} declares loss '{declared}', but the loss the installed "
                f"library builds for it is {type(rebuilt).__name__} with link "
                f"{actual_link or 'not exposed'}, which is not what the artefact's own class "
                "implemented. Refusing it rather than predicting with a different loss.",
                details={
                    "artefact": report["artefact"],
                    "declared_loss": declared,
                    "rebuilt_loss": type(rebuilt).__name__,
                    "rebuilt_link": actual_link,
                },
            )
    value._loss = rebuilt
    report["loss_rebuilt"] = {
        "declared_loss": declared,
        "parameter_renamed_from": declared if installed != declared else None,
        "restored_class": cls.__name__,
        "restored_attributes": sorted(vars(instance)),
        "installed_class": type(rebuilt).__name__,
        "missing_attributes": missing,
    }
    report.setdefault("compatibility_notes", []).append(
        f"the loss object the file carries is a {cls.__name__} with {len(vars(instance))} attribute(s) "
        f"({', '.join(sorted(vars(instance))) or 'none'}), while this release's class defines "
        f"{len(complete)}; it was rebuilt as {type(rebuilt).__name__}, the installed library's own "
        "loss for the declared "
        f"'{declared}'"
        + (f" (the file's parameter name '{declared}' is this release's '{installed}')"
           if installed != declared else "")
        + ". The fitted trees and every other attribute are the file's own."
    )


def summary(report: dict) -> dict:
    """The compact form of a load report, for a status payload or a prediction.

    Keeps what a reader needs to judge the artefact - which class paths it records, which of them
    had to be resolved against another name, and with what loader - and drops the raw reference
    table, which is long and only useful when diagnosing by hand.
    """

    references = report.get("references", [])
    substitutions = report.get("substitutions", [])
    resolved_as = {entry.get("requested"): entry.get("resolved") for entry in substitutions}
    # A path the file records that this environment does not have is still worth listing: it is why
    # the load needed a substitution at all.  ``resolved_as`` says what became of it, and stays null
    # when nothing here could stand in for it - which is the case that fails a load.
    missing = [
        {
            "class_path": f"{row['module']}.{row['name']}",
            "source": row.get("source"),
            "resolved_as": resolved_as.get(f"{row['module']}.{row['name']}"),
        }
        for row in references
        if not row.get("resolvable")
    ]
    return {
        "artefact": report.get("artefact"),
        "container": report.get("container"),
        "protocol": report.get("protocol"),
        "bytes": report.get("bytes"),
        "loader": report.get("loader"),
        "reference_rows": len(references),
        "class_paths_missing_here": missing,
        "resolved_from_other_modules": substitutions,
        "loss_rebuilt": report.get("loss_rebuilt"),
        "aliased_modules": report.get("aliased_modules", []),
        "notes": report.get("compatibility_notes", []),
        "reading": report.get("reading", []),
    }


def load(path: str | Path, *, artefact: str | None = None) -> tuple[object, dict]:
    """Load one joblib artefact, resolving only name changes, and report every one of them.

    Returns ``(object, report)``.  Raises :class:`ModelUnavailableError` - with the artefact name,
    the requested ``module.attribute`` and the candidates that were tried - when the object cannot
    be reconstructed safely, so the caller can report that state instead of retrying with a guess.
    """

    file = Path(path)
    name = artefact or file.name
    report = describe(file)
    report["artefact"] = name
    report["substitutions"] = []
    report["aliased_modules"] = []
    report["compatibility_notes"] = []
    report["loss_rebuilt"] = None  # set when the file's loss object had to be completed
    report["loader"] = "joblib"
    substitutions: list[dict] = report["substitutions"]
    registered: list[str] = report["aliased_modules"]
    try:
        with _ALIAS_LOCK:
            with _scoped_aliases(registered):
                value = _aliases_for(file, substitutions, registered)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        report["first_attempt_failed_with"] = f"ModuleNotFoundError: No module named '{missing}'"
        raise ModelUnavailableError(
            f"{name} needs the module '{missing}', which does not exist in this environment and has "
            "no equivalent under another name either. Install the library version that wrote the "
            "artefact - `pip install -r backend/requirements.txt` provides the versions this "
            "deployment is verified against. The file itself is intact and was not modified.",
            details={
                "artefact": name,
                "path": str(file),
                "missing_module": missing,
                "tried_modules": candidate_modules(missing)[:8],
                "unresolvable_modules": report["unresolvable_modules"],
                "aliased_modules": registered,
            },
        ) from exc
    except AttributeError as exc:
        raise ModelUnavailableError(
            f"{name} refers to '{exc}', which does not exist in the installed libraries; the class "
            "it was written with has no equivalent here, so the artefact is refused rather than "
            "substituted with a different implementation.",
            details={
                "artefact": name,
                "path": str(file),
                "attribute": str(exc),
                "aliased_modules": registered,
                "unresolvable_modules": report["unresolvable_modules"],
            },
        ) from exc
    if substitutions:
        report["loader"] = "joblib with compatibility aliases"
    verify_estimator(value, report)
    repair_estimator(value, report)
    return value, report


def verify_estimator(value: object, report: dict) -> None:
    """Refuse a substituted loss that the estimator's own declared loss does not allow.

    A module-path substitution cannot change behaviour: the class keeps its name and, for NumPy, the
    renamed module resolves to the identical object.  A *class* substitution can change behaviour, so
    it is trusted only when the estimator's declared ``loss`` is implemented by the substituted class
    paired with the right link.  When that cannot be shown, the artefact is refused.
    """

    renamed = [entry for entry in report.get("substitutions", []) if entry.get("renamed_class")]
    if not renamed:
        return
    loss = getattr(value, "_loss", None)
    declared = getattr(value, "loss", None)
    if loss is None or declared is None:  # not a gradient-boosting estimator
        report.setdefault("compatibility_notes", []).append(
            "a class was substituted for an object that is not a gradient-boosting estimator; the "
            "substitutions are listed for inspection."
        )
        return
    allowed = LOSS_ALLOWED.get(str(declared))
    if allowed is None:
        raise ModelUnavailableError(
            f"{report['artefact']} declares the loss '{declared}', which this build does not "
            "recognise, and its loss class had to be substituted; refusing rather than running the "
            "meta-learner with a loss it was not trained against.",
            details={
                "artefact": report["artefact"],
                "declared_loss": str(declared),
                "substitutions": renamed,
            },
        )
    classes, links = allowed
    actual_class = type(loss).__name__
    # A Cython loss does not expose its link the way the Python classes do; when it cannot be read,
    # the class check alone decides, and the note says so rather than implying a check that did not
    # happen.
    link = getattr(loss, "link", None)
    actual_link = type(link).__name__ if link is not None else None
    if actual_class not in classes or not (actual_link is None or actual_link in links):
        raise ModelUnavailableError(
            f"{report['artefact']} declares loss '{declared}', which needs one of {sorted(classes)} "
            f"with link {sorted(links)}, but the substituted object is {actual_class} with link "
            f"{actual_link or 'not exposed'}. Refusing the artefact instead of running the head on a "
            "loss it was not trained against.",
            details={
                "artefact": report["artefact"],
                "declared_loss": str(declared),
                "substituted_loss": actual_class,
                "substituted_link": actual_link,
                "substitutions": renamed,
            },
        )
    report.setdefault("compatibility_notes", []).append(
        f"loss '{declared}' was reconstructed as {actual_class} because the artefact records the "
        "class name of an earlier scikit-learn release; the estimator's own declared loss agrees "
        "with it"
        + (f" (link {actual_link})" if actual_link else " (the class does not expose its link)")
        + "."
    )


__all__ = [
    "BARE_MODULE_ROOTS",
    "LOSS_ALLOWED",
    "LOSS_PARAMETER_RENAMES",
    "LOSS_CLASS_RENAMES",
    "MODULE_ALIASES",
    "candidate_modules",
    "describe",
    "load",
    "repair_estimator",
    "summary",
    "verify_estimator",
]
