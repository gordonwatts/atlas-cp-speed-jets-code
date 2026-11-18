"""Borrow original chunk arrays for the jet CorrectionLib provider on this thread."""
import ctypes as ct
from contextlib import contextmanager

import numpy as np


class JetBuffers:
    def __init__(self, provider):
        self.lib = ct.CDLL(str(provider))
        self.lib.jets_buffer_error.restype = ct.c_char_p
        off = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
        integer = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        real = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        for name, columns in (("truth", [real] * 4), ("vertices", [integer] * 2), ("weights", [real])):
            bind = getattr(self.lib, "jets_bind_" + name)
            bind.argtypes = [off, ct.c_size_t, *columns, ct.c_size_t]
            bind.restype = ct.c_int
            release = getattr(self.lib, "jets_unbind_" + name)
            release.argtypes, release.restype = [], ct.c_int
        self._bind_jmr_truth = getattr(self.lib, "jets_bind_jmr_truth", self.lib.jets_bind_truth)
        self._bind_jmr_truth.argtypes = self.lib.jets_bind_truth.argtypes
        self._bind_jmr_truth.restype = ct.c_int

    @contextmanager
    def _borrow(self, name, offsets, dtype, *columns, binder=None):
        offsets = np.ascontiguousarray(offsets,dtype=np.int64)
        arrays = [np.ascontiguousarray(a,dtype=dtype) for a in columns]
        if any(a.ndim != 1 or len(a)!=len(arrays[0]) for a in arrays):
            raise ValueError("jagged buffer columns must have matching flat lengths")
        if binder is None:
            binder = getattr(self.lib, "jets_bind_" + name)
        if binder(offsets, len(offsets), *arrays, len(arrays[0])):
            raise ValueError(self.lib.jets_buffer_error().decode())
        try:
            yield
        finally:
            if getattr(self.lib, "jets_unbind_" + name)():
                raise RuntimeError(self.lib.jets_buffer_error().decode())

    @contextmanager
    def unsigned_sign_pairs(self, rows):
        pairing = _UnsignedSignPairs(self.lib)
        pairing.call("jets_begin_unsigned_pairs", rows)
        try:
            yield pairing
        finally:
            pairing.call("jets_end_unsigned_pairs")

    def truth(self, offsets, pt, eta, phi, mass):
        return self._borrow("truth",offsets,np.float64,pt,eta,phi,mass)

    def jmr_truth(self, offsets, pt, eta, phi, mass):
        return self._borrow("truth", offsets, np.float64, pt, eta, phi, mass, binder=self._bind_jmr_truth)

    def vertices(self, offsets, types, ntracks):
        return self._borrow("vertices",offsets,np.int32,types,ntracks)

    def weights(self, offsets, weights):
        return self._borrow("weights",offsets,np.float64,weights)

class _UnsignedSignPairs:
    def __init__(self, library):
        if not hasattr(library, "jets_begin_unsigned_pairs"):
            raise ValueError("The sign-pair experiment requires the opt-in compiled provider")
        self.library = library
        library.jets_unsigned_pair_error.argtypes = []
        library.jets_unsigned_pair_error.restype = ct.c_char_p
        signatures = (
            ("jets_begin_unsigned_pairs", [ct.c_size_t]),
            ("jets_end_unsigned_pairs", []),
            ("jets_unsigned_pair_begin_state", []),
            ("jets_unsigned_pair_end_state", [ct.c_int]),
            ("jets_unsigned_pair_discard", []),
        )
        for name, arguments in signatures:
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ct.c_int

    def call(self, name, *args):
        if getattr(self.library, name)(*args):
            raise RuntimeError(self.library.jets_unsigned_pair_error().decode())

    def evaluate(self, evaluator, *args):
        self.call("jets_unsigned_pair_begin_state")
        completed = False
        try:
            values = evaluator(*args)
            completed = True
            return values
        finally:
            self.call("jets_unsigned_pair_end_state", int(completed))

    def discard(self):
        self.call("jets_unsigned_pair_discard")
