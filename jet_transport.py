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

    @contextmanager
    def _borrow(self, name, offsets, dtype, *columns):
        offsets = np.ascontiguousarray(offsets,dtype=np.int64)
        arrays = [np.ascontiguousarray(a,dtype=dtype) for a in columns]
        if any(a.ndim != 1 or len(a)!=len(arrays[0]) for a in arrays):
            raise ValueError("jagged buffer columns must have matching flat lengths")
        binder = getattr(self.lib, "jets_bind_" + name)
        if binder(offsets, len(offsets), *arrays, len(arrays[0])):
            raise ValueError(self.lib.jets_buffer_error().decode())
        try:
            yield
        finally:
            if getattr(self.lib, "jets_unbind_" + name)():
                raise RuntimeError(self.lib.jets_buffer_error().decode())

    def truth(self, offsets, pt, eta, phi, mass):
        return self._borrow("truth",offsets,np.float64,pt,eta,phi,mass)

    def vertices(self, offsets, types, ntracks):
        return self._borrow("vertices",offsets,np.int32,types,ntracks)

    def weights(self, offsets, weights):
        return self._borrow("weights",offsets,np.float64,weights)
