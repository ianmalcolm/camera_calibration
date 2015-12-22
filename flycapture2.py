'''Wrapper for FlyCapture2_C.h

Generated with:
./ctypesgen.py -lflycapture-c /usr/include/flycapture/C/FlyCapture2_C.h /usr/include/flycapture/C/FlyCapture2Defs_C.h /usr/include/flycapture/C/FlyCapture2GUI_C.h /usr/include/flycapture/C/FlyCapture2Internal_C.h /usr/include/flycapture/C/FlyCapture2Platform_C.h -o /home/ian/github/flycapture2.py

Do not modify this file.
'''

__docformat__ =  'restructuredtext'

# Begin preamble

import ctypes, os, sys
from ctypes import *

_int_types = (c_int16, c_int32)
if hasattr(ctypes, 'c_int64'):
    # Some builds of ctypes apparently do not have c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (c_int64,)
for t in _int_types:
    if sizeof(t) == sizeof(c_size_t):
        c_ptrdiff_t = t
del t
del _int_types

class c_void(Structure):
    # c_void_p is a buggy return type, converting to int, so
    # POINTER(None) == c_void_p is actually written as
    # POINTER(c_void), so it can be treated as a real pointer.
    _fields_ = [('dummy', c_int)]

def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

class UserString:
    def __init__(self, seq):
        if isinstance(seq, basestring):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq)
    def __str__(self): return str(self.data)
    def __repr__(self): return repr(self.data)
    def __int__(self): return int(self.data)
    def __long__(self): return long(self.data)
    def __float__(self): return float(self.data)
    def __complex__(self): return complex(self.data)
    def __hash__(self): return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)
    def __contains__(self, char):
        return char in self.data

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.__class__(self.data[index])
    def __getslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, basestring):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other))
    def __radd__(self, other):
        if isinstance(other, basestring):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other) + self.data)
    def __mul__(self, n):
        return self.__class__(self.data*n)
    __rmul__ = __mul__
    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self): return self.__class__(self.data.capitalize())
    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))
    def count(self, sub, start=0, end=sys.maxint):
        return self.data.count(sub, start, end)
    def decode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())
    def encode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())
    def endswith(self, suffix, start=0, end=sys.maxint):
        return self.data.endswith(suffix, start, end)
    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))
    def find(self, sub, start=0, end=sys.maxint):
        return self.data.find(sub, start, end)
    def index(self, sub, start=0, end=sys.maxint):
        return self.data.index(sub, start, end)
    def isalpha(self): return self.data.isalpha()
    def isalnum(self): return self.data.isalnum()
    def isdecimal(self): return self.data.isdecimal()
    def isdigit(self): return self.data.isdigit()
    def islower(self): return self.data.islower()
    def isnumeric(self): return self.data.isnumeric()
    def isspace(self): return self.data.isspace()
    def istitle(self): return self.data.istitle()
    def isupper(self): return self.data.isupper()
    def join(self, seq): return self.data.join(seq)
    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))
    def lower(self): return self.__class__(self.data.lower())
    def lstrip(self, chars=None): return self.__class__(self.data.lstrip(chars))
    def partition(self, sep):
        return self.data.partition(sep)
    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))
    def rfind(self, sub, start=0, end=sys.maxint):
        return self.data.rfind(sub, start, end)
    def rindex(self, sub, start=0, end=sys.maxint):
        return self.data.rindex(sub, start, end)
    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))
    def rpartition(self, sep):
        return self.data.rpartition(sep)
    def rstrip(self, chars=None): return self.__class__(self.data.rstrip(chars))
    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)
    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)
    def splitlines(self, keepends=0): return self.data.splitlines(keepends)
    def startswith(self, prefix, start=0, end=sys.maxint):
        return self.data.startswith(prefix, start, end)
    def strip(self, chars=None): return self.__class__(self.data.strip(chars))
    def swapcase(self): return self.__class__(self.data.swapcase())
    def title(self): return self.__class__(self.data.title())
    def translate(self, *args):
        return self.__class__(self.data.translate(*args))
    def upper(self): return self.__class__(self.data.upper())
    def zfill(self, width): return self.__class__(self.data.zfill(width))

class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""
    def __init__(self, string=""):
        self.data = string
    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")
    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + sub + self.data[index+1:]
    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index+1:]
    def __setslice__(self, start, end, sub):
        start = max(start, 0); end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start]+sub.data+self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start]+sub+self.data[end:]
        else:
            self.data =  self.data[:start]+str(sub)+self.data[end:]
    def __delslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]
    def immutable(self):
        return UserString(self.data)
    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, basestring):
            self.data += other
        else:
            self.data += str(other)
        return self
    def __imul__(self, n):
        self.data *= n
        return self

class String(MutableString, Union):

    _fields_ = [('raw', POINTER(c_char)),
                ('data', c_char_p)]

    def __init__(self, obj=""):
        if isinstance(obj, (str, unicode, UserString)):
            self.data = str(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(POINTER(c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj)

        # Convert from c_char_p
        elif isinstance(obj, c_char_p):
            return obj

        # Convert from POINTER(c_char)
        elif isinstance(obj, POINTER(c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(cast(obj, POINTER(c_char)))

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)
    from_param = classmethod(from_param)

def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)

# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to c_void_p.
def UNCHECKED(type):
    if (hasattr(type, "_type_") and isinstance(type._type_, str)
        and type._type_ != "P"):
        return type
    else:
        return c_void_p

# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self,func,restype,argtypes):
        self.func=func
        self.func.restype=restype
        self.argtypes=argtypes
    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func
    def __call__(self,*args):
        fixed_args=[]
        i=0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i+=1
        return self.func(*fixed_args+list(args[i:]))

# End preamble

_libs = {}
_libdirs = []

# Begin loader

# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import os.path, re, sys, glob
import platform
import ctypes
import ctypes.util

def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []

class LibraryLoader(object):
    def __init__(self):
        self.other_dirs=[]

    def load_library(self,libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self,path):
        """Given a path to a library, load it."""
        try:
            # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
            # of the default RTLD_LOCAL.  Without this, you end up with
            # libraries not being loadable, resulting in "Symbol not found"
            # errors
            if sys.platform == 'darwin':
                return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
            else:
                return ctypes.cdll.LoadLibrary(path)
        except OSError,e:
            raise ImportError(e)

    def getpaths(self,libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # FIXME / TODO return '.' and os.path.dirname(__file__)
            for path in self.getplatformpaths(libname):
                yield path

            path = ctypes.util.find_library(libname)
            if path: yield path

    def getplatformpaths(self, libname):
        return []

# Darwin (Mac OS X)

class DarwinLibraryLoader(LibraryLoader):
    name_formats = ["lib%s.dylib", "lib%s.so", "lib%s.bundle", "%s.dylib",
                "%s.so", "%s.bundle", "%s"]

    def getplatformpaths(self,libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir,name)

    def getdirs(self,libname):
        '''Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        '''

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [os.path.expanduser('~/lib'),
                                          '/usr/local/lib', '/usr/lib']

        dirs = []

        if '/' in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))

        dirs.extend(self.other_dirs)
        dirs.append(".")
        dirs.append(os.path.dirname(__file__))

        if hasattr(sys, 'frozen') and sys.frozen == 'macosx_app':
            dirs.append(os.path.join(
                os.environ['RESOURCEPATH'],
                '..',
                'Frameworks'))

        dirs.extend(dyld_fallback_library_path)

        return dirs

# Posix

class PosixLibraryLoader(LibraryLoader):
    _ld_so_cache = None

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = []
        for name in ("LD_LIBRARY_PATH",
                     "SHLIB_PATH", # HPUX
                     "LIBPATH", # OS/2, AIX
                     "LIBRARY_PATH", # BE/OS
                    ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")
        directories.append(os.path.dirname(__file__))

        try: directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError: pass

        unix_lib_dirs_list = ['/lib', '/usr/lib', '/lib64', '/usr/lib64']
        if sys.platform.startswith('linux'):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            bitage = platform.architecture()[0]
            if bitage.startswith('32'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/i386-linux-gnu', '/usr/lib/i386-linux-gnu']
            elif bitage.startswith('64'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu']
            else:
                # guess...
                unix_lib_dirs_list += glob.glob('/lib/*linux-gnu')
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r'lib(.*)\.s[ol]')
        ext_re = re.compile(r'\.s[ol]$')
        for dir in directories:
            try:
                for path in glob.glob("%s/*.s[ol]*" % dir):
                    file = os.path.basename(path)

                    # Index by filename
                    if file not in cache:
                        cache[file] = path

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname)
        if result: yield result

        path = ctypes.util.find_library(libname)
        if path: yield os.path.join("/lib",path)

# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try: return getattr(self.cdll,name)
        except AttributeError:
            try: return getattr(self.windll,name)
            except AttributeError:
                raise

class WindowsLibraryLoader(LibraryLoader):
    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll"]

    def load_library(self, libname):
        try:
            result = LibraryLoader.load_library(self, libname)
        except ImportError:
            result = None
            if os.path.sep not in libname:
                for name in self.name_formats:
                    try:
                        result = getattr(ctypes.cdll, name % libname)
                        if result:
                            break
                    except WindowsError:
                        result = None
            if result is None:
                try:
                    result = getattr(ctypes.cdll, libname)
                except WindowsError:
                    result = None
            if result is None:
                raise ImportError("%s not found." % libname)
        return result

    def load(self, path):
        return _WindowsLibrary(path)

    def getplatformpaths(self, libname):
        if os.path.sep not in libname:
            for name in self.name_formats:
                dll_in_current_dir = os.path.abspath(name % libname)
                if os.path.exists(dll_in_current_dir):
                    yield dll_in_current_dir
                path = ctypes.util.find_library(name % libname)
                if path:
                    yield path

# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin":   DarwinLibraryLoader,
    "cygwin":   WindowsLibraryLoader,
    "win32":    WindowsLibraryLoader
}

loader = loaderclass.get(sys.platform, PosixLibraryLoader)()

def add_library_search_dirs(other_dirs):
    loader.other_dirs = other_dirs

load_library = loader.load_library

del loaderclass

# End loader

add_library_search_dirs([])

# Begin libraries

_libs["flycapture-c"] = load_library("flycapture-c")

# 1 libraries
# End libraries

# No modules

BOOL = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 44

fc2Context = POINTER(None) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 64

fc2GuiContext = POINTER(None) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 70

fc2ImageImpl = POINTER(None) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 75

fc2AVIContext = POINTER(None) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 80

fc2ImageStatisticsContext = POINTER(None) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 85

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 94
class struct__fc2PGRGuid(Structure):
    pass

struct__fc2PGRGuid.__slots__ = [
    'value',
]
struct__fc2PGRGuid._fields_ = [
    ('value', c_uint * 4),
]

fc2PGRGuid = struct__fc2PGRGuid # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 94

enum__fc2Error = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_UNDEFINED = (-1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_OK = (FC2_ERROR_UNDEFINED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_FAILED = (FC2_ERROR_OK + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_NOT_IMPLEMENTED = (FC2_ERROR_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_FAILED_BUS_MASTER_CONNECTION = (FC2_ERROR_NOT_IMPLEMENTED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_NOT_CONNECTED = (FC2_ERROR_FAILED_BUS_MASTER_CONNECTION + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INIT_FAILED = (FC2_ERROR_NOT_CONNECTED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_NOT_INTITIALIZED = (FC2_ERROR_INIT_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INVALID_PARAMETER = (FC2_ERROR_NOT_INTITIALIZED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INVALID_SETTINGS = (FC2_ERROR_INVALID_PARAMETER + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INVALID_BUS_MANAGER = (FC2_ERROR_INVALID_SETTINGS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_MEMORY_ALLOCATION_FAILED = (FC2_ERROR_INVALID_BUS_MANAGER + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_LOW_LEVEL_FAILURE = (FC2_ERROR_MEMORY_ALLOCATION_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_NOT_FOUND = (FC2_ERROR_LOW_LEVEL_FAILURE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_FAILED_GUID = (FC2_ERROR_NOT_FOUND + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INVALID_PACKET_SIZE = (FC2_ERROR_FAILED_GUID + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INVALID_MODE = (FC2_ERROR_INVALID_PACKET_SIZE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_NOT_IN_FORMAT7 = (FC2_ERROR_INVALID_MODE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_NOT_SUPPORTED = (FC2_ERROR_NOT_IN_FORMAT7 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_TIMEOUT = (FC2_ERROR_NOT_SUPPORTED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_BUS_MASTER_FAILED = (FC2_ERROR_TIMEOUT + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_INVALID_GENERATION = (FC2_ERROR_BUS_MASTER_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_LUT_FAILED = (FC2_ERROR_INVALID_GENERATION + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_IIDC_FAILED = (FC2_ERROR_LUT_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_STROBE_FAILED = (FC2_ERROR_IIDC_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_TRIGGER_FAILED = (FC2_ERROR_STROBE_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_PROPERTY_FAILED = (FC2_ERROR_TRIGGER_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_PROPERTY_NOT_PRESENT = (FC2_ERROR_PROPERTY_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_REGISTER_FAILED = (FC2_ERROR_PROPERTY_NOT_PRESENT + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_READ_REGISTER_FAILED = (FC2_ERROR_REGISTER_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_WRITE_REGISTER_FAILED = (FC2_ERROR_READ_REGISTER_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_FAILED = (FC2_ERROR_WRITE_REGISTER_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_ALREADY_STARTED = (FC2_ERROR_ISOCH_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_NOT_STARTED = (FC2_ERROR_ISOCH_ALREADY_STARTED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_START_FAILED = (FC2_ERROR_ISOCH_NOT_STARTED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_RETRIEVE_BUFFER_FAILED = (FC2_ERROR_ISOCH_START_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_STOP_FAILED = (FC2_ERROR_ISOCH_RETRIEVE_BUFFER_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_SYNC_FAILED = (FC2_ERROR_ISOCH_STOP_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_ISOCH_BANDWIDTH_EXCEEDED = (FC2_ERROR_ISOCH_SYNC_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_IMAGE_CONVERSION_FAILED = (FC2_ERROR_ISOCH_BANDWIDTH_EXCEEDED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_IMAGE_LIBRARY_FAILURE = (FC2_ERROR_IMAGE_CONVERSION_FAILED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_BUFFER_TOO_SMALL = (FC2_ERROR_IMAGE_LIBRARY_FAILURE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_IMAGE_CONSISTENCY_ERROR = (FC2_ERROR_BUFFER_TOO_SMALL + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

FC2_ERROR_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

fc2Error = enum__fc2Error # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 147

enum__fc2BusCallbackType = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 156

FC2_BUS_RESET = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 156

FC2_ARRIVAL = (FC2_BUS_RESET + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 156

FC2_REMOVAL = (FC2_ARRIVAL + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 156

FC2_CALLBACK_TYPE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 156

fc2BusCallbackType = enum__fc2BusCallbackType # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 156

enum__fc2GrabMode = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 165

FC2_DROP_FRAMES = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 165

FC2_BUFFER_FRAMES = (FC2_DROP_FRAMES + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 165

FC2_UNSPECIFIED_GRAB_MODE = (FC2_BUFFER_FRAMES + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 165

FC2_GRAB_MODE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 165

fc2GrabMode = enum__fc2GrabMode # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 165

enum__fc2GrabTimeout = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 174

FC2_TIMEOUT_NONE = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 174

FC2_TIMEOUT_INFINITE = (-1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 174

FC2_TIMEOUT_UNSPECIFIED = (-2) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 174

FC2_GRAB_TIMEOUT_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 174

fc2GrabTimeout = enum__fc2GrabTimeout # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 174

enum__fc2BandwidthAllocation = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

FC2_BANDWIDTH_ALLOCATION_OFF = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

FC2_BANDWIDTH_ALLOCATION_ON = 1 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

FC2_BANDWIDTH_ALLOCATION_UNSUPPORTED = 2 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

FC2_BANDWIDTH_ALLOCATION_UNSPECIFIED = 3 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

FC2_BANDWIDTH_ALLOCATION_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

fc2BandwidthAllocation = enum__fc2BandwidthAllocation # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 184

enum__fc2InterfaceType = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

FC2_INTERFACE_IEEE1394 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

FC2_INTERFACE_USB_2 = (FC2_INTERFACE_IEEE1394 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

FC2_INTERFACE_USB_3 = (FC2_INTERFACE_USB_2 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

FC2_INTERFACE_GIGE = (FC2_INTERFACE_USB_3 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

FC2_INTERFACE_UNKNOWN = (FC2_INTERFACE_GIGE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

FC2_INTERFACE_TYPE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

fc2InterfaceType = enum__fc2InterfaceType # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 195

enum__fc2DriverType = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_1394_CAM = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_1394_PRO = (FC2_DRIVER_1394_CAM + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_1394_JUJU = (FC2_DRIVER_1394_PRO + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_1394_VIDEO1394 = (FC2_DRIVER_1394_JUJU + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_1394_RAW1394 = (FC2_DRIVER_1394_VIDEO1394 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_USB_NONE = (FC2_DRIVER_1394_RAW1394 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_USB_CAM = (FC2_DRIVER_USB_NONE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_USB3_PRO = (FC2_DRIVER_USB_CAM + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_GIGE_NONE = (FC2_DRIVER_USB3_PRO + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_GIGE_FILTER = (FC2_DRIVER_GIGE_NONE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_GIGE_PRO = (FC2_DRIVER_GIGE_FILTER + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_UNKNOWN = (-1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

FC2_DRIVER_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

fc2DriverType = enum__fc2DriverType # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 213

enum__fc2PropertyType = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_BRIGHTNESS = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_AUTO_EXPOSURE = (FC2_BRIGHTNESS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_SHARPNESS = (FC2_AUTO_EXPOSURE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_WHITE_BALANCE = (FC2_SHARPNESS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_HUE = (FC2_WHITE_BALANCE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_SATURATION = (FC2_HUE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_GAMMA = (FC2_SATURATION + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_IRIS = (FC2_GAMMA + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_FOCUS = (FC2_IRIS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_ZOOM = (FC2_FOCUS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_PAN = (FC2_ZOOM + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_TILT = (FC2_PAN + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_SHUTTER = (FC2_TILT + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_GAIN = (FC2_SHUTTER + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_TRIGGER_MODE = (FC2_GAIN + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_TRIGGER_DELAY = (FC2_TRIGGER_MODE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_FRAME_RATE = (FC2_TRIGGER_DELAY + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_TEMPERATURE = (FC2_FRAME_RATE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_UNSPECIFIED_PROPERTY_TYPE = (FC2_TEMPERATURE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

FC2_PROPERTY_TYPE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

fc2PropertyType = enum__fc2PropertyType # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 238

enum__fc2FrameRate = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_1_875 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_3_75 = (FC2_FRAMERATE_1_875 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_7_5 = (FC2_FRAMERATE_3_75 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_15 = (FC2_FRAMERATE_7_5 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_30 = (FC2_FRAMERATE_15 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_60 = (FC2_FRAMERATE_30 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_120 = (FC2_FRAMERATE_60 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_240 = (FC2_FRAMERATE_120 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_FORMAT7 = (FC2_FRAMERATE_240 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_NUM_FRAMERATES = (FC2_FRAMERATE_FORMAT7 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

FC2_FRAMERATE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

fc2FrameRate = enum__fc2FrameRate # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 254

enum__fc2VideoMode = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_160x120YUV444 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_320x240YUV422 = (FC2_VIDEOMODE_160x120YUV444 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_640x480YUV411 = (FC2_VIDEOMODE_320x240YUV422 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_640x480YUV422 = (FC2_VIDEOMODE_640x480YUV411 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_640x480RGB = (FC2_VIDEOMODE_640x480YUV422 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_640x480Y8 = (FC2_VIDEOMODE_640x480RGB + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_640x480Y16 = (FC2_VIDEOMODE_640x480Y8 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_800x600YUV422 = (FC2_VIDEOMODE_640x480Y16 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_800x600RGB = (FC2_VIDEOMODE_800x600YUV422 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_800x600Y8 = (FC2_VIDEOMODE_800x600RGB + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_800x600Y16 = (FC2_VIDEOMODE_800x600Y8 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1024x768YUV422 = (FC2_VIDEOMODE_800x600Y16 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1024x768RGB = (FC2_VIDEOMODE_1024x768YUV422 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1024x768Y8 = (FC2_VIDEOMODE_1024x768RGB + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1024x768Y16 = (FC2_VIDEOMODE_1024x768Y8 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1280x960YUV422 = (FC2_VIDEOMODE_1024x768Y16 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1280x960RGB = (FC2_VIDEOMODE_1280x960YUV422 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1280x960Y8 = (FC2_VIDEOMODE_1280x960RGB + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1280x960Y16 = (FC2_VIDEOMODE_1280x960Y8 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1600x1200YUV422 = (FC2_VIDEOMODE_1280x960Y16 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1600x1200RGB = (FC2_VIDEOMODE_1600x1200YUV422 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1600x1200Y8 = (FC2_VIDEOMODE_1600x1200RGB + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_1600x1200Y16 = (FC2_VIDEOMODE_1600x1200Y8 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_FORMAT7 = (FC2_VIDEOMODE_1600x1200Y16 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_NUM_VIDEOMODES = (FC2_VIDEOMODE_FORMAT7 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

FC2_VIDEOMODE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

fc2VideoMode = enum__fc2VideoMode # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 285

enum__fc2Mode = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_0 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_1 = (FC2_MODE_0 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_2 = (FC2_MODE_1 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_3 = (FC2_MODE_2 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_4 = (FC2_MODE_3 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_5 = (FC2_MODE_4 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_6 = (FC2_MODE_5 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_7 = (FC2_MODE_6 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_8 = (FC2_MODE_7 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_9 = (FC2_MODE_8 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_10 = (FC2_MODE_9 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_11 = (FC2_MODE_10 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_12 = (FC2_MODE_11 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_13 = (FC2_MODE_12 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_14 = (FC2_MODE_13 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_15 = (FC2_MODE_14 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_16 = (FC2_MODE_15 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_17 = (FC2_MODE_16 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_18 = (FC2_MODE_17 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_19 = (FC2_MODE_18 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_20 = (FC2_MODE_19 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_21 = (FC2_MODE_20 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_22 = (FC2_MODE_21 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_23 = (FC2_MODE_22 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_24 = (FC2_MODE_23 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_25 = (FC2_MODE_24 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_26 = (FC2_MODE_25 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_27 = (FC2_MODE_26 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_28 = (FC2_MODE_27 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_29 = (FC2_MODE_28 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_30 = (FC2_MODE_29 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_31 = (FC2_MODE_30 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_NUM_MODES = (FC2_MODE_31 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

FC2_MODE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

fc2Mode = enum__fc2Mode # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 324

enum__fc2PixelFormat = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_MONO8 = 2147483648 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_411YUV8 = 1073741824 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_422YUV8 = 536870912 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_444YUV8 = 268435456 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RGB8 = 134217728 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_MONO16 = 67108864 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RGB16 = 33554432 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_S_MONO16 = 16777216 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_S_RGB16 = 8388608 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RAW8 = 4194304 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RAW16 = 2097152 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_MONO12 = 1048576 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RAW12 = 524288 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_BGR = 2147483656 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_BGRU = 1073741832 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RGB = FC2_PIXEL_FORMAT_RGB8 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_RGBU = 1073741826 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_BGR16 = 33554433 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_BGRU16 = 33554434 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_PIXEL_FORMAT_422YUV8_JPEG = 1073741825 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_NUM_PIXEL_FORMATS = 20 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

FC2_UNSPECIFIED_PIXEL_FORMAT = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

fc2PixelFormat = enum__fc2PixelFormat # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 351

enum__fc2BusSpeed = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S100 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S200 = (FC2_BUSSPEED_S100 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S400 = (FC2_BUSSPEED_S200 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S480 = (FC2_BUSSPEED_S400 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S800 = (FC2_BUSSPEED_S480 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S1600 = (FC2_BUSSPEED_S800 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S3200 = (FC2_BUSSPEED_S1600 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S5000 = (FC2_BUSSPEED_S3200 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_10BASE_T = (FC2_BUSSPEED_S5000 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_100BASE_T = (FC2_BUSSPEED_10BASE_T + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_1000BASE_T = (FC2_BUSSPEED_100BASE_T + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_10000BASE_T = (FC2_BUSSPEED_1000BASE_T + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_S_FASTEST = (FC2_BUSSPEED_10000BASE_T + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_ANY = (FC2_BUSSPEED_S_FASTEST + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_SPEED_UNKNOWN = (-1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

FC2_BUSSPEED_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

fc2BusSpeed = enum__fc2BusSpeed # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 372

enum__fc2PCIeBusSpeed = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 380

FC2_PCIE_BUSSPEED_2_5 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 380

FC2_PCIE_BUSSPEED_5_0 = (FC2_PCIE_BUSSPEED_2_5 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 380

FC2_PCIE_BUSSPEED_UNKNOWN = (-1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 380

FC2_PCIE_BUSSPEED_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 380

fc2PCIeBusSpeed = enum__fc2PCIeBusSpeed # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 380

enum__fc2ColorProcessingAlgorithm = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_DEFAULT = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_NO_COLOR_PROCESSING = (FC2_DEFAULT + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_NEAREST_NEIGHBOR_FAST = (FC2_NO_COLOR_PROCESSING + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_EDGE_SENSING = (FC2_NEAREST_NEIGHBOR_FAST + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_HQ_LINEAR = (FC2_EDGE_SENSING + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_RIGOROUS = (FC2_HQ_LINEAR + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_IPP = (FC2_RIGOROUS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_DIRECTIONAL = (FC2_IPP + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

FC2_COLOR_PROCESSING_ALGORITHM_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

fc2ColorProcessingAlgorithm = enum__fc2ColorProcessingAlgorithm # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 394

enum__fc2BayerTileFormat = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

FC2_BT_NONE = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

FC2_BT_RGGB = (FC2_BT_NONE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

FC2_BT_GRBG = (FC2_BT_RGGB + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

FC2_BT_GBRG = (FC2_BT_GRBG + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

FC2_BT_BGGR = (FC2_BT_GBRG + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

FC2_BT_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

fc2BayerTileFormat = enum__fc2BayerTileFormat # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 405

enum__fc2ImageFileFormat = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_FROM_FILE_EXT = (-1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_PGM = (FC2_FROM_FILE_EXT + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_PPM = (FC2_PGM + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_BMP = (FC2_PPM + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_JPEG = (FC2_BMP + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_JPEG2000 = (FC2_JPEG + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_TIFF = (FC2_JPEG2000 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_PNG = (FC2_TIFF + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_RAW = (FC2_PNG + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

FC2_IMAGE_FILE_FORMAT_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

fc2ImageFileFormat = enum__fc2ImageFileFormat # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 420

enum__fc2GigEPropertyType = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 427

FC2_HEARTBEAT = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 427

FC2_HEARTBEAT_TIMEOUT = (FC2_HEARTBEAT + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 427

fc2GigEPropertyType = enum__fc2GigEPropertyType # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 427

enum__fc2StatisticsChannel = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_GREY = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_RED = (FC2_STATISTICS_GREY + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_GREEN = (FC2_STATISTICS_RED + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_BLUE = (FC2_STATISTICS_GREEN + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_HUE = (FC2_STATISTICS_BLUE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_SATURATION = (FC2_STATISTICS_HUE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_LIGHTNESS = (FC2_STATISTICS_SATURATION + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

FC2_STATISTICS_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

fc2StatisticsChannel = enum__fc2StatisticsChannel # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 439

enum__fc2OSType = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_WINDOWS_X86 = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_WINDOWS_X64 = (FC2_WINDOWS_X86 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_LINUX_X86 = (FC2_WINDOWS_X64 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_LINUX_X64 = (FC2_LINUX_X86 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_MAC = (FC2_LINUX_X64 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_UNKNOWN_OS = (FC2_MAC + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

FC2_OSTYPE_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

fc2OSType = enum__fc2OSType # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 451

enum__fc2ByteOrder = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 458

FC2_BYTE_ORDER_LITTLE_ENDIAN = 0 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 458

FC2_BYTE_ORDER_BIG_ENDIAN = (FC2_BYTE_ORDER_LITTLE_ENDIAN + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 458

FC2_BYTE_ORDER_FORCE_32BITS = 2147483647 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 458

fc2ByteOrder = enum__fc2ByteOrder # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 458

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 482
class struct__fc2Image(Structure):
    pass

struct__fc2Image.__slots__ = [
    'rows',
    'cols',
    'stride',
    'pData',
    'dataSize',
    'receivedDataSize',
    'format',
    'bayerFormat',
    'imageImpl',
]
struct__fc2Image._fields_ = [
    ('rows', c_uint),
    ('cols', c_uint),
    ('stride', c_uint),
    ('pData', POINTER(c_ubyte)),
    ('dataSize', c_uint),
    ('receivedDataSize', c_uint),
    ('format', fc2PixelFormat),
    ('bayerFormat', fc2BayerTileFormat),
    ('imageImpl', fc2ImageImpl),
]

fc2Image = struct__fc2Image # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 482

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 499
class struct__fc2SystemInfo(Structure):
    pass

struct__fc2SystemInfo.__slots__ = [
    'osType',
    'osDescription',
    'byteOrder',
    'sysMemSize',
    'cpuDescription',
    'numCpuCores',
    'driverList',
    'libraryList',
    'gpuDescription',
    'screenWidth',
    'screenHeight',
    'reserved',
]
struct__fc2SystemInfo._fields_ = [
    ('osType', fc2OSType),
    ('osDescription', c_char * 512),
    ('byteOrder', fc2ByteOrder),
    ('sysMemSize', c_size_t),
    ('cpuDescription', c_char * 512),
    ('numCpuCores', c_size_t),
    ('driverList', c_char * 512),
    ('libraryList', c_char * 512),
    ('gpuDescription', c_char * 512),
    ('screenWidth', c_size_t),
    ('screenHeight', c_size_t),
    ('reserved', c_uint * 16),
]

fc2SystemInfo = struct__fc2SystemInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 499

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 507
class struct__fc2Version(Structure):
    pass

struct__fc2Version.__slots__ = [
    'major',
    'minor',
    'type',
    'build',
]
struct__fc2Version._fields_ = [
    ('major', c_uint),
    ('minor', c_uint),
    ('type', c_uint),
    ('build', c_uint),
]

fc2Version = struct__fc2Version # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 507

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 523
class struct__fc2Config(Structure):
    pass

struct__fc2Config.__slots__ = [
    'numBuffers',
    'numImageNotifications',
    'minNumImageNotifications',
    'grabTimeout',
    'grabMode',
    'isochBusSpeed',
    'asyncBusSpeed',
    'bandwidthAllocation',
    'registerTimeoutRetries',
    'registerTimeout',
    'reserved',
]
struct__fc2Config._fields_ = [
    ('numBuffers', c_uint),
    ('numImageNotifications', c_uint),
    ('minNumImageNotifications', c_uint),
    ('grabTimeout', c_int),
    ('grabMode', fc2GrabMode),
    ('isochBusSpeed', fc2BusSpeed),
    ('asyncBusSpeed', fc2BusSpeed),
    ('bandwidthAllocation', fc2BandwidthAllocation),
    ('registerTimeoutRetries', c_uint),
    ('registerTimeout', c_uint),
    ('reserved', c_uint * 16),
]

fc2Config = struct__fc2Config # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 523

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 543
class struct__fc2PropertyInfo(Structure):
    pass

struct__fc2PropertyInfo.__slots__ = [
    'type',
    'present',
    'autoSupported',
    'manualSupported',
    'onOffSupported',
    'onePushSupported',
    'absValSupported',
    'readOutSupported',
    'min',
    'max',
    'absMin',
    'absMax',
    'pUnits',
    'pUnitAbbr',
    'reserved',
]
struct__fc2PropertyInfo._fields_ = [
    ('type', fc2PropertyType),
    ('present', BOOL),
    ('autoSupported', BOOL),
    ('manualSupported', BOOL),
    ('onOffSupported', BOOL),
    ('onePushSupported', BOOL),
    ('absValSupported', BOOL),
    ('readOutSupported', BOOL),
    ('min', c_uint),
    ('max', c_uint),
    ('absMin', c_float),
    ('absMax', c_float),
    ('pUnits', c_char * 512),
    ('pUnitAbbr', c_char * 512),
    ('reserved', c_uint * 8),
]

fc2PropertyInfo = struct__fc2PropertyInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 543

fc2TriggerDelayInfo = struct__fc2PropertyInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 543

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 562
class struct__Property(Structure):
    pass

struct__Property.__slots__ = [
    'type',
    'present',
    'absControl',
    'onePush',
    'onOff',
    'autoManualMode',
    'valueA',
    'valueB',
    'absValue',
    'reserved',
]
struct__Property._fields_ = [
    ('type', fc2PropertyType),
    ('present', BOOL),
    ('absControl', BOOL),
    ('onePush', BOOL),
    ('onOff', BOOL),
    ('autoManualMode', BOOL),
    ('valueA', c_uint),
    ('valueB', c_uint),
    ('absValue', c_float),
    ('reserved', c_uint * 8),
]

fc2Property = struct__Property # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 562

fc2TriggerDelay = struct__Property # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 562

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 576
class struct__fc2TriggerModeInfo(Structure):
    pass

struct__fc2TriggerModeInfo.__slots__ = [
    'present',
    'readOutSupported',
    'onOffSupported',
    'polaritySupported',
    'valueReadable',
    'sourceMask',
    'softwareTriggerSupported',
    'modeMask',
    'reserved',
]
struct__fc2TriggerModeInfo._fields_ = [
    ('present', BOOL),
    ('readOutSupported', BOOL),
    ('onOffSupported', BOOL),
    ('polaritySupported', BOOL),
    ('valueReadable', BOOL),
    ('sourceMask', c_uint),
    ('softwareTriggerSupported', BOOL),
    ('modeMask', c_uint),
    ('reserved', c_uint * 8),
]

fc2TriggerModeInfo = struct__fc2TriggerModeInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 576

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 587
class struct__fc2TriggerMode(Structure):
    pass

struct__fc2TriggerMode.__slots__ = [
    'onOff',
    'polarity',
    'source',
    'mode',
    'parameter',
    'reserved',
]
struct__fc2TriggerMode._fields_ = [
    ('onOff', BOOL),
    ('polarity', c_uint),
    ('source', c_uint),
    ('mode', c_uint),
    ('parameter', c_uint),
    ('reserved', c_uint * 8),
]

fc2TriggerMode = struct__fc2TriggerMode # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 587

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 600
class struct__fc2StrobeInfo(Structure):
    pass

struct__fc2StrobeInfo.__slots__ = [
    'source',
    'present',
    'readOutSupported',
    'onOffSupported',
    'polaritySupported',
    'minValue',
    'maxValue',
    'reserved',
]
struct__fc2StrobeInfo._fields_ = [
    ('source', c_uint),
    ('present', BOOL),
    ('readOutSupported', BOOL),
    ('onOffSupported', BOOL),
    ('polaritySupported', BOOL),
    ('minValue', c_float),
    ('maxValue', c_float),
    ('reserved', c_uint * 8),
]

fc2StrobeInfo = struct__fc2StrobeInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 600

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 611
class struct__fc2StrobeControl(Structure):
    pass

struct__fc2StrobeControl.__slots__ = [
    'source',
    'onOff',
    'polarity',
    'delay',
    'duration',
    'reserved',
]
struct__fc2StrobeControl._fields_ = [
    ('source', c_uint),
    ('onOff', BOOL),
    ('polarity', c_uint),
    ('delay', c_float),
    ('duration', c_float),
    ('reserved', c_uint * 8),
]

fc2StrobeControl = struct__fc2StrobeControl # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 611

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 623
class struct__fc2Format7ImageSettings(Structure):
    pass

struct__fc2Format7ImageSettings.__slots__ = [
    'mode',
    'offsetX',
    'offsetY',
    'width',
    'height',
    'pixelFormat',
    'reserved',
]
struct__fc2Format7ImageSettings._fields_ = [
    ('mode', fc2Mode),
    ('offsetX', c_uint),
    ('offsetY', c_uint),
    ('width', c_uint),
    ('height', c_uint),
    ('pixelFormat', fc2PixelFormat),
    ('reserved', c_uint * 8),
]

fc2Format7ImageSettings = struct__fc2Format7ImageSettings # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 623

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 643
class struct__fc2Format7Info(Structure):
    pass

struct__fc2Format7Info.__slots__ = [
    'mode',
    'maxWidth',
    'maxHeight',
    'offsetHStepSize',
    'offsetVStepSize',
    'imageHStepSize',
    'imageVStepSize',
    'pixelFormatBitField',
    'vendorPixelFormatBitField',
    'packetSize',
    'minPacketSize',
    'maxPacketSize',
    'percentage',
    'reserved',
]
struct__fc2Format7Info._fields_ = [
    ('mode', fc2Mode),
    ('maxWidth', c_uint),
    ('maxHeight', c_uint),
    ('offsetHStepSize', c_uint),
    ('offsetVStepSize', c_uint),
    ('imageHStepSize', c_uint),
    ('imageVStepSize', c_uint),
    ('pixelFormatBitField', c_uint),
    ('vendorPixelFormatBitField', c_uint),
    ('packetSize', c_uint),
    ('minPacketSize', c_uint),
    ('maxPacketSize', c_uint),
    ('percentage', c_float),
    ('reserved', c_uint * 16),
]

fc2Format7Info = struct__fc2Format7Info # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 643

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 652
class struct__fc2Format7PacketInfo(Structure):
    pass

struct__fc2Format7PacketInfo.__slots__ = [
    'recommendedBytesPerPacket',
    'maxBytesPerPacket',
    'unitBytesPerPacket',
    'reserved',
]
struct__fc2Format7PacketInfo._fields_ = [
    ('recommendedBytesPerPacket', c_uint),
    ('maxBytesPerPacket', c_uint),
    ('unitBytesPerPacket', c_uint),
    ('reserved', c_uint * 8),
]

fc2Format7PacketInfo = struct__fc2Format7PacketInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 652

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 657
class struct__fc2IPAddress(Structure):
    pass

struct__fc2IPAddress.__slots__ = [
    'octets',
]
struct__fc2IPAddress._fields_ = [
    ('octets', c_ubyte * 4),
]

fc2IPAddress = struct__fc2IPAddress # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 657

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 662
class struct__fc2MACAddress(Structure):
    pass

struct__fc2MACAddress.__slots__ = [
    'octets',
]
struct__fc2MACAddress._fields_ = [
    ('octets', c_ubyte * 6),
]

fc2MACAddress = struct__fc2MACAddress # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 662

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 674
class struct__fc2GigEProperty(Structure):
    pass

struct__fc2GigEProperty.__slots__ = [
    'propType',
    'isReadable',
    'isWritable',
    'min',
    'max',
    'value',
    'reserved',
]
struct__fc2GigEProperty._fields_ = [
    ('propType', fc2GigEPropertyType),
    ('isReadable', BOOL),
    ('isWritable', BOOL),
    ('min', c_uint),
    ('max', c_uint),
    ('value', c_uint),
    ('reserved', c_uint * 8),
]

fc2GigEProperty = struct__fc2GigEProperty # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 674

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 687
class struct__fc2GigEStreamChannel(Structure):
    pass

struct__fc2GigEStreamChannel.__slots__ = [
    'networkInterfaceIndex',
    'hostPost',
    'doNotFragment',
    'packetSize',
    'interPacketDelay',
    'destinationIpAddress',
    'sourcePort',
    'reserved',
]
struct__fc2GigEStreamChannel._fields_ = [
    ('networkInterfaceIndex', c_uint),
    ('hostPost', c_uint),
    ('doNotFragment', BOOL),
    ('packetSize', c_uint),
    ('interPacketDelay', c_uint),
    ('destinationIpAddress', fc2IPAddress),
    ('sourcePort', c_uint),
    ('reserved', c_uint * 8),
]

fc2GigEStreamChannel = struct__fc2GigEStreamChannel # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 687

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 699
class struct__fc2GigEConfig(Structure):
    pass

struct__fc2GigEConfig.__slots__ = [
    'enablePacketResend',
    'timeoutForPacketResend',
    'maxPacketsToResend',
    'reserved',
]
struct__fc2GigEConfig._fields_ = [
    ('enablePacketResend', BOOL),
    ('timeoutForPacketResend', c_uint),
    ('maxPacketsToResend', c_uint),
    ('reserved', c_uint * 8),
]

fc2GigEConfig = struct__fc2GigEConfig # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 699

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 713
class struct__fc2GigEImageSettingsInfo(Structure):
    pass

struct__fc2GigEImageSettingsInfo.__slots__ = [
    'maxWidth',
    'maxHeight',
    'offsetHStepSize',
    'offsetVStepSize',
    'imageHStepSize',
    'imageVStepSize',
    'pixelFormatBitField',
    'vendorPixelFormatBitField',
    'reserved',
]
struct__fc2GigEImageSettingsInfo._fields_ = [
    ('maxWidth', c_uint),
    ('maxHeight', c_uint),
    ('offsetHStepSize', c_uint),
    ('offsetVStepSize', c_uint),
    ('imageHStepSize', c_uint),
    ('imageVStepSize', c_uint),
    ('pixelFormatBitField', c_uint),
    ('vendorPixelFormatBitField', c_uint),
    ('reserved', c_uint * 16),
]

fc2GigEImageSettingsInfo = struct__fc2GigEImageSettingsInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 713

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 724
class struct__fc2GigEImageSettings(Structure):
    pass

struct__fc2GigEImageSettings.__slots__ = [
    'offsetX',
    'offsetY',
    'width',
    'height',
    'pixelFormat',
    'reserved',
]
struct__fc2GigEImageSettings._fields_ = [
    ('offsetX', c_uint),
    ('offsetY', c_uint),
    ('width', c_uint),
    ('height', c_uint),
    ('pixelFormat', fc2PixelFormat),
    ('reserved', c_uint * 8),
]

fc2GigEImageSettings = struct__fc2GigEImageSettings # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 724

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 735
class struct__fc2TimeStamp(Structure):
    pass

struct__fc2TimeStamp.__slots__ = [
    'seconds',
    'microSeconds',
    'cycleSeconds',
    'cycleCount',
    'cycleOffset',
    'reserved',
]
struct__fc2TimeStamp._fields_ = [
    ('seconds', c_longlong),
    ('microSeconds', c_uint),
    ('cycleSeconds', c_uint),
    ('cycleCount', c_uint),
    ('cycleOffset', c_uint),
    ('reserved', c_uint * 8),
]

fc2TimeStamp = struct__fc2TimeStamp # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 735

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 752
class struct__fc2ConfigROM(Structure):
    pass

struct__fc2ConfigROM.__slots__ = [
    'nodeVendorId',
    'chipIdHi',
    'chipIdLo',
    'unitSpecId',
    'unitSWVer',
    'unitSubSWVer',
    'vendorUniqueInfo_0',
    'vendorUniqueInfo_1',
    'vendorUniqueInfo_2',
    'vendorUniqueInfo_3',
    'pszKeyword',
    'reserved',
]
struct__fc2ConfigROM._fields_ = [
    ('nodeVendorId', c_uint),
    ('chipIdHi', c_uint),
    ('chipIdLo', c_uint),
    ('unitSpecId', c_uint),
    ('unitSWVer', c_uint),
    ('unitSubSWVer', c_uint),
    ('vendorUniqueInfo_0', c_uint),
    ('vendorUniqueInfo_1', c_uint),
    ('vendorUniqueInfo_2', c_uint),
    ('vendorUniqueInfo_3', c_uint),
    ('pszKeyword', c_char * 512),
    ('reserved', c_uint * 16),
]

fc2ConfigROM = struct__fc2ConfigROM # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 752

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 797
class struct__fc2CameraInfo(Structure):
    pass

struct__fc2CameraInfo.__slots__ = [
    'serialNumber',
    'interfaceType',
    'driverType',
    'isColorCamera',
    'modelName',
    'vendorName',
    'sensorInfo',
    'sensorResolution',
    'driverName',
    'firmwareVersion',
    'firmwareBuildTime',
    'maximumBusSpeed',
    'pcieBusSpeed',
    'bayerTileFormat',
    'busNumber',
    'nodeNumber',
    'iidcVer',
    'configROM',
    'gigEMajorVersion',
    'gigEMinorVersion',
    'userDefinedName',
    'xmlURL1',
    'xmlURL2',
    'macAddress',
    'ipAddress',
    'subnetMask',
    'defaultGateway',
    'ccpStatus',
    'applicationIPAddress',
    'applicationPort',
    'reserved',
]
struct__fc2CameraInfo._fields_ = [
    ('serialNumber', c_uint),
    ('interfaceType', fc2InterfaceType),
    ('driverType', fc2DriverType),
    ('isColorCamera', BOOL),
    ('modelName', c_char * 512),
    ('vendorName', c_char * 512),
    ('sensorInfo', c_char * 512),
    ('sensorResolution', c_char * 512),
    ('driverName', c_char * 512),
    ('firmwareVersion', c_char * 512),
    ('firmwareBuildTime', c_char * 512),
    ('maximumBusSpeed', fc2BusSpeed),
    ('pcieBusSpeed', fc2PCIeBusSpeed),
    ('bayerTileFormat', fc2BayerTileFormat),
    ('busNumber', c_ushort),
    ('nodeNumber', c_ushort),
    ('iidcVer', c_uint),
    ('configROM', fc2ConfigROM),
    ('gigEMajorVersion', c_uint),
    ('gigEMinorVersion', c_uint),
    ('userDefinedName', c_char * 512),
    ('xmlURL1', c_char * 512),
    ('xmlURL2', c_char * 512),
    ('macAddress', fc2MACAddress),
    ('ipAddress', fc2IPAddress),
    ('subnetMask', fc2IPAddress),
    ('defaultGateway', fc2IPAddress),
    ('ccpStatus', c_uint),
    ('applicationIPAddress', c_uint),
    ('applicationPort', c_uint),
    ('reserved', c_uint * 16),
]

fc2CameraInfo = struct__fc2CameraInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 797

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 804
class struct__fc2EmbeddedImageInfoProperty(Structure):
    pass

struct__fc2EmbeddedImageInfoProperty.__slots__ = [
    'available',
    'onOff',
]
struct__fc2EmbeddedImageInfoProperty._fields_ = [
    ('available', BOOL),
    ('onOff', BOOL),
]

fc2EmbeddedImageInfoProperty = struct__fc2EmbeddedImageInfoProperty # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 804

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 819
class struct__fc2EmbeddedImageInfo(Structure):
    pass

struct__fc2EmbeddedImageInfo.__slots__ = [
    'timestamp',
    'gain',
    'shutter',
    'brightness',
    'exposure',
    'whiteBalance',
    'frameCounter',
    'strobePattern',
    'GPIOPinState',
    'ROIPosition',
]
struct__fc2EmbeddedImageInfo._fields_ = [
    ('timestamp', fc2EmbeddedImageInfoProperty),
    ('gain', fc2EmbeddedImageInfoProperty),
    ('shutter', fc2EmbeddedImageInfoProperty),
    ('brightness', fc2EmbeddedImageInfoProperty),
    ('exposure', fc2EmbeddedImageInfoProperty),
    ('whiteBalance', fc2EmbeddedImageInfoProperty),
    ('frameCounter', fc2EmbeddedImageInfoProperty),
    ('strobePattern', fc2EmbeddedImageInfoProperty),
    ('GPIOPinState', fc2EmbeddedImageInfoProperty),
    ('ROIPosition', fc2EmbeddedImageInfoProperty),
]

fc2EmbeddedImageInfo = struct__fc2EmbeddedImageInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 819

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 835
class struct__fc2ImageMetadata(Structure):
    pass

struct__fc2ImageMetadata.__slots__ = [
    'embeddedTimeStamp',
    'embeddedGain',
    'embeddedShutter',
    'embeddedBrightness',
    'embeddedExposure',
    'embeddedWhiteBalance',
    'embeddedFrameCounter',
    'embeddedStrobePattern',
    'embeddedGPIOPinState',
    'embeddedROIPosition',
    'reserved',
]
struct__fc2ImageMetadata._fields_ = [
    ('embeddedTimeStamp', c_uint),
    ('embeddedGain', c_uint),
    ('embeddedShutter', c_uint),
    ('embeddedBrightness', c_uint),
    ('embeddedExposure', c_uint),
    ('embeddedWhiteBalance', c_uint),
    ('embeddedFrameCounter', c_uint),
    ('embeddedStrobePattern', c_uint),
    ('embeddedGPIOPinState', c_uint),
    ('embeddedROIPosition', c_uint),
    ('reserved', c_uint * 31),
]

fc2ImageMetadata = struct__fc2ImageMetadata # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 835

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 848
class struct__fc2LUTData(Structure):
    pass

struct__fc2LUTData.__slots__ = [
    'supported',
    'enabled',
    'numBanks',
    'numChannels',
    'inputBitDepth',
    'outputBitDepth',
    'numEntries',
    'reserved',
]
struct__fc2LUTData._fields_ = [
    ('supported', BOOL),
    ('enabled', BOOL),
    ('numBanks', c_uint),
    ('numChannels', c_uint),
    ('inputBitDepth', c_uint),
    ('outputBitDepth', c_uint),
    ('numEntries', c_uint),
    ('reserved', c_uint * 8),
]

fc2LUTData = struct__fc2LUTData # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 848

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 856
class struct__fc2PNGOption(Structure):
    pass

struct__fc2PNGOption.__slots__ = [
    'interlaced',
    'compressionLevel',
    'reserved',
]
struct__fc2PNGOption._fields_ = [
    ('interlaced', BOOL),
    ('compressionLevel', c_uint),
    ('reserved', c_uint * 16),
]

fc2PNGOption = struct__fc2PNGOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 856

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 863
class struct__fc2PPMOption(Structure):
    pass

struct__fc2PPMOption.__slots__ = [
    'binaryFile',
    'reserved',
]
struct__fc2PPMOption._fields_ = [
    ('binaryFile', BOOL),
    ('reserved', c_uint * 16),
]

fc2PPMOption = struct__fc2PPMOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 863

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 870
class struct__fc2PGMOption(Structure):
    pass

struct__fc2PGMOption.__slots__ = [
    'binaryFile',
    'reserved',
]
struct__fc2PGMOption._fields_ = [
    ('binaryFile', BOOL),
    ('reserved', c_uint * 16),
]

fc2PGMOption = struct__fc2PGMOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 870

enum__fc2TIFFCompressionMethod = c_int # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_NONE = 1 # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_PACKBITS = (FC2_TIFF_NONE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_DEFLATE = (FC2_TIFF_PACKBITS + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_ADOBE_DEFLATE = (FC2_TIFF_DEFLATE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_CCITTFAX3 = (FC2_TIFF_ADOBE_DEFLATE + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_CCITTFAX4 = (FC2_TIFF_CCITTFAX3 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_LZW = (FC2_TIFF_CCITTFAX4 + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

FC2_TIFF_JPEG = (FC2_TIFF_LZW + 1) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

fc2TIFFCompressionMethod = enum__fc2TIFFCompressionMethod # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 882

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 889
class struct__fc2TIFFOption(Structure):
    pass

struct__fc2TIFFOption.__slots__ = [
    'compression',
    'reserved',
]
struct__fc2TIFFOption._fields_ = [
    ('compression', fc2TIFFCompressionMethod),
    ('reserved', c_uint * 16),
]

fc2TIFFOption = struct__fc2TIFFOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 889

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 897
class struct__fc2JPEGOption(Structure):
    pass

struct__fc2JPEGOption.__slots__ = [
    'progressive',
    'quality',
    'reserved',
]
struct__fc2JPEGOption._fields_ = [
    ('progressive', BOOL),
    ('quality', c_uint),
    ('reserved', c_uint * 16),
]

fc2JPEGOption = struct__fc2JPEGOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 897

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 903
class struct__fc2JPG2Option(Structure):
    pass

struct__fc2JPG2Option.__slots__ = [
    'quality',
    'reserved',
]
struct__fc2JPG2Option._fields_ = [
    ('quality', c_uint),
    ('reserved', c_uint * 16),
]

fc2JPG2Option = struct__fc2JPG2Option # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 903

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 910
class struct__fc2AVIOption(Structure):
    pass

struct__fc2AVIOption.__slots__ = [
    'frameRate',
    'reserved',
]
struct__fc2AVIOption._fields_ = [
    ('frameRate', c_float),
    ('reserved', c_uint * 256),
]

fc2AVIOption = struct__fc2AVIOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 910

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 918
class struct__fc2MJPGOption(Structure):
    pass

struct__fc2MJPGOption.__slots__ = [
    'frameRate',
    'quality',
    'reserved',
]
struct__fc2MJPGOption._fields_ = [
    ('frameRate', c_float),
    ('quality', c_uint),
    ('reserved', c_uint * 256),
]

fc2MJPGOption = struct__fc2MJPGOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 918

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 929
class struct__fc2H264Option(Structure):
    pass

struct__fc2H264Option.__slots__ = [
    'frameRate',
    'width',
    'height',
    'bitrate',
    'reserved',
]
struct__fc2H264Option._fields_ = [
    ('frameRate', c_float),
    ('width', c_uint),
    ('height', c_uint),
    ('bitrate', c_uint),
    ('reserved', c_uint * 256),
]

fc2H264Option = struct__fc2H264Option # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 929

fc2CallbackHandle = POINTER(None) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 936

fc2BusEventCallback = CFUNCTYPE(UNCHECKED(None), POINTER(None), c_uint) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 937

fc2ImageEventCallback = CFUNCTYPE(UNCHECKED(None), POINTER(fc2Image), POINTER(None)) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 938

fc2AsyncCommandCallback = CFUNCTYPE(UNCHECKED(None), fc2Error, POINTER(None)) # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 939

# /usr/include/flycapture/C/FlyCapture2_C.h: 48
if hasattr(_libs['flycapture-c'], 'fc2CreateContext'):
    fc2CreateContext = _libs['flycapture-c'].fc2CreateContext
    fc2CreateContext.argtypes = [POINTER(fc2Context)]
    fc2CreateContext.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 61
if hasattr(_libs['flycapture-c'], 'fc2CreateGigEContext'):
    fc2CreateGigEContext = _libs['flycapture-c'].fc2CreateGigEContext
    fc2CreateGigEContext.argtypes = [POINTER(fc2Context)]
    fc2CreateGigEContext.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 73
if hasattr(_libs['flycapture-c'], 'fc2DestroyContext'):
    fc2DestroyContext = _libs['flycapture-c'].fc2DestroyContext
    fc2DestroyContext.argtypes = [fc2Context]
    fc2DestroyContext.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 87
if hasattr(_libs['flycapture-c'], 'fc2FireBusReset'):
    fc2FireBusReset = _libs['flycapture-c'].fc2FireBusReset
    fc2FireBusReset.argtypes = [fc2Context, POINTER(fc2PGRGuid)]
    fc2FireBusReset.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 100
if hasattr(_libs['flycapture-c'], 'fc2GetNumOfCameras'):
    fc2GetNumOfCameras = _libs['flycapture-c'].fc2GetNumOfCameras
    fc2GetNumOfCameras.argtypes = [fc2Context, POINTER(c_uint)]
    fc2GetNumOfCameras.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 114
if hasattr(_libs['flycapture-c'], 'fc2IsCameraControlable'):
    fc2IsCameraControlable = _libs['flycapture-c'].fc2IsCameraControlable
    fc2IsCameraControlable.argtypes = [fc2Context, POINTER(fc2PGRGuid), POINTER(BOOL)]
    fc2IsCameraControlable.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 131
if hasattr(_libs['flycapture-c'], 'fc2GetCameraFromIndex'):
    fc2GetCameraFromIndex = _libs['flycapture-c'].fc2GetCameraFromIndex
    fc2GetCameraFromIndex.argtypes = [fc2Context, c_uint, POINTER(fc2PGRGuid)]
    fc2GetCameraFromIndex.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 148
if hasattr(_libs['flycapture-c'], 'fc2GetCameraFromSerialNumber'):
    fc2GetCameraFromSerialNumber = _libs['flycapture-c'].fc2GetCameraFromSerialNumber
    fc2GetCameraFromSerialNumber.argtypes = [fc2Context, c_uint, POINTER(fc2PGRGuid)]
    fc2GetCameraFromSerialNumber.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 163
if hasattr(_libs['flycapture-c'], 'fc2GetCameraSerialNumberFromIndex'):
    fc2GetCameraSerialNumberFromIndex = _libs['flycapture-c'].fc2GetCameraSerialNumberFromIndex
    fc2GetCameraSerialNumberFromIndex.argtypes = [fc2Context, c_uint, POINTER(c_uint)]
    fc2GetCameraSerialNumberFromIndex.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 180
if hasattr(_libs['flycapture-c'], 'fc2GetInterfaceTypeFromGuid'):
    fc2GetInterfaceTypeFromGuid = _libs['flycapture-c'].fc2GetInterfaceTypeFromGuid
    fc2GetInterfaceTypeFromGuid.argtypes = [fc2Context, POINTER(fc2PGRGuid), POINTER(fc2InterfaceType)]
    fc2GetInterfaceTypeFromGuid.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 195
if hasattr(_libs['flycapture-c'], 'fc2GetNumOfDevices'):
    fc2GetNumOfDevices = _libs['flycapture-c'].fc2GetNumOfDevices
    fc2GetNumOfDevices.argtypes = [fc2Context, POINTER(c_uint)]
    fc2GetNumOfDevices.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 212
if hasattr(_libs['flycapture-c'], 'fc2GetDeviceFromIndex'):
    fc2GetDeviceFromIndex = _libs['flycapture-c'].fc2GetDeviceFromIndex
    fc2GetDeviceFromIndex.argtypes = [fc2Context, c_uint, POINTER(fc2PGRGuid)]
    fc2GetDeviceFromIndex.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 231
if hasattr(_libs['flycapture-c'], 'fc2RegisterCallback'):
    fc2RegisterCallback = _libs['flycapture-c'].fc2RegisterCallback
    fc2RegisterCallback.argtypes = [fc2Context, fc2BusEventCallback, fc2BusCallbackType, POINTER(None), POINTER(fc2CallbackHandle)]
    fc2RegisterCallback.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 247
if hasattr(_libs['flycapture-c'], 'fc2UnregisterCallback'):
    fc2UnregisterCallback = _libs['flycapture-c'].fc2UnregisterCallback
    fc2UnregisterCallback.argtypes = [fc2Context, fc2CallbackHandle]
    fc2UnregisterCallback.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 259
if hasattr(_libs['flycapture-c'], 'fc2RescanBus'):
    fc2RescanBus = _libs['flycapture-c'].fc2RescanBus
    fc2RescanBus.argtypes = [fc2Context]
    fc2RescanBus.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 276
if hasattr(_libs['flycapture-c'], 'fc2ForceIPAddressToCamera'):
    fc2ForceIPAddressToCamera = _libs['flycapture-c'].fc2ForceIPAddressToCamera
    fc2ForceIPAddressToCamera.argtypes = [fc2Context, fc2MACAddress, fc2IPAddress, fc2IPAddress, fc2IPAddress]
    fc2ForceIPAddressToCamera.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 293
if hasattr(_libs['flycapture-c'], 'fc2ForceAllIPAddressesAutomatically'):
    fc2ForceAllIPAddressesAutomatically = _libs['flycapture-c'].fc2ForceAllIPAddressesAutomatically
    fc2ForceAllIPAddressesAutomatically.argtypes = []
    fc2ForceAllIPAddressesAutomatically.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 305
if hasattr(_libs['flycapture-c'], 'fc2ForceIPAddressAutomatically'):
    fc2ForceIPAddressAutomatically = _libs['flycapture-c'].fc2ForceIPAddressAutomatically
    fc2ForceIPAddressAutomatically.argtypes = [c_uint]
    fc2ForceIPAddressAutomatically.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 325
if hasattr(_libs['flycapture-c'], 'fc2DiscoverGigECameras'):
    fc2DiscoverGigECameras = _libs['flycapture-c'].fc2DiscoverGigECameras
    fc2DiscoverGigECameras.argtypes = [fc2Context, POINTER(fc2CameraInfo), POINTER(c_uint)]
    fc2DiscoverGigECameras.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 340
if hasattr(_libs['flycapture-c'], 'fc2WriteRegister'):
    fc2WriteRegister = _libs['flycapture-c'].fc2WriteRegister
    fc2WriteRegister.argtypes = [fc2Context, c_uint, c_uint]
    fc2WriteRegister.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 355
if hasattr(_libs['flycapture-c'], 'fc2WriteRegisterBroadcast'):
    fc2WriteRegisterBroadcast = _libs['flycapture-c'].fc2WriteRegisterBroadcast
    fc2WriteRegisterBroadcast.argtypes = [fc2Context, c_uint, c_uint]
    fc2WriteRegisterBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 370
if hasattr(_libs['flycapture-c'], 'fc2ReadRegister'):
    fc2ReadRegister = _libs['flycapture-c'].fc2ReadRegister
    fc2ReadRegister.argtypes = [fc2Context, c_uint, POINTER(c_uint)]
    fc2ReadRegister.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 387
if hasattr(_libs['flycapture-c'], 'fc2WriteRegisterBlock'):
    fc2WriteRegisterBlock = _libs['flycapture-c'].fc2WriteRegisterBlock
    fc2WriteRegisterBlock.argtypes = [fc2Context, c_ushort, c_uint, POINTER(c_uint), c_uint]
    fc2WriteRegisterBlock.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 406
if hasattr(_libs['flycapture-c'], 'fc2ReadRegisterBlock'):
    fc2ReadRegisterBlock = _libs['flycapture-c'].fc2ReadRegisterBlock
    fc2ReadRegisterBlock.argtypes = [fc2Context, c_ushort, c_uint, POINTER(c_uint), c_uint]
    fc2ReadRegisterBlock.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 422
if hasattr(_libs['flycapture-c'], 'fc2Connect'):
    fc2Connect = _libs['flycapture-c'].fc2Connect
    fc2Connect.argtypes = [fc2Context, POINTER(fc2PGRGuid)]
    fc2Connect.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 434
if hasattr(_libs['flycapture-c'], 'fc2Disconnect'):
    fc2Disconnect = _libs['flycapture-c'].fc2Disconnect
    fc2Disconnect.argtypes = [fc2Context]
    fc2Disconnect.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 450
if hasattr(_libs['flycapture-c'], 'fc2SetCallback'):
    fc2SetCallback = _libs['flycapture-c'].fc2SetCallback
    fc2SetCallback.argtypes = [fc2Context, fc2ImageEventCallback, POINTER(None)]
    fc2SetCallback.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 464
if hasattr(_libs['flycapture-c'], 'fc2StartCapture'):
    fc2StartCapture = _libs['flycapture-c'].fc2StartCapture
    fc2StartCapture.argtypes = [fc2Context]
    fc2StartCapture.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 480
if hasattr(_libs['flycapture-c'], 'fc2StartCaptureCallback'):
    fc2StartCaptureCallback = _libs['flycapture-c'].fc2StartCaptureCallback
    fc2StartCaptureCallback.argtypes = [fc2Context, fc2ImageEventCallback, POINTER(None)]
    fc2StartCaptureCallback.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 494
if hasattr(_libs['flycapture-c'], 'fc2StartSyncCapture'):
    fc2StartSyncCapture = _libs['flycapture-c'].fc2StartSyncCapture
    fc2StartSyncCapture.argtypes = [c_uint, POINTER(fc2Context)]
    fc2StartSyncCapture.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 509
if hasattr(_libs['flycapture-c'], 'fc2StartSyncCaptureCallback'):
    fc2StartSyncCaptureCallback = _libs['flycapture-c'].fc2StartSyncCaptureCallback
    fc2StartSyncCaptureCallback.argtypes = [c_uint, POINTER(fc2Context), POINTER(fc2ImageEventCallback), POINTER(POINTER(None))]
    fc2StartSyncCaptureCallback.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 524
if hasattr(_libs['flycapture-c'], 'fc2RetrieveBuffer'):
    fc2RetrieveBuffer = _libs['flycapture-c'].fc2RetrieveBuffer
    fc2RetrieveBuffer.argtypes = [fc2Context, POINTER(fc2Image)]
    fc2RetrieveBuffer.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 537
if hasattr(_libs['flycapture-c'], 'fc2StopCapture'):
    fc2StopCapture = _libs['flycapture-c'].fc2StopCapture
    fc2StopCapture.argtypes = [fc2Context]
    fc2StopCapture.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 553
if hasattr(_libs['flycapture-c'], 'fc2SetUserBuffers'):
    fc2SetUserBuffers = _libs['flycapture-c'].fc2SetUserBuffers
    fc2SetUserBuffers.argtypes = [fc2Context, POINTER(c_ubyte), c_int, c_int]
    fc2SetUserBuffers.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 568
if hasattr(_libs['flycapture-c'], 'fc2GetConfiguration'):
    fc2GetConfiguration = _libs['flycapture-c'].fc2GetConfiguration
    fc2GetConfiguration.argtypes = [fc2Context, POINTER(fc2Config)]
    fc2GetConfiguration.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 581
if hasattr(_libs['flycapture-c'], 'fc2SetConfiguration'):
    fc2SetConfiguration = _libs['flycapture-c'].fc2SetConfiguration
    fc2SetConfiguration.argtypes = [fc2Context, POINTER(fc2Config)]
    fc2SetConfiguration.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 596
if hasattr(_libs['flycapture-c'], 'fc2GetCameraInfo'):
    fc2GetCameraInfo = _libs['flycapture-c'].fc2GetCameraInfo
    fc2GetCameraInfo.argtypes = [fc2Context, POINTER(fc2CameraInfo)]
    fc2GetCameraInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 611
if hasattr(_libs['flycapture-c'], 'fc2GetPropertyInfo'):
    fc2GetPropertyInfo = _libs['flycapture-c'].fc2GetPropertyInfo
    fc2GetPropertyInfo.argtypes = [fc2Context, POINTER(fc2PropertyInfo)]
    fc2GetPropertyInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 628
if hasattr(_libs['flycapture-c'], 'fc2GetProperty'):
    fc2GetProperty = _libs['flycapture-c'].fc2GetProperty
    fc2GetProperty.argtypes = [fc2Context, POINTER(fc2Property)]
    fc2GetProperty.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 645
if hasattr(_libs['flycapture-c'], 'fc2SetProperty'):
    fc2SetProperty = _libs['flycapture-c'].fc2SetProperty
    fc2SetProperty.argtypes = [fc2Context, POINTER(fc2Property)]
    fc2SetProperty.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 662
if hasattr(_libs['flycapture-c'], 'fc2SetPropertyBroadcast'):
    fc2SetPropertyBroadcast = _libs['flycapture-c'].fc2SetPropertyBroadcast
    fc2SetPropertyBroadcast.argtypes = [fc2Context, POINTER(fc2Property)]
    fc2SetPropertyBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 678
if hasattr(_libs['flycapture-c'], 'fc2GetGPIOPinDirection'):
    fc2GetGPIOPinDirection = _libs['flycapture-c'].fc2GetGPIOPinDirection
    fc2GetGPIOPinDirection.argtypes = [fc2Context, c_uint, POINTER(c_uint)]
    fc2GetGPIOPinDirection.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 697
if hasattr(_libs['flycapture-c'], 'fc2SetGPIOPinDirection'):
    fc2SetGPIOPinDirection = _libs['flycapture-c'].fc2SetGPIOPinDirection
    fc2SetGPIOPinDirection.argtypes = [fc2Context, c_uint, c_uint]
    fc2SetGPIOPinDirection.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 716
if hasattr(_libs['flycapture-c'], 'fc2SetGPIOPinDirectionBroadcast'):
    fc2SetGPIOPinDirectionBroadcast = _libs['flycapture-c'].fc2SetGPIOPinDirectionBroadcast
    fc2SetGPIOPinDirectionBroadcast.argtypes = [fc2Context, c_uint, c_uint]
    fc2SetGPIOPinDirectionBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 730
if hasattr(_libs['flycapture-c'], 'fc2GetTriggerModeInfo'):
    fc2GetTriggerModeInfo = _libs['flycapture-c'].fc2GetTriggerModeInfo
    fc2GetTriggerModeInfo.argtypes = [fc2Context, POINTER(fc2TriggerModeInfo)]
    fc2GetTriggerModeInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 743
if hasattr(_libs['flycapture-c'], 'fc2GetTriggerMode'):
    fc2GetTriggerMode = _libs['flycapture-c'].fc2GetTriggerMode
    fc2GetTriggerMode.argtypes = [fc2Context, POINTER(fc2TriggerMode)]
    fc2GetTriggerMode.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 756
if hasattr(_libs['flycapture-c'], 'fc2SetTriggerMode'):
    fc2SetTriggerMode = _libs['flycapture-c'].fc2SetTriggerMode
    fc2SetTriggerMode.argtypes = [fc2Context, POINTER(fc2TriggerMode)]
    fc2SetTriggerMode.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 769
if hasattr(_libs['flycapture-c'], 'fc2SetTriggerModeBroadcast'):
    fc2SetTriggerModeBroadcast = _libs['flycapture-c'].fc2SetTriggerModeBroadcast
    fc2SetTriggerModeBroadcast.argtypes = [fc2Context, POINTER(fc2TriggerMode)]
    fc2SetTriggerModeBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 781
if hasattr(_libs['flycapture-c'], 'fc2FireSoftwareTrigger'):
    fc2FireSoftwareTrigger = _libs['flycapture-c'].fc2FireSoftwareTrigger
    fc2FireSoftwareTrigger.argtypes = [fc2Context]
    fc2FireSoftwareTrigger.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 792
if hasattr(_libs['flycapture-c'], 'fc2FireSoftwareTriggerBroadcast'):
    fc2FireSoftwareTriggerBroadcast = _libs['flycapture-c'].fc2FireSoftwareTriggerBroadcast
    fc2FireSoftwareTriggerBroadcast.argtypes = [fc2Context]
    fc2FireSoftwareTriggerBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 804
if hasattr(_libs['flycapture-c'], 'fc2GetTriggerDelayInfo'):
    fc2GetTriggerDelayInfo = _libs['flycapture-c'].fc2GetTriggerDelayInfo
    fc2GetTriggerDelayInfo.argtypes = [fc2Context, POINTER(fc2TriggerDelayInfo)]
    fc2GetTriggerDelayInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 817
if hasattr(_libs['flycapture-c'], 'fc2GetTriggerDelay'):
    fc2GetTriggerDelay = _libs['flycapture-c'].fc2GetTriggerDelay
    fc2GetTriggerDelay.argtypes = [fc2Context, POINTER(fc2TriggerDelay)]
    fc2GetTriggerDelay.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 830
if hasattr(_libs['flycapture-c'], 'fc2SetTriggerDelay'):
    fc2SetTriggerDelay = _libs['flycapture-c'].fc2SetTriggerDelay
    fc2SetTriggerDelay.argtypes = [fc2Context, POINTER(fc2TriggerDelay)]
    fc2SetTriggerDelay.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 843
if hasattr(_libs['flycapture-c'], 'fc2SetTriggerDelayBroadcast'):
    fc2SetTriggerDelayBroadcast = _libs['flycapture-c'].fc2SetTriggerDelayBroadcast
    fc2SetTriggerDelayBroadcast.argtypes = [fc2Context, POINTER(fc2TriggerDelay)]
    fc2SetTriggerDelayBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 856
if hasattr(_libs['flycapture-c'], 'fc2GetStrobeInfo'):
    fc2GetStrobeInfo = _libs['flycapture-c'].fc2GetStrobeInfo
    fc2GetStrobeInfo.argtypes = [fc2Context, POINTER(fc2StrobeInfo)]
    fc2GetStrobeInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 871
if hasattr(_libs['flycapture-c'], 'fc2GetStrobe'):
    fc2GetStrobe = _libs['flycapture-c'].fc2GetStrobe
    fc2GetStrobe.argtypes = [fc2Context, POINTER(fc2StrobeControl)]
    fc2GetStrobe.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 886
if hasattr(_libs['flycapture-c'], 'fc2SetStrobe'):
    fc2SetStrobe = _libs['flycapture-c'].fc2SetStrobe
    fc2SetStrobe.argtypes = [fc2Context, POINTER(fc2StrobeControl)]
    fc2SetStrobe.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 901
if hasattr(_libs['flycapture-c'], 'fc2SetStrobeBroadcast'):
    fc2SetStrobeBroadcast = _libs['flycapture-c'].fc2SetStrobeBroadcast
    fc2SetStrobeBroadcast.argtypes = [fc2Context, POINTER(fc2StrobeControl)]
    fc2SetStrobeBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 917
if hasattr(_libs['flycapture-c'], 'fc2GetVideoModeAndFrameRateInfo'):
    fc2GetVideoModeAndFrameRateInfo = _libs['flycapture-c'].fc2GetVideoModeAndFrameRateInfo
    fc2GetVideoModeAndFrameRateInfo.argtypes = [fc2Context, fc2VideoMode, fc2FrameRate, POINTER(BOOL)]
    fc2GetVideoModeAndFrameRateInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 935
if hasattr(_libs['flycapture-c'], 'fc2GetVideoModeAndFrameRate'):
    fc2GetVideoModeAndFrameRate = _libs['flycapture-c'].fc2GetVideoModeAndFrameRate
    fc2GetVideoModeAndFrameRate.argtypes = [fc2Context, POINTER(fc2VideoMode), POINTER(fc2FrameRate)]
    fc2GetVideoModeAndFrameRate.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 953
if hasattr(_libs['flycapture-c'], 'fc2SetVideoModeAndFrameRate'):
    fc2SetVideoModeAndFrameRate = _libs['flycapture-c'].fc2SetVideoModeAndFrameRate
    fc2SetVideoModeAndFrameRate.argtypes = [fc2Context, fc2VideoMode, fc2FrameRate]
    fc2SetVideoModeAndFrameRate.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 972
if hasattr(_libs['flycapture-c'], 'fc2GetFormat7Info'):
    fc2GetFormat7Info = _libs['flycapture-c'].fc2GetFormat7Info
    fc2GetFormat7Info.argtypes = [fc2Context, POINTER(fc2Format7Info), POINTER(BOOL)]
    fc2GetFormat7Info.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 992
if hasattr(_libs['flycapture-c'], 'fc2ValidateFormat7Settings'):
    fc2ValidateFormat7Settings = _libs['flycapture-c'].fc2ValidateFormat7Settings
    fc2ValidateFormat7Settings.argtypes = [fc2Context, POINTER(fc2Format7ImageSettings), POINTER(BOOL), POINTER(fc2Format7PacketInfo)]
    fc2ValidateFormat7Settings.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1010
if hasattr(_libs['flycapture-c'], 'fc2GetFormat7Configuration'):
    fc2GetFormat7Configuration = _libs['flycapture-c'].fc2GetFormat7Configuration
    fc2GetFormat7Configuration.argtypes = [fc2Context, POINTER(fc2Format7ImageSettings), POINTER(c_uint), POINTER(c_float)]
    fc2GetFormat7Configuration.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1026
if hasattr(_libs['flycapture-c'], 'fc2SetFormat7ConfigurationPacket'):
    fc2SetFormat7ConfigurationPacket = _libs['flycapture-c'].fc2SetFormat7ConfigurationPacket
    fc2SetFormat7ConfigurationPacket.argtypes = [fc2Context, POINTER(fc2Format7ImageSettings), c_uint]
    fc2SetFormat7ConfigurationPacket.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1041
if hasattr(_libs['flycapture-c'], 'fc2SetFormat7Configuration'):
    fc2SetFormat7Configuration = _libs['flycapture-c'].fc2SetFormat7Configuration
    fc2SetFormat7Configuration.argtypes = [fc2Context, POINTER(fc2Format7ImageSettings), c_float]
    fc2SetFormat7Configuration.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1056
if hasattr(_libs['flycapture-c'], 'fc2WriteGVCPRegister'):
    fc2WriteGVCPRegister = _libs['flycapture-c'].fc2WriteGVCPRegister
    fc2WriteGVCPRegister.argtypes = [fc2Context, c_uint, c_uint]
    fc2WriteGVCPRegister.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1071
if hasattr(_libs['flycapture-c'], 'fc2WriteGVCPRegisterBroadcast'):
    fc2WriteGVCPRegisterBroadcast = _libs['flycapture-c'].fc2WriteGVCPRegisterBroadcast
    fc2WriteGVCPRegisterBroadcast.argtypes = [fc2Context, c_uint, c_uint]
    fc2WriteGVCPRegisterBroadcast.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1086
if hasattr(_libs['flycapture-c'], 'fc2ReadGVCPRegister'):
    fc2ReadGVCPRegister = _libs['flycapture-c'].fc2ReadGVCPRegister
    fc2ReadGVCPRegister.argtypes = [fc2Context, c_uint, POINTER(c_uint)]
    fc2ReadGVCPRegister.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1102
if hasattr(_libs['flycapture-c'], 'fc2WriteGVCPRegisterBlock'):
    fc2WriteGVCPRegisterBlock = _libs['flycapture-c'].fc2WriteGVCPRegisterBlock
    fc2WriteGVCPRegisterBlock.argtypes = [fc2Context, c_uint, POINTER(c_uint), c_uint]
    fc2WriteGVCPRegisterBlock.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1119
if hasattr(_libs['flycapture-c'], 'fc2ReadGVCPRegisterBlock'):
    fc2ReadGVCPRegisterBlock = _libs['flycapture-c'].fc2ReadGVCPRegisterBlock
    fc2ReadGVCPRegisterBlock.argtypes = [fc2Context, c_uint, POINTER(c_uint), c_uint]
    fc2ReadGVCPRegisterBlock.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1136
if hasattr(_libs['flycapture-c'], 'fc2WriteGVCPMemory'):
    fc2WriteGVCPMemory = _libs['flycapture-c'].fc2WriteGVCPMemory
    fc2WriteGVCPMemory.argtypes = [fc2Context, c_uint, POINTER(c_ubyte), c_uint]
    fc2WriteGVCPMemory.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1153
if hasattr(_libs['flycapture-c'], 'fc2ReadGVCPMemory'):
    fc2ReadGVCPMemory = _libs['flycapture-c'].fc2ReadGVCPMemory
    fc2ReadGVCPMemory.argtypes = [fc2Context, c_uint, POINTER(c_ubyte), c_uint]
    fc2ReadGVCPMemory.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1169
if hasattr(_libs['flycapture-c'], 'fc2GetGigEProperty'):
    fc2GetGigEProperty = _libs['flycapture-c'].fc2GetGigEProperty
    fc2GetGigEProperty.argtypes = [fc2Context, POINTER(fc2GigEProperty)]
    fc2GetGigEProperty.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1183
if hasattr(_libs['flycapture-c'], 'fc2SetGigEProperty'):
    fc2SetGigEProperty = _libs['flycapture-c'].fc2SetGigEProperty
    fc2SetGigEProperty.argtypes = [fc2Context, POINTER(fc2GigEProperty)]
    fc2SetGigEProperty.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1188
if hasattr(_libs['flycapture-c'], 'fc2QueryGigEImagingMode'):
    fc2QueryGigEImagingMode = _libs['flycapture-c'].fc2QueryGigEImagingMode
    fc2QueryGigEImagingMode.argtypes = [fc2Context, fc2Mode, POINTER(BOOL)]
    fc2QueryGigEImagingMode.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1194
if hasattr(_libs['flycapture-c'], 'fc2GetGigEImagingMode'):
    fc2GetGigEImagingMode = _libs['flycapture-c'].fc2GetGigEImagingMode
    fc2GetGigEImagingMode.argtypes = [fc2Context, POINTER(fc2Mode)]
    fc2GetGigEImagingMode.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1199
if hasattr(_libs['flycapture-c'], 'fc2SetGigEImagingMode'):
    fc2SetGigEImagingMode = _libs['flycapture-c'].fc2SetGigEImagingMode
    fc2SetGigEImagingMode.argtypes = [fc2Context, fc2Mode]
    fc2SetGigEImagingMode.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1204
if hasattr(_libs['flycapture-c'], 'fc2GetGigEImageSettingsInfo'):
    fc2GetGigEImageSettingsInfo = _libs['flycapture-c'].fc2GetGigEImageSettingsInfo
    fc2GetGigEImageSettingsInfo.argtypes = [fc2Context, POINTER(fc2GigEImageSettingsInfo)]
    fc2GetGigEImageSettingsInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1209
if hasattr(_libs['flycapture-c'], 'fc2GetGigEImageSettings'):
    fc2GetGigEImageSettings = _libs['flycapture-c'].fc2GetGigEImageSettings
    fc2GetGigEImageSettings.argtypes = [fc2Context, POINTER(fc2GigEImageSettings)]
    fc2GetGigEImageSettings.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1214
if hasattr(_libs['flycapture-c'], 'fc2SetGigEImageSettings'):
    fc2SetGigEImageSettings = _libs['flycapture-c'].fc2SetGigEImageSettings
    fc2SetGigEImageSettings.argtypes = [fc2Context, POINTER(fc2GigEImageSettings)]
    fc2SetGigEImageSettings.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1219
if hasattr(_libs['flycapture-c'], 'fc2GetGigEConfig'):
    fc2GetGigEConfig = _libs['flycapture-c'].fc2GetGigEConfig
    fc2GetGigEConfig.argtypes = [fc2Context, POINTER(fc2GigEConfig)]
    fc2GetGigEConfig.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1224
if hasattr(_libs['flycapture-c'], 'fc2SetGigEConfig'):
    fc2SetGigEConfig = _libs['flycapture-c'].fc2SetGigEConfig
    fc2SetGigEConfig.argtypes = [fc2Context, POINTER(fc2GigEConfig)]
    fc2SetGigEConfig.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1229
if hasattr(_libs['flycapture-c'], 'fc2GetGigEImageBinningSettings'):
    fc2GetGigEImageBinningSettings = _libs['flycapture-c'].fc2GetGigEImageBinningSettings
    fc2GetGigEImageBinningSettings.argtypes = [fc2Context, POINTER(c_uint), POINTER(c_uint)]
    fc2GetGigEImageBinningSettings.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1235
if hasattr(_libs['flycapture-c'], 'fc2SetGigEImageBinningSettings'):
    fc2SetGigEImageBinningSettings = _libs['flycapture-c'].fc2SetGigEImageBinningSettings
    fc2SetGigEImageBinningSettings.argtypes = [fc2Context, c_uint, c_uint]
    fc2SetGigEImageBinningSettings.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1241
if hasattr(_libs['flycapture-c'], 'fc2GetNumStreamChannels'):
    fc2GetNumStreamChannels = _libs['flycapture-c'].fc2GetNumStreamChannels
    fc2GetNumStreamChannels.argtypes = [fc2Context, POINTER(c_uint)]
    fc2GetNumStreamChannels.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1246
if hasattr(_libs['flycapture-c'], 'fc2GetGigEStreamChannelInfo'):
    fc2GetGigEStreamChannelInfo = _libs['flycapture-c'].fc2GetGigEStreamChannelInfo
    fc2GetGigEStreamChannelInfo.argtypes = [fc2Context, c_uint, POINTER(fc2GigEStreamChannel)]
    fc2GetGigEStreamChannelInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1252
if hasattr(_libs['flycapture-c'], 'fc2SetGigEStreamChannelInfo'):
    fc2SetGigEStreamChannelInfo = _libs['flycapture-c'].fc2SetGigEStreamChannelInfo
    fc2SetGigEStreamChannelInfo.argtypes = [fc2Context, c_uint, POINTER(fc2GigEStreamChannel)]
    fc2SetGigEStreamChannelInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1268
if hasattr(_libs['flycapture-c'], 'fc2GetLUTInfo'):
    fc2GetLUTInfo = _libs['flycapture-c'].fc2GetLUTInfo
    fc2GetLUTInfo.argtypes = [fc2Context, POINTER(fc2LUTData)]
    fc2GetLUTInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1283
if hasattr(_libs['flycapture-c'], 'fc2GetLUTBankInfo'):
    fc2GetLUTBankInfo = _libs['flycapture-c'].fc2GetLUTBankInfo
    fc2GetLUTBankInfo.argtypes = [fc2Context, c_uint, POINTER(BOOL), POINTER(BOOL)]
    fc2GetLUTBankInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1299
if hasattr(_libs['flycapture-c'], 'fc2GetActiveLUTBank'):
    fc2GetActiveLUTBank = _libs['flycapture-c'].fc2GetActiveLUTBank
    fc2GetActiveLUTBank.argtypes = [fc2Context, POINTER(c_uint)]
    fc2GetActiveLUTBank.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1312
if hasattr(_libs['flycapture-c'], 'fc2SetActiveLUTBank'):
    fc2SetActiveLUTBank = _libs['flycapture-c'].fc2SetActiveLUTBank
    fc2SetActiveLUTBank.argtypes = [fc2Context, c_uint]
    fc2SetActiveLUTBank.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1325
if hasattr(_libs['flycapture-c'], 'fc2EnableLUT'):
    fc2EnableLUT = _libs['flycapture-c'].fc2EnableLUT
    fc2EnableLUT.argtypes = [fc2Context, BOOL]
    fc2EnableLUT.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1341
if hasattr(_libs['flycapture-c'], 'fc2GetLUTChannel'):
    fc2GetLUTChannel = _libs['flycapture-c'].fc2GetLUTChannel
    fc2GetLUTChannel.argtypes = [fc2Context, c_uint, c_uint, c_uint, POINTER(c_uint)]
    fc2GetLUTChannel.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1361
if hasattr(_libs['flycapture-c'], 'fc2SetLUTChannel'):
    fc2SetLUTChannel = _libs['flycapture-c'].fc2SetLUTChannel
    fc2SetLUTChannel.argtypes = [fc2Context, c_uint, c_uint, c_uint, POINTER(c_uint)]
    fc2SetLUTChannel.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1377
if hasattr(_libs['flycapture-c'], 'fc2GetMemoryChannel'):
    fc2GetMemoryChannel = _libs['flycapture-c'].fc2GetMemoryChannel
    fc2GetMemoryChannel.argtypes = [fc2Context, POINTER(c_uint)]
    fc2GetMemoryChannel.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1390
if hasattr(_libs['flycapture-c'], 'fc2SaveToMemoryChannel'):
    fc2SaveToMemoryChannel = _libs['flycapture-c'].fc2SaveToMemoryChannel
    fc2SaveToMemoryChannel.argtypes = [fc2Context, c_uint]
    fc2SaveToMemoryChannel.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1403
if hasattr(_libs['flycapture-c'], 'fc2RestoreFromMemoryChannel'):
    fc2RestoreFromMemoryChannel = _libs['flycapture-c'].fc2RestoreFromMemoryChannel
    fc2RestoreFromMemoryChannel.argtypes = [fc2Context, c_uint]
    fc2RestoreFromMemoryChannel.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1417
if hasattr(_libs['flycapture-c'], 'fc2GetMemoryChannelInfo'):
    fc2GetMemoryChannelInfo = _libs['flycapture-c'].fc2GetMemoryChannelInfo
    fc2GetMemoryChannelInfo.argtypes = [fc2Context, POINTER(c_uint)]
    fc2GetMemoryChannelInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1431
if hasattr(_libs['flycapture-c'], 'fc2GetEmbeddedImageInfo'):
    fc2GetEmbeddedImageInfo = _libs['flycapture-c'].fc2GetEmbeddedImageInfo
    fc2GetEmbeddedImageInfo.argtypes = [fc2Context, POINTER(fc2EmbeddedImageInfo)]
    fc2GetEmbeddedImageInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1445
if hasattr(_libs['flycapture-c'], 'fc2SetEmbeddedImageInfo'):
    fc2SetEmbeddedImageInfo = _libs['flycapture-c'].fc2SetEmbeddedImageInfo
    fc2SetEmbeddedImageInfo.argtypes = [fc2Context, POINTER(fc2EmbeddedImageInfo)]
    fc2SetEmbeddedImageInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1456
if hasattr(_libs['flycapture-c'], 'fc2GetRegisterString'):
    fc2GetRegisterString = _libs['flycapture-c'].fc2GetRegisterString
    fc2GetRegisterString.argtypes = [c_uint]
    if sizeof(c_int) == sizeof(c_void_p):
        fc2GetRegisterString.restype = ReturnString
    else:
        fc2GetRegisterString.restype = String
        fc2GetRegisterString.errcheck = ReturnString

# /usr/include/flycapture/C/FlyCapture2_C.h: 1473
if hasattr(_libs['flycapture-c'], 'fc2CreateImage'):
    fc2CreateImage = _libs['flycapture-c'].fc2CreateImage
    fc2CreateImage.argtypes = [POINTER(fc2Image)]
    fc2CreateImage.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1484
if hasattr(_libs['flycapture-c'], 'fc2DestroyImage'):
    fc2DestroyImage = _libs['flycapture-c'].fc2DestroyImage
    fc2DestroyImage.argtypes = [POINTER(fc2Image)]
    fc2DestroyImage.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1499
if hasattr(_libs['flycapture-c'], 'fc2SetDefaultColorProcessing'):
    fc2SetDefaultColorProcessing = _libs['flycapture-c'].fc2SetDefaultColorProcessing
    fc2SetDefaultColorProcessing.argtypes = [fc2ColorProcessingAlgorithm]
    fc2SetDefaultColorProcessing.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1510
if hasattr(_libs['flycapture-c'], 'fc2GetDefaultColorProcessing'):
    fc2GetDefaultColorProcessing = _libs['flycapture-c'].fc2GetDefaultColorProcessing
    fc2GetDefaultColorProcessing.argtypes = [POINTER(fc2ColorProcessingAlgorithm)]
    fc2GetDefaultColorProcessing.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1525
if hasattr(_libs['flycapture-c'], 'fc2SetDefaultOutputFormat'):
    fc2SetDefaultOutputFormat = _libs['flycapture-c'].fc2SetDefaultOutputFormat
    fc2SetDefaultOutputFormat.argtypes = [fc2PixelFormat]
    fc2SetDefaultOutputFormat.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1536
if hasattr(_libs['flycapture-c'], 'fc2GetDefaultOutputFormat'):
    fc2GetDefaultOutputFormat = _libs['flycapture-c'].fc2GetDefaultOutputFormat
    fc2GetDefaultOutputFormat.argtypes = [POINTER(fc2PixelFormat)]
    fc2GetDefaultOutputFormat.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1548
if hasattr(_libs['flycapture-c'], 'fc2DetermineBitsPerPixel'):
    fc2DetermineBitsPerPixel = _libs['flycapture-c'].fc2DetermineBitsPerPixel
    fc2DetermineBitsPerPixel.argtypes = [fc2PixelFormat, POINTER(c_uint)]
    fc2DetermineBitsPerPixel.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1563
if hasattr(_libs['flycapture-c'], 'fc2SaveImage'):
    fc2SaveImage = _libs['flycapture-c'].fc2SaveImage
    fc2SaveImage.argtypes = [POINTER(fc2Image), String, fc2ImageFileFormat]
    fc2SaveImage.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1580
if hasattr(_libs['flycapture-c'], 'fc2SaveImageWithOption'):
    fc2SaveImageWithOption = _libs['flycapture-c'].fc2SaveImageWithOption
    fc2SaveImageWithOption.argtypes = [POINTER(fc2Image), String, fc2ImageFileFormat, POINTER(None)]
    fc2SaveImageWithOption.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1595
if hasattr(_libs['flycapture-c'], 'fc2ConvertImage'):
    fc2ConvertImage = _libs['flycapture-c'].fc2ConvertImage
    fc2ConvertImage.argtypes = [POINTER(fc2Image), POINTER(fc2Image)]
    fc2ConvertImage.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1611
if hasattr(_libs['flycapture-c'], 'fc2ConvertImageTo'):
    fc2ConvertImageTo = _libs['flycapture-c'].fc2ConvertImageTo
    fc2ConvertImageTo.argtypes = [fc2PixelFormat, POINTER(fc2Image), POINTER(fc2Image)]
    fc2ConvertImageTo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1628
if hasattr(_libs['flycapture-c'], 'fc2GetImageData'):
    fc2GetImageData = _libs['flycapture-c'].fc2GetImageData
    fc2GetImageData.argtypes = [POINTER(fc2Image), POINTER(POINTER(c_ubyte))]
    fc2GetImageData.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1645
if hasattr(_libs['flycapture-c'], 'fc2SetImageData'):
    fc2SetImageData = _libs['flycapture-c'].fc2SetImageData
    fc2SetImageData.argtypes = [POINTER(fc2Image), POINTER(c_ubyte), c_uint]
    fc2SetImageData.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1663
if hasattr(_libs['flycapture-c'], 'fc2SetImageDimensions'):
    fc2SetImageDimensions = _libs['flycapture-c'].fc2SetImageDimensions
    fc2SetImageDimensions.argtypes = [POINTER(fc2Image), c_uint, c_uint, c_uint, fc2PixelFormat, fc2BayerTileFormat]
    fc2SetImageDimensions.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1679
if hasattr(_libs['flycapture-c'], 'fc2GetImageTimeStamp'):
    fc2GetImageTimeStamp = _libs['flycapture-c'].fc2GetImageTimeStamp
    fc2GetImageTimeStamp.argtypes = [POINTER(fc2Image)]
    fc2GetImageTimeStamp.restype = fc2TimeStamp

# /usr/include/flycapture/C/FlyCapture2_C.h: 1695
if hasattr(_libs['flycapture-c'], 'fc2CalculateImageStatistics'):
    fc2CalculateImageStatistics = _libs['flycapture-c'].fc2CalculateImageStatistics
    fc2CalculateImageStatistics.argtypes = [POINTER(fc2Image), POINTER(fc2ImageStatisticsContext)]
    fc2CalculateImageStatistics.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1707
if hasattr(_libs['flycapture-c'], 'fc2CreateImageStatistics'):
    fc2CreateImageStatistics = _libs['flycapture-c'].fc2CreateImageStatistics
    fc2CreateImageStatistics.argtypes = [POINTER(fc2ImageStatisticsContext)]
    fc2CreateImageStatistics.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1718
if hasattr(_libs['flycapture-c'], 'fc2DestroyImageStatistics'):
    fc2DestroyImageStatistics = _libs['flycapture-c'].fc2DestroyImageStatistics
    fc2DestroyImageStatistics.argtypes = [fc2ImageStatisticsContext]
    fc2DestroyImageStatistics.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1733
if hasattr(_libs['flycapture-c'], 'fc2GetChannelStatus'):
    fc2GetChannelStatus = _libs['flycapture-c'].fc2GetChannelStatus
    fc2GetChannelStatus.argtypes = [fc2ImageStatisticsContext, fc2StatisticsChannel, POINTER(BOOL)]
    fc2GetChannelStatus.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1750
if hasattr(_libs['flycapture-c'], 'fc2SetChannelStatus'):
    fc2SetChannelStatus = _libs['flycapture-c'].fc2SetChannelStatus
    fc2SetChannelStatus.argtypes = [fc2ImageStatisticsContext, fc2StatisticsChannel, BOOL]
    fc2SetChannelStatus.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1771
if hasattr(_libs['flycapture-c'], 'fc2GetImageStatistics'):
    fc2GetImageStatistics = _libs['flycapture-c'].fc2GetImageStatistics
    fc2GetImageStatistics.argtypes = [fc2ImageStatisticsContext, fc2StatisticsChannel, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(POINTER(c_int))]
    fc2GetImageStatistics.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1790
if hasattr(_libs['flycapture-c'], 'fc2CreateAVI'):
    fc2CreateAVI = _libs['flycapture-c'].fc2CreateAVI
    fc2CreateAVI.argtypes = [POINTER(fc2AVIContext)]
    fc2CreateAVI.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1805
if hasattr(_libs['flycapture-c'], 'fc2AVIOpen'):
    fc2AVIOpen = _libs['flycapture-c'].fc2AVIOpen
    fc2AVIOpen.argtypes = [fc2AVIContext, String, POINTER(fc2AVIOption)]
    fc2AVIOpen.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1822
if hasattr(_libs['flycapture-c'], 'fc2MJPGOpen'):
    fc2MJPGOpen = _libs['flycapture-c'].fc2MJPGOpen
    fc2MJPGOpen.argtypes = [fc2AVIContext, String, POINTER(fc2MJPGOption)]
    fc2MJPGOpen.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1839
if hasattr(_libs['flycapture-c'], 'fc2H264Open'):
    fc2H264Open = _libs['flycapture-c'].fc2H264Open
    fc2H264Open.argtypes = [fc2AVIContext, String, POINTER(fc2H264Option)]
    fc2H264Open.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1853
if hasattr(_libs['flycapture-c'], 'fc2AVIAppend'):
    fc2AVIAppend = _libs['flycapture-c'].fc2AVIAppend
    fc2AVIAppend.argtypes = [fc2AVIContext, POINTER(fc2Image)]
    fc2AVIAppend.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1865
if hasattr(_libs['flycapture-c'], 'fc2AVIClose'):
    fc2AVIClose = _libs['flycapture-c'].fc2AVIClose
    fc2AVIClose.argtypes = [fc2AVIContext]
    fc2AVIClose.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1876
if hasattr(_libs['flycapture-c'], 'fc2DestroyAVI'):
    fc2DestroyAVI = _libs['flycapture-c'].fc2DestroyAVI
    fc2DestroyAVI.argtypes = [fc2AVIContext]
    fc2DestroyAVI.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1887
if hasattr(_libs['flycapture-c'], 'fc2GetSystemInfo'):
    fc2GetSystemInfo = _libs['flycapture-c'].fc2GetSystemInfo
    fc2GetSystemInfo.argtypes = [POINTER(fc2SystemInfo)]
    fc2GetSystemInfo.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1897
if hasattr(_libs['flycapture-c'], 'fc2GetLibraryVersion'):
    fc2GetLibraryVersion = _libs['flycapture-c'].fc2GetLibraryVersion
    fc2GetLibraryVersion.argtypes = [POINTER(fc2Version)]
    fc2GetLibraryVersion.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1907
if hasattr(_libs['flycapture-c'], 'fc2LaunchBrowser'):
    fc2LaunchBrowser = _libs['flycapture-c'].fc2LaunchBrowser
    fc2LaunchBrowser.argtypes = [String]
    fc2LaunchBrowser.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1917
if hasattr(_libs['flycapture-c'], 'fc2LaunchHelp'):
    fc2LaunchHelp = _libs['flycapture-c'].fc2LaunchHelp
    fc2LaunchHelp.argtypes = [String]
    fc2LaunchHelp.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1928
if hasattr(_libs['flycapture-c'], 'fc2LaunchCommand'):
    fc2LaunchCommand = _libs['flycapture-c'].fc2LaunchCommand
    fc2LaunchCommand.argtypes = [String]
    fc2LaunchCommand.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1942
if hasattr(_libs['flycapture-c'], 'fc2LaunchCommandAsync'):
    fc2LaunchCommandAsync = _libs['flycapture-c'].fc2LaunchCommandAsync
    fc2LaunchCommandAsync.argtypes = [String, fc2AsyncCommandCallback, POINTER(None)]
    fc2LaunchCommandAsync.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2_C.h: 1954
if hasattr(_libs['flycapture-c'], 'fc2ErrorToDescription'):
    fc2ErrorToDescription = _libs['flycapture-c'].fc2ErrorToDescription
    fc2ErrorToDescription.argtypes = [fc2Error]
    if sizeof(c_int) == sizeof(c_void_p):
        fc2ErrorToDescription.restype = ReturnString
    else:
        fc2ErrorToDescription.restype = String
        fc2ErrorToDescription.errcheck = ReturnString

# /usr/include/flycapture/C/FlyCapture2_C.h: 1966
if hasattr(_libs['flycapture-c'], 'fc2GetCycleTime'):
    fc2GetCycleTime = _libs['flycapture-c'].fc2GetCycleTime
    fc2GetCycleTime.argtypes = [fc2Context, POINTER(fc2TimeStamp)]
    fc2GetCycleTime.restype = fc2Error

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 46
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2CreateGUIContext'):
        continue
    fc2CreateGUIContext = _lib.fc2CreateGUIContext
    fc2CreateGUIContext.argtypes = [POINTER(fc2GuiContext)]
    fc2CreateGUIContext.restype = fc2Error
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 57
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2DestroyGUIContext'):
        continue
    fc2DestroyGUIContext = _lib.fc2DestroyGUIContext
    fc2DestroyGUIContext.argtypes = [fc2GuiContext]
    fc2DestroyGUIContext.restype = fc2Error
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 69
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2GUIConnect'):
        continue
    fc2GUIConnect = _lib.fc2GUIConnect
    fc2GUIConnect.argtypes = [fc2GuiContext, fc2Context]
    fc2GUIConnect.restype = None
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 81
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2GUIDisconnect'):
        continue
    fc2GUIDisconnect = _lib.fc2GUIDisconnect
    fc2GUIDisconnect.argtypes = [fc2GuiContext]
    fc2GUIDisconnect.restype = None
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 97
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2Disonnect'):
        continue
    fc2Disonnect = _lib.fc2Disonnect
    fc2Disonnect.argtypes = [fc2GuiContext]
    fc2Disonnect.restype = None
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 108
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2Show'):
        continue
    fc2Show = _lib.fc2Show
    fc2Show.argtypes = [fc2GuiContext]
    fc2Show.restype = None
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 119
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2Hide'):
        continue
    fc2Hide = _lib.fc2Hide
    fc2Hide.argtypes = [fc2GuiContext]
    fc2Hide.restype = None
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 130
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2IsVisible'):
        continue
    fc2IsVisible = _lib.fc2IsVisible
    fc2IsVisible.argtypes = [fc2GuiContext]
    fc2IsVisible.restype = BOOL
    break

# /usr/include/flycapture/C/FlyCapture2GUI_C.h: 142
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fc2ShowModal'):
        continue
    fc2ShowModal = _lib.fc2ShowModal
    fc2ShowModal.argtypes = [fc2GuiContext, POINTER(BOOL), POINTER(fc2PGRGuid), POINTER(c_uint)]
    fc2ShowModal.restype = None
    break

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 47
try:
    FALSE = 0
except:
    pass

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 51
try:
    TRUE = 1
except:
    pass

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 55
try:
    FULL_32BIT_VALUE = 2147483647
except:
    pass

# /usr/include/flycapture/C/FlyCapture2Defs_C.h: 58
try:
    MAX_STRING_LENGTH = 512
except:
    pass

_fc2PGRGuid = struct__fc2PGRGuid # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 94

_fc2Image = struct__fc2Image # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 482

_fc2SystemInfo = struct__fc2SystemInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 499

_fc2Version = struct__fc2Version # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 507

_fc2Config = struct__fc2Config # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 523

_fc2PropertyInfo = struct__fc2PropertyInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 543

_Property = struct__Property # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 562

_fc2TriggerModeInfo = struct__fc2TriggerModeInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 576

_fc2TriggerMode = struct__fc2TriggerMode # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 587

_fc2StrobeInfo = struct__fc2StrobeInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 600

_fc2StrobeControl = struct__fc2StrobeControl # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 611

_fc2Format7ImageSettings = struct__fc2Format7ImageSettings # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 623

_fc2Format7Info = struct__fc2Format7Info # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 643

_fc2Format7PacketInfo = struct__fc2Format7PacketInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 652

_fc2IPAddress = struct__fc2IPAddress # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 657

_fc2MACAddress = struct__fc2MACAddress # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 662

_fc2GigEProperty = struct__fc2GigEProperty # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 674

_fc2GigEStreamChannel = struct__fc2GigEStreamChannel # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 687

_fc2GigEConfig = struct__fc2GigEConfig # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 699

_fc2GigEImageSettingsInfo = struct__fc2GigEImageSettingsInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 713

_fc2GigEImageSettings = struct__fc2GigEImageSettings # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 724

_fc2TimeStamp = struct__fc2TimeStamp # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 735

_fc2ConfigROM = struct__fc2ConfigROM # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 752

_fc2CameraInfo = struct__fc2CameraInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 797

_fc2EmbeddedImageInfoProperty = struct__fc2EmbeddedImageInfoProperty # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 804

_fc2EmbeddedImageInfo = struct__fc2EmbeddedImageInfo # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 819

_fc2ImageMetadata = struct__fc2ImageMetadata # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 835

_fc2LUTData = struct__fc2LUTData # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 848

_fc2PNGOption = struct__fc2PNGOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 856

_fc2PPMOption = struct__fc2PPMOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 863

_fc2PGMOption = struct__fc2PGMOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 870

_fc2TIFFOption = struct__fc2TIFFOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 889

_fc2JPEGOption = struct__fc2JPEGOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 897

_fc2JPG2Option = struct__fc2JPG2Option # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 903

_fc2AVIOption = struct__fc2AVIOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 910

_fc2MJPGOption = struct__fc2MJPGOption # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 918

_fc2H264Option = struct__fc2H264Option # /usr/include/flycapture/C/FlyCapture2Defs_C.h: 929

# No inserted files

