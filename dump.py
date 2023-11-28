from typing import Callable, List, Tuple, Union, Dict

from tinygrad import Tensor, TinyJit
from tinygrad.helpers import DType
from tinygrad.runtime.lib import RawBuffer


def _dump_kernels(jit_fn: TinyJit, special_names, name: str, scratch_heap: bool) -> Tuple[Dict[str, str], List[str], Dict[int, Tuple[str, int, DType]], Dict[str, RawBuffer], Dict[int, int]]:
  functions, bufs, bufnum, byte_offset, buf_offsets, statements, bufs_to_save = {}, {}, 0, 0, {}, [], {}
  for ji in jit_fn.jit_cache:
    functions[ji.prg.name] = ji.prg.prg.replace(ji.prg.name, f"{name}_{ji.prg.name}") # type: ignore

    cargs = []
    for i, buf in enumerate(ji.rawbufs):
      assert isinstance(buf, RawBuffer)
      if (key := id(buf)) not in bufs:
        if key in special_names: bufs[key] = (special_names[key], buf.size, buf.dtype)
        else:
          bufs[key] = (f"scratch_{bufnum}", buf.size, buf.dtype)
          bufnum += 1
          if i > 0:
            bufs_to_save[bufs[key][0]] = buf

            # offset into weights
            if key not in buf_offsets:
              buf_offsets[key] = byte_offset
              byte_offset += buf.size * buf.dtype.itemsize

      # use offset into weights
      if key in special_names or bufs[key][0] not in bufs_to_save:
        if "scratch" in bufs[key][0] and scratch_heap: cargs.append(f"({str(buf.dtype)[7:]}*)({name}->{bufs[key][0]})")
        else: cargs.append(bufs[key][0])
      else: cargs.append(f"({str(bufs[key][2])[7:]}*)({name}->weights + {buf_offsets[key]})")
    statements.append(f"{name}_{ji.prg.name}({', '.join(cargs)});") # type: ignore
  return functions, statements, bufs, bufs_to_save, buf_offsets

def _dump(jit_fn: TinyJit, special_names, name: str, scratch_heap: bool, scratch_static: bool) -> Tuple[str, str, bytearray, Dict[int, Tuple[int, int]]]:
  functions, statements, bufs, bufs_to_save, buf_offsets = _dump_kernels(jit_fn, special_names, name, scratch_heap)
  scratch_buffers = [(str(dtype)[7:], name, len) for name, len, dtype in bufs.values() if name not in special_names.values() and name not in bufs_to_save]

  # build header and source
  h, c = "", ""

  # initial header
  h += f"""
#pragma once
#include <stdlib.h>
#include <math.h>

#define max(x,y) ((x>y)?x:y)
#define uint8 unsigned char
#define half __fp16

typedef struct {{
  void *weights;
  size_t weights_len;

{f"{chr(10)}".join(f"  {dtype} *{bname};" for dtype, bname, _ in scratch_buffers) if scratch_heap else ""}
}} {name}_t;

{name}_t *{name}_init(void *weights, size_t weights_len);

void {name}_free({name}_t *{name});
"""

  # fn signature
  fn_args = sorted([f"{'const ' if 'input' in special_names[key] else ''}{str(bufs[key][2])[7:]}* restrict {bufs[key][0]}" for key in bufs if key in special_names])
  h += f"""
void {name}_fn({name}_t *{name}, {", ".join(fn_args)});
"""

  # c header
  c += f"""
#include "{name}.h"

{name}_t *{name}_init(void *weights, size_t weights_len) {{
  {name}_t *{name} = malloc(sizeof({name}_t));
  {name}->weights = weights;
  {name}->weights_len = weights_len;

  // scratch buffers
{f"{chr(10)}".join(f"  {name}->{bname} = malloc(sizeof({dtype}) * {len});" for dtype, bname, len in scratch_buffers) if scratch_heap else ""}

  return {name};
}}

void {name}_free({name}_t *{name}) {{
{f"{chr(10)}".join(f"  free({name}->{bname});" for _, bname, _ in scratch_buffers) if scratch_heap else ""}
  free({name});
}}

"""

  # functions
  for function in functions.values():
    c += f"{function}\n"

  # fn definition
  c += f"""
void {name}_fn({name}_t *{name}, {", ".join(fn_args)}) {{
"""
  if not scratch_heap:
    for dtype, bname, len in scratch_buffers:
      c += f"  {'static ' if scratch_static else ''}{dtype} {bname}[{len}];\n"
  c += f"\n"
  for statement in statements:
    c += f"  {statement}\n"
  c += f"}}\n"

  # weights
  weights, weight_map = bytearray(), {}
  for buf in bufs_to_save.values():
    weights += buf.toCPU().tobytes()
    weight_map[buf_offsets[id(buf)]] = (buf.size * buf.dtype.itemsize, id(buf))

  return h, c, weights, weight_map

def dump(fn: Callable[..., Union[List[Tensor], Tuple[Tensor, ...], Tensor]], params: List[Tensor], args: List[Tensor], name: str, scratch_heap: bool = True, scratch_static: bool = False) -> Tuple[str, str, bytearray, Dict[int, Tuple[int, int]]]:
  @TinyJit
  def jit_fn(*x) -> List[Tensor]:
    out = fn(*x)
    out = [out] if isinstance(out, Tensor) else out
    return [o.realize() for o in out] # type: ignore

  # run twice to initialize
  for param in params: param.realize()
  jit_fn(*args)
  output = jit_fn(*args)

  # fix stuff
  for (j, i), idx in jit_fn.input_replace.items():
    jit_fn.jit_cache[j].rawbufs[i] = args[idx].lazydata.realized

  # get special names
  special_names = {}
  for (_, i), idx in jit_fn.input_replace.items():
    special_names[id(args[idx].lazydata.realized)] = f"input_{i}"
  for i, out in enumerate(output):
    special_names[id(out.lazydata.realized)] = f"output_{i}"

  # dump
  return _dump(jit_fn, special_names, name, scratch_heap, scratch_static)
