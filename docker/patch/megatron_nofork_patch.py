"""Patch Megatron's async dist-checkpoint writer to run in-process (no fork).

Works around a ROCm 7.0.2 fork-after-HIP-init segfault in FileSystemWriterAsync
that breaks HF->torch_dist convert and training checkpoint saves. Idempotent.
"""
import py_compile

F = "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"

with open(F) as fh:
    src = fh.read()

if "write(sync,inproc)" in src:
    print("megatron no-fork patch already applied; skipping")
    raise SystemExit(0)

with open(F + ".bak", "w") as fh:
    fh.write(src)

start_marker = "    @staticmethod\n    @_disable_gc()\n    def write_preloaded_data_multiproc("
next_marker = "    @staticmethod\n    @_disable_gc()\n    def write_preloaded_data("
i = src.index(start_marker)
j = src.index(next_marker)
assert i < j, (i, j)

new_method = '''    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc(
        transform_list,
        use_msc,
        rank,
        write_buckets,
        global_results_queue,
    ) -> None:
        """In-process (no-fork) writer. ROCm 7.0.2 segfaults when the async
        checkpoint writer is run in a fork()ed subprocess (fork after HIP init).
        Write each bucket sequentially in the current process instead."""
        import queue as _q

        logger = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc = dict()
        results_queue = _q.Queue()
        count_queue = _q.Queue()
        try:
            for i, write_bucket in enumerate(write_buckets):
                count_queue.put(i)
                kwargs = {
                    "local_proc_idx": i,
                    "write_bucket": write_bucket,
                    "results_queue": results_queue,
                    "count_queue": count_queue,
                    "use_fsync": True,
                }
                if use_msc:
                    import inspect

                    sig = inspect.signature(FileSystemWriterAsync.write_preloaded_data)
                    if len(sig.parameters) > 6:
                        kwargs["use_msc"] = use_msc
                FileSystemWriterAsync.write_preloaded_data(transform_list, **kwargs)
            for _ in range(len(write_buckets)):
                local_proc_idx, local_results_or_exc = results_queue.get()
                if isinstance(local_results_or_exc, Exception):
                    write_results_or_exc = local_results_or_exc
                    break
                write_results_or_exc[local_proc_idx] = local_results_or_exc
        except Exception as e:
            write_results_or_exc = e
        global_results_queue.put(write_results_or_exc)
        w_end = time()
        logger.debug(f"{w_end}, rank: {rank}, write(sync,inproc): {w_end - w_start}")

'''

with open(F, "w") as fh:
    fh.write(src[:i] + new_method + src[j:])

py_compile.compile(F, doraise=True)
print("megatron no-fork patch applied OK")
