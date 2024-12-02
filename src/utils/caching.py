import os
import hashlib
import json
import pathlib
from functools import wraps

import joblib


def cached(
    cache_dir_path: str | None = None,
    max_size: int = -1,
    max_entries: int = -1,
    compress: int = 3,
    verbose: bool = False,
):
    def wrapper(func):
        class MethodDecoratorAdapter(object):
            def __init__(self, func):
                self.func = func
                self.is_method = False
                self.owner = ""

            def __get__(self, instance, owner):
                if not self.is_method:
                    self.is_method = True
                self.instance = instance
                self.owner = owner.__name__

                return self

            def __call__(self, *args, **kwargs):
                cache_key = json.dumps(
                    {
                        "name": func.__name__,
                        "class": self.owner,
                        "args": args,
                        "kwargs": kwargs,
                    },
                    default=str,
                    sort_keys=True,
                )
                h = hashlib.sha256(cache_key.encode()).hexdigest()
                if verbose:
                    print(f"Cache key: {cache_key} hash: {h}")

                if self.is_method:
                    name = f"{self.owner}.{func.__name__}"
                else:
                    name = func.__name__

                # try from args, env var or default
                if cache_dir_path:
                    cache_dir = pathlib.Path(cache_dir_path)
                else:
                    cache_dir = pathlib.Path(os.getenv("CACHE_DIR", ".cache"))

                path = cache_dir / f"{name}_{str(h)}"

                if path.exists():
                    if verbose:
                        print("Cache hit")
                    data = joblib.load(path)
                else:
                    if verbose:
                        print("Cache miss")

                    cache_dir.mkdir(parents=True, exist_ok=True)
                    if self.is_method:
                        data = self.func(self.instance, *args, **kwargs)
                    else:
                        data = self.func(*args, **kwargs)
                    joblib.dump(data, path, compress=compress)

                if max_size > 0 or max_entries > 0:
                    file_details = []
                    for file in cache_dir.glob("*"):
                        if file == path:
                            continue

                        stat = file.stat()
                        file_details.append(
                            {
                                "path": file,
                                "size": stat.st_size,
                                "last_access": stat.st_atime,
                            }
                        )

                    file_details.sort(
                        key=lambda x: (x["last_access"], x["size"]), reverse=True
                    )

                    removed_files = []
                    if max_size > 0:
                        total_size = sum(file["size"] for file in file_details)
                        while total_size > max_size and file_details:
                            file_to_remove = file_details.pop()
                            file_to_remove["path"].unlink()
                            removed_files.append(file_to_remove["path"])
                            total_size -= file_to_remove["size"]

                    if max_entries > 0:
                        file_count = len(file_details)
                        if file_count > max_entries - 1:
                            while file_count > max_entries - 1 and file_details:
                                file_to_remove = file_details.pop()
                                file_to_remove["path"].unlink()
                                removed_files.append(file_to_remove["path"])
                                file_count -= 1

                    if removed_files:
                        print(f"GC: Removed the following files:")
                        for file in removed_files:
                            print(file)
                    else:
                        if verbose:
                            print(f"No files removed")

                return data

        return wraps(func)(MethodDecoratorAdapter(func))

    return wrapper
