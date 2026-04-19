"""
Modular Model Converter with Huggingface Transformers

Usage:
    python ./thelethe/utils/converter.py --files-to-parse {Absolute Path}/theRiverLethe/thelethe/architectures/titans/models/origin/modular_origin.py
"""
import urllib.request
import types
import sys


url_base = "https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/utils"
converter = "modular_model_converter.py"
mapper = "create_dependency_mapping.py"
checker = "check_modular_conversion.py"


def import_from_url(remote_path, file_name, code_patch=lambda c: c):
    module_name = file_name.split(".")[0]
    with urllib.request.urlopen(remote_path + "/" + file_name) as response:
        code = code_patch(response.read().decode('utf-8'))
    module = types.ModuleType(file_name)
    exec(code, module.__dict__)
    sys.modules[module_name] = module
    return module


def run_from_url(remote_path, file_name, code_patch=lambda c: c, args=None):
    original_argv = sys.argv[:]

    if args is None:
        sys.argv = [file_name] + sys.argv[1:]
    elif args is False:
        sys.argv = [file_name]
    else:
        sys.argv = [file_name] + args

    try:
        response = urllib.request.urlopen(remote_path + "/" + file_name)
        code = code_patch(response.read().decode('utf-8'))
        print(">> Running script with", sys.argv)
        exec(code, globals())
    finally:
        sys.argv = original_argv


def run_converter(*args, **kwargs):
    import_from_url(url_base, mapper)

    def patch(c):
        return c.replace("src/transformers/.*|examples/.*", "src/transformers/.*|examples/.*|thelethe/architectures/.*")

    module = import_from_url(url_base, converter, patch)
    module.run_converter(*args, **kwargs)


if __name__ == '__main__':
    import_from_url(url_base, mapper)
    run_from_url(url_base, converter)
