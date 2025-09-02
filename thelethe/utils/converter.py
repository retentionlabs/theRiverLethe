import urllib.request
import sys


def run_remote_py(remote_path, file_name, args=None):
    if args is None:
        args = []

    original_argv = sys.argv[:]
    sys.argv = [file_name] + args

    try:
        response = urllib.request.urlopen(remote_path + "/" + file_name)
        code = response.read().decode('utf-8')
        exec(code, {'__name__': '__main__'})
    finally:
        sys.argv = original_argv



if __name__ == '__main__':
    url_base = "https://github.com/huggingface/transformers/blob/main/utils"
    converter = "modular_model_converter.py"
    checker = "check_modular_conversion.py"

    run_remote_py(url_base, "https://github.com/huggingface/transformers/blob/main/utils/modular_model_converter.py")
