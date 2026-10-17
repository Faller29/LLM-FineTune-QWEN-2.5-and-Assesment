"""Run with Windows 11 Python 3.12 / CPU PyTorch, before downloading anything large."""
import importlib
import platform
import sys

print('Python:', sys.version.split()[0], platform.architecture()[0])
print('System:', platform.platform())
for name in ['torch', 'transformers', 'peft', 'huggingface_hub', 'fasttext']:
    try:
        module = importlib.import_module(name)
        print(name, ': OK', getattr(module, '__version__', ''))
        if name == 'torch':
            print('CPU tensor test:', (module.tensor([2]) + 2).item())
    except Exception as exc:
        print(name, ': ERROR:', repr(exc))
