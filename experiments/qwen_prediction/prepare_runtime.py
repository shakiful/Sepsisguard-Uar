"""Remove an unused, incompatible Colab optional quantizer before PEFT import."""
import importlib.metadata
import subprocess
import sys
from packaging.version import Version


def main():
    try: version = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('Optional torchao is absent; BF16 LoRA does not require it.')
        return
    if Version(version) < Version('0.16.0'):
        print(f'Removing unused torchao {version}: PEFT 0.21 requires >=0.16 if torchao is installed.',flush=True)
        subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
    else:
        print('Optional torchao version accepted:',version)


if __name__ == '__main__': main()
