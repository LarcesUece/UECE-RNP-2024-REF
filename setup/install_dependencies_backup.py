
import subprocess
import sys
import os

def run_command(command, description):
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(f"Saída: {e.stdout}")
        if e.stderr:
            print(f"Erro: {e.stderr}")
        return False

def test_tensorflow():
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 {len(gpus)} GPU(s) detectada(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("Nenhuma GPU detectada - usando CPU")
        
        return True
    except ImportError as e:
        print(f"Erro ao importar TensorFlow: {e}")
        return False

def main():
    basic_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "scikit-learn", "statsmodels", "jupyter"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Instalando {package}"):
            print(f"Falha ao instalar {package}, continuando...")
    
    tensorflow_strategies = [
        ("tensorflow[and-cuda]", "TensorFlow com GPU (automático)"),
        ("tensorflow==2.15.0", "TensorFlow versão estável"),
        ("tensorflow==2.14.0", "TensorFlow versão anterior"),
        ("tensorflow", "TensorFlow CPU apenas")
    ]
    
    tensorflow_installed = False
    
    for package, description in tensorflow_strategies:
        if run_command(f"pip install {package}", description):
            if test_tensorflow():
                tensorflow_installed = True
                break
            else:
                run_command(f"pip uninstall -y tensorflow tensorflow-gpu", "Removendo versão problemática")
    
    if not tensorflow_installed:
        return False
    
    extra_packages = ["xgboost", "psutil"]
    
    for package in extra_packages:
        run_command(f"pip install {package}", f"Instalando {package}")
    
    if test_tensorflow():
        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("concluído com sucesso!")
    else:
        print("concluído com erros")
