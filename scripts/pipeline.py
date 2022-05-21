import subprocess

if __name__ == '__main__':
    subprocess.call('python preprocess.py', shell=True)
    subprocess.call('python predict.py', shell=True)