import subprocess

subprocess.run("python prepro_csd.py", shell=True)
subprocess.run("python prepro_jingju.py", shell=True)
subprocess.run("python prepro_jsut.py", shell=True)
subprocess.run("python prepro_jvs.py", shell=True)
# subprocess.run('python prepro_librispeech.py',shell=True)
subprocess.run("python prepro_mir1k.py", shell=True)
subprocess.run("python prepro_k_multitimbre.py", shell=True)
subprocess.run("python prepro_k_multisinger.py", shell=True)
subprocess.run("python prepro_tonas.py", shell=True)
subprocess.run("python prepro_vocadito.py", shell=True)
subprocess.run("python prepro_vocalset.py", shell=True)
