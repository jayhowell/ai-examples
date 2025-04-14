This is my repository for sample ai python scripts.

Download the scripts.


Granite Models - I'm using the 3Billion model from hugging face because anything more will have my OOMkiller kill it in linux on my laptop.  
This will work on most Modern laptops.  Please be aware that it's very slow.  You need to be patient. 

This demo does indeed run on a Lenovo P1 laptop with an I7 processor with 32 GB of memory.  
Create an environment, pip the dependencies and run the files.(assuming everyone is python3)

python -m venv granite_cpu_env
source granite_cpu_env/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate
