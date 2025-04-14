# AI Examples

This is my repository for sample ai python scripts.



## granitepytorch Project

This project does not require an nvidia process and doesn't use vllm(becuase it required cuda - nvidia processor)
It works with just an intel projecessor.  This will not work with an AMD chipset. 

Granite Models - I'm using the 3Billion IBM model(called ganite) from hugging face because the 8Billion granite model kills my system. My OOMkiller kill it in linux on my laptop.  
This example will work on most Modern laptops.  Please be aware that it's very slow.  You need to be patient. 

This demo does indeed run on a Lenovo P1 laptop with an I7 processor with 32 GB of memory.  
Create an environment, pip the dependencies and run the files.(assuming everyone is python3)

```
python -m venv granite_cpu_env
source granite_cpu_env/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate
cd granitepytorch
python
```
