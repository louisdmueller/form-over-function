`module load devel/cuda/12.8`
`module load devel/python/3.12.3-gnu-14.2`
`conda create -n vLLM-test python=3.12 -y` # this should install it into /home/hd/hd_hd/hd_*/miniconda3/envs/vLLM-test
`conda activate vLLM-test`
`pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` do not use uv!
`pip install vllm`
`pip check` for me it showed one conflict: vllm 0.11.0 has requirement setuptools<80,>=77.0.3; python_version > "3.11", but you have setuptools 80.9.0.
`pip install "setuptools<80,>=77.0.3"`
`conda deactivate`

As I said this was installed in miniconda3/ and somehow miniforge was not able to find it
so I moved it to .conda/ folder
`cp -r /home/hd/hd_hd/hd_*/miniconda3/envs/vLLM-test/ /home/hd/hd_hd/hd_*/.conda/envs/`

This may be the reason why it did not work.

Remember to deactivate the env before submitting jobs that use the env

If you get this error: `vllm/vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda9SetDeviceEab`
Then something possibly went wrong. I also tried a long time debugging this error but I came to no solution. Rebuilding vLLM from source takes more than two hours on which I gave up on this idea. 

If you get this error: `ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory` then you are probably not using Conda. This error is the entire reason for using conda. 

