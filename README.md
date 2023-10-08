# kv260-vitis-flow
Self contained example of training, inspecting, and quantizing a model for the KV260 DPU

https://github.com/Xilinx/Vitis-AI/tree/v3.5/src/vai_library

## Compiling the model
After quantizing and exporting the model still needs to be compiled for the specific DPU arch
Do this after you get your xmodel from pt2
https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/customize_ai_models.html

arch.json (get it from xdputil query)
```
{
    "fingerprint":"0x101000016010406"
}
```
Then compile to produce deploy.xmodel and md5sum.txt
```
vai_c_xir -x /path/to/xmodel -a /path/to/arch.json
```

idk about compatibility guarantees but model quantized and compiled by Vitis 3.5
worked on the KV260 image that's on Vitis runtime/library 2.5
