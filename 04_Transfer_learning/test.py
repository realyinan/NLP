import torch
import transformers

print("Transformers 版本：", transformers.__version__)
print("是否支持 GPU: ", torch.cuda.is_available())
print("当前 GPU 数量：", torch.cuda.device_count())
print("当前使用的设备：", torch.cuda.current_device())
print("GPU 名称：", torch.cuda.get_device_name(0))
