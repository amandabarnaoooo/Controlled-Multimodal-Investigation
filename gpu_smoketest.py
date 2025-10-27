import torch, os, time
print("Python:", os.sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
# do a tiny tensor op on GPU if available
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.matmul(x, x.T)
    print("Matmul done with shape:", y.shape)
print("Done.")
time.sleep(1)
