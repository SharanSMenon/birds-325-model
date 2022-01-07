import torch

if __name__ == "__main__":
    loaded = torch.jit.load("../birds-325-deit-tiny-patch16-224.zip")
    state_dict = loaded.state_dict()
    torch.save(state_dict, "../birds-325-deit-tiny-patch16-224.pth")