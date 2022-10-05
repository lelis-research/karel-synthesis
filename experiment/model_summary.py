import sys
from torchsummary import summary
# Apply this commit to make summary work with multiple inputs:
# https://github.com/sksq96/pytorch-summary/commit/eb07421c49ddc99f943577bef9f0792d4e6a89b2#diff-ebda1cc7f304708e45ef4e19fb0484036eff8eb3c4b47a2886ca1cf0f731c0bb

sys.path.insert(0, '.')

from experiment.model import StatePredictor

if __name__ == '__main__':

    model = StatePredictor()
    summary(model, [(16, 8, 8), (256,)], 64)
    print(model)