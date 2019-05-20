import torch.nn as nn
import torch


class BlockBatchNorm2d(nn.Module):
    def __init__(self, num_features, blocks=None, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):
        super(BlockBatchNorm2d, self).__init__()
        if blocks is None:
            blocks = [2, 2]

        self.row = blocks[0]
        self.col = blocks[1]
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
                                 for _ in range(self.row * self.col)])

    def forward(self, x):
        block_width = x.shape[-1] // self.col
        block_height = x.shape[-2] // self.row
        results = []
        for i in range(self.row):
            for j in range(self.col):
                results.append(self.bn[i * self.col + j](
                    x[:, :, block_height * i:block_height * (i + 1), block_width * j:block_width * (j + 1)]))

        row_results = []
        for i in range(self.row):
            row_results.append(torch.cat(results[i*self.col:(i+1)*self.col], dim=3))
        out = torch.cat(row_results, dim=2)
        return out


if __name__ == '__main__':
    input = torch.rand(16, 32, 64, 72)
    model = BlockBatchNorm2d(32, [2, 3])
    out = model(input)
    print(out.shape)
    print('END')
