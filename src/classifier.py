import torch 
import torch.nn as nn

class GestureClassifier(nn.Module):
    def __init__(self,input_size=63,num_classes=4):
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(input_size,128), nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128,64), nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,num_classes),
        )

    def forward(self,x):
        return self.network(x)


if __name__ == "__main__":
    model = GestureClassifier()

    example = torch.randn(8, 63)
    output = model(example)

    print(model)
    print("Input shape:", example.shape)
    print("Output shape:", output.shape)