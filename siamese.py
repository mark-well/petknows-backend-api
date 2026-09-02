import torch.nn as nn
import torchvision.models as models

#Model
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

class SiamseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SiameseEncoder()

    def forward(self, image1, image2):
        embedding1 = self.encoder(image1)
        embedding2 = self.encoder(image2)

        return embedding1, embedding2
    
    def get_embedding(self, image):
        return self.encoder(image)
    
class SiameseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #Resnet as the feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(512, 128)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)

        #Normalize the embedding
        x = nn.functional.normalize(x, p=2, dim=1)
        return x