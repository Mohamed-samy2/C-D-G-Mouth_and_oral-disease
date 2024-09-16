import torch
import torchvision.models as models
import torch.nn as nn
from base_model import ImageEncoder

class ImageTransformer(nn.Module):
    def __init__(self, num_classes , num_sites, embedding_dim, nhead, dim_feedforward,dropout,n_layers,base ):
        super().__init__()
        
        self.encoder = ImageEncoder(base=base)
        self.site_embedding = nn.Embedding(num_sites, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim,num_classes)
        

    def forward(self, image, site):
        image_features = self.encoder(image)
        site_embedding = self.site_embedding(site)
        combined_features = image_features + site_embedding
        
        transformer_output = self.transformer_encoder(combined_features)
        logits = self.fc(transformer_output)
        return logits

