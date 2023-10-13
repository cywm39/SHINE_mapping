import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import SHINEConfig


class SDFDecoder(nn.Module):
    def __init__(self, config: SHINEConfig, is_time_conditioned = False): 
        
        super().__init__()
        
        mlp_hidden_dim = config.sdf_mlp_hidden_dim
        mlp_bias_on = config.sdf_mlp_bias_on
        mlp_level = config.sdf_mlp_level

        input_layer_count = config.feature_dim
        if is_time_conditioned:
            input_layer_count += 1

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)
        # self.bn = nn.BatchNorm1d(self.hidden_dim, affine=False)

        self.to(config.device)
        # torch.cuda.empty_cache()

    def forward(self, feature):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.predict_sdf(feature)
        return output

    # predict the sdf (opposite sign to the actual sdf)
    def predict_sdf(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out
    
    def time_conditionded_sdf(self, sum_features, ts):
        time_conditioned_feature = torch.torch.cat((sum_features, ts.view(-1, 1)), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(time_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

