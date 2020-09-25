from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
import numpy as np

torch, nn = try_import_torch()


class ResidualBlock(nn.Module):
    def __init__(self, channels, init_normed=False):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        if init_normed:
            self.conv0.weight.data *= 1 / self.conv0.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
            nn.init.zeros_(self.conv0.bias)
            self.conv1.weight.data *= 1 / self.conv1.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
            nn.init.zeros_(self.conv1.bias)

    
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, init_normed=False):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels, init_normed)
        self.res_block1 = ResidualBlock(self._out_channels, init_normed)
        if init_normed:
            self.conv.weight.data *= 1 / self.conv.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, pool=True):
        x = self.conv(x)
        if pool:
            x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaCNN(TorchModelV2, nn.Module):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, device):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.device = device
        depths = model_config['custom_options'].get('depths') or [16, 32, 32]
        nlatents = model_config['custom_options'].get('nlatents') or 256
        init_normed = model_config['custom_options'].get('init_normed') or False
        self.use_layernorm = model_config['custom_options'].get('use_layernorm') or True
        
        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in depths:
            conv_seq = ConvSequence(shape, out_channels, init_normed)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=nlatents)
        if init_normed:
            self.hidden_fc.weight.data *= 1.4 / self.hidden_fc.weight.norm(dim=1, p=2, keepdim=True)
            nn.init.zeros_(self.hidden_fc.bias)
        self.pi_fc = nn.Linear(in_features=nlatents, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=nlatents, out_features=1)
        if init_normed:
            self.pi_fc.weight.data *= 0.1 / self.pi_fc.weight.norm(dim=1, p=2, keepdim=True)
            self.value_fc.weight.data *= 0.1 / self.value_fc.weight.norm(dim=1, p=2, keepdim=True)
        else:
            nn.init.orthogonal_(self.pi_fc.weight, gain=0.01)
            nn.init.orthogonal_(self.value_fc.weight, gain=1)
        nn.init.zeros_(self.pi_fc.bias)
        nn.init.zeros_(self.value_fc.bias)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(nlatents)

    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = self.conv_seqs[0](x)
        x = self.conv_seqs[1](x)
        x = self.conv_seqs[2](x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        if self.use_layernorm:
            x = self.layernorm(x)
            x = torch.tanh(x)
        else:
            x = nn.functional.relu(x)
        logits = self.pi_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value
    
    def vf_pi(self, obs, no_grad=False, ret_numpy=False, to_torch=False):
        if to_torch:
            obs = torch.tensor(obs).to(self.device)
            
        def v_pi(obs):
            pi, _ = self.forward({"obs": obs}, None, None)
            v = self.value_function()
            return v, pi
        
        if no_grad:
            with torch.no_grad():
                v, pi = v_pi(obs)
        else:
            v, pi = v_pi(obs)
        
        if ret_numpy:
            return v.cpu().numpy(), pi.cpu().numpy()
        else:
            return v, pi
    
ModelCatalog.register_custom_model("impala_torch_custom", ImpalaCNN)
