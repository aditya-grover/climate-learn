import torch
import numpy as np
import unittest
import sys
import os
from src.climate_learn.models.modules.utils.metrics import categorical_loss

class TestCategoricalLoss(unittest.TestCase):
    
    def categorical_loss(self, pred, y, transform, vars, lat, clim, log_postfix):
        """
        y: [B, 50, C, H, W]
        pred: [B, 50, C, H, W]
        vars: list of variable names
        lat: H
        """
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        # get the labels [128, 1, 32, 64]
        _, labels = y.max(dim=1) # y.shape = pred.shape = [128, 50, 1, 32, 64] 
        error = loss(pred, labels.to(pred.device)) # error.shape [128, 1, 32, 64]
        print(error.shape)

        # lattitude weights
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = (
            torch.from_numpy(w_lat)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(dtype=error.dtype, device=error.device)
        )

        print(w_lat.shape)

        loss_dict = {}
        with torch.no_grad():
            for i, var in enumerate(vars):
                loss_dict[f"categorical_{var}_{log_postfix}"] = torch.mean(error[:, i] * w_lat)

        loss_dict["w_categorical"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

        return loss_dict

    def test_categorical_loss(self):
        # create test input tensors
        batch_size = 128
        num_bins = 50
        num_channels = 2
        height = 32
        width = 64
        pred = torch.randn(batch_size, num_bins, num_channels, height, width)
        y = torch.randint(low=0, high=num_bins, size=(batch_size, num_bins, num_channels, height, width))
        
        # call the categorical_loss function
        transform = None
        vars = ["var1", "var2"]
        lat = np.random.rand(32)
        clim = None
        log_postfix = "test"
        loss_dict = self.categorical_loss(pred, y, transform, vars, lat, clim, log_postfix)
        
        # check the shape of the output dictionary
        self.assertEqual(len(loss_dict), len(vars) + 1) # +1 for "w_categorical" key
        
        # check the shape and type of each loss
        for var in vars:
            loss_key = f"categorical_{var}_{log_postfix}"
            self.assertIn(loss_key, loss_dict)
            self.assertIsInstance(loss_dict[loss_key], torch.Tensor)
            self.assertEqual(loss_dict[loss_key].shape, torch.Size([]))
        
        # check the shape and type of the mean loss
        self.assertIn("w_categorical", loss_dict)
        self.assertIsInstance(loss_dict["w_categorical"], np.float32)

        print(loss_dict)