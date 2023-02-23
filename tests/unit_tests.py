import torch
import numpy as np
import unittest
import sys
import os
from src.climate_learn.models.modules.utils.metrics import categorical_loss

class TestCategoricalLoss(unittest.TestCase):
    
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
        loss_dict = categorical_loss(pred, y, transform, vars, lat, clim, log_postfix)
        
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