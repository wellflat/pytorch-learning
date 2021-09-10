#!/usr/bin/env python
# -*- using: utf-8 -*-

import torch
from torch import package
from alexnet import AlexNet

model_path = "./cifar10_net.pth"
pt_path = "./alexnet_cifar10.pt"
package_name = "alexnet_cifar10"
resource_name = "model.pkl"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet()
model.load_state_dict(torch.load(model_path, map_location=device))
print(model)

#with package.PackageExporter(pt_path, verbose=False) as exporter:
    #exporter.intern("alexnet")
    #exp.extern("numpy.**")
    #exporter.save_pickle(package_name, resource_name, model)

importer = package.PackageImporter(pt_path)
print(importer.file_structure())
print(importer.get_name())
print(importer.get_source())
loaded_model = importer.load_pickle(package_name, resource_name)
#print(loaded_model)

