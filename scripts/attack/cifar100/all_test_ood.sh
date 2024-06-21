#!/bin/bash

python scripts/attack_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --att pgd \
   --eps "8/255"
   
python scripts/attack_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --att fgsm \
   --eps "8/255"

python scripts/attack_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --att df \
   --eps "8/255"

   python scripts/attack_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --att mpgd \
   --masked-patch-size 8
