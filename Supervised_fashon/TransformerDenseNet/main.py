import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from Dataset import CustomDataset, load_data
from model import ImageTransformer 
from train import train
from base_model import device
from config import batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2, embedding_dim,dropout
from config import full_train_data_path, full_val_data_path, full_test_data_path

from helpful.vis_metrics import plots, DoAna

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    # Add arguments for each hyperparameter
    parser.add_argument('--num_classes', type=int, default=num_classes, help="Number of classes")
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=dropout, help="Dropout")
    parser.add_argument('--num_sites', type=int, default=num_sites, help="Number of sites")
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help="embedding_dim")
    parser.add_argument('--shape', type=int, default=224, help="Learning rate")
    
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument('--l2', type=float, default=l2, help="L2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")
    parser.add_argument('--gamma', type=float, default=gamma, help="Gamma")
    parser.add_argument('--optim', type=str, default="AdamW", help="Optimizer")

    parser.add_argument('--full_train_data_path', type=str, default=full_train_data_path, help="Full train data path")
    parser.add_argument('--full_val_data_path', type=str, default=full_val_data_path, help="Full validation data path")
    parser.add_argument('--full_test_data_path', type=str, default=full_test_data_path, help="Full test data path")
    parser.add_argument('--base', type=str, default='densenet', help="Base model")

    # Boolean flags
    parser.add_argument('--use_scheduler', action='store_true', help="Use scheduler")
    parser.add_argument('--freeze_base', action='store_true', help="Freeze Base True")
    parser.add_argument('--freeze', action='store_true', help="Freeze True")
    parser.add_argument('--to_freeze', type=int, default=0, help="parameters to freeze")
    parser.add_argument('--compile', action='store_true', help="Compile")

    return parser.parse_args()

args = parse_args()

    # Define the transformations based on the description provided
transform = transforms.Compose([
        transforms.Resize((args.shape, args.shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

test_transform = transforms.Compose([
    transforms.Resize((args.shape, args.shape)),
    transforms.ToTensor(),                                      # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

stra_train_data, idx_to_class, idx_to_site = load_data(args.full_train_data_path, False)
stra_test_data, _, _ = load_data(args.full_test_data_path, False)
stra_val_data, _, _ = load_data(args.full_val_data_path, False)
print(idx_to_site)

print(f"Used Device: {device}"  )

train_set = CustomDataset(stra_train_data, transform, "train_distribution", oversample =False, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=False, ignore=False)
val_set = CustomDataset(stra_test_data, transform, "val_distribution", oversample = False, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=False, ignore=False)
test_set = CustomDataset(stra_test_data, test_transform, title = "test_distribution", oversample=False, ignore=False)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory =True)

torch.cuda.empty_cache()

model = ImageTransformer(num_classes=args.num_classes,
                        base= args.base,
                        freeze_base=args.freeze_base)

model = nn.DataParallel(model).to(device)

if args.compile:
    model = torch.compile(model=model) # use for training not debugging

if args.optim == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2,betas=(0.9,0.95),eps=1e-8,fused=True)
    
elif args.optim=='RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate,weight_decay=args.l2)
    
elif args.optim=='Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate,weight_decay=args.l2)
    
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate ,  weight_decay=args.l2 , betas=(0.9,0.95),eps=1e-8)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = sche_milestones, gamma = args.gamma)
criterion = nn.CrossEntropyLoss()

train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss = train(model, criterion, optimizer, scheduler, train_loader, val_loader, args.num_epochs, args.base, args.freeze,args.to_freeze,args.use_scheduler)

plots(train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss, idx_to_class, idx_to_site, num_classes)
DoAna(model, test_loader, idx_to_class, idx_to_site)