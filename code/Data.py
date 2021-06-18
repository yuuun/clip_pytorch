import torchvision.transforms as transforms
import torchvision.datasets as dset
import random
from torch.utils.data import random_split, DataLoader

class Image_Data(object):
    def __init__(self, BATCH_SIZE):
        tfms = transforms.Compose([transforms.Resize((128, 128)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                        std=(0.229, 0.224, 0.225))
                                ])
        inv_tfm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                    ])

        #target_tfm = lambda x: random.choice(x)
        target_tfm = self.rand_choice
        
        self.cap = dset.CocoCaptions(root = '../datasets/train2014/',
                                annFile = '../datasets/annotations/captions_train2014.json',
                                transform=tfms,
                                target_transform=target_tfm,
        )

        self.split_data(BATCH_SIZE)

    def rand_choice(self, x):
        return random.choice(x)

    def split_data(self, BATCH_SIZE):
        self.train_len = int(0.8 * len(self.cap))
        self.train_data, self.valid_data = random_split(self.cap, [self.train_len, len(self.cap) - self.train_len])
        
        self.train_dl = DataLoader(self.train_data, BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=4, drop_last=True)
        self.valid_dl = DataLoader(self.valid_data, BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=4, drop_last=False)
