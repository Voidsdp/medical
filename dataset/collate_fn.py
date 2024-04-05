import torch

def get_img_collate(transform):
    def demo_collate(batch):
        imgs, labels = zip(*batch)
        imgs = [transform(img) for img in imgs]
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)

        return imgs, labels
    
    return demo_collate

def get_img_co_collate(transform):
    def demo_collate(batch):
        imgs1, imgs2, labels = zip(*batch)
        imgs1 = [transform(img) for img in imgs1]
        imgs2 = [transform(img) for img in imgs2]  

        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)
        labels = torch.tensor(labels)

        return imgs1, imgs2, labels
    
    return demo_collate