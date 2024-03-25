import torch

def get_img_collate(transform):
    def demo_collate(batch):
        imgs, labels = zip(*batch)
        imgs = [transform(img) for img in imgs]
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)

        return imgs, labels
    
    return demo_collate