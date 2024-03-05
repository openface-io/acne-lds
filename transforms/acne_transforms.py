"""Class with image transforms."""

from torchvision import transforms


class AcneTransformsTorch:
    """A class for creating and applying transforms on images using Torchvision."""

    def __init__(self, train=True):
        """Create a list of transforms and then their composition with transforms.Compose()."""
        self.train = train
        if self.train:
            transforms_list = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        else:
            transforms_list = [transforms.Resize((224, 224))]

        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266], std=[0.2814769, 0.226306, 0.20132513]),
        ]

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, image):
        """Apply transforms to image."""
        return self.transform(image)
