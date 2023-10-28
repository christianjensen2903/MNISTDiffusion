from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

DIMS = 2048


def calculate_fid(pred, target, device):
    print("Calculating fid...")
    target = target.repeat(1, 3, 1, 1)
    pred = pred.repeat(1, 3, 1, 1)

    fid_calculator = FrechetInceptionDistance(normalize=True)
    fid_calculator = fid_calculator.to(device)
    fid_calculator.update(target, real=True)
    fid_calculator.update(pred, real=False)

    return fid_calculator.compute()


def calculate_ssim(pred, target):
    print("Calculating ssim...")
    ssim_calculator = StructuralSimilarityIndexMeasure()
    ssim_calculator.update(pred, target)
    return ssim_calculator.compute()
