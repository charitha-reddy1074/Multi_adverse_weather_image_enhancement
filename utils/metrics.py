from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(prediction, target):
    """Inputs are RGB float arrays in [0,1], HWC."""
    min_side=min(prediction.shape[0],prediction.shape[1])
    if min_side<3: raise ValueError("SSIM requires both image dimensions to be at least 3")
    win_size=min(7,min_side if min_side%2 else min_side-1)
    return (float(peak_signal_noise_ratio(target,prediction,data_range=1.0)),
            float(structural_similarity(target,prediction,data_range=1.0,channel_axis=2,
                                        win_size=win_size)))
