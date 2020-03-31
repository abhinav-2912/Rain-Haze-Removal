def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def guided_filter(data, width=image_size, height=image_size, channel=num_channels):
    r = 15
    eps = 1.0
    batch_q = np.zeros((height, width, channel))
    for j in range(channel):
        I = data[:, :, j]
        p = data[:, :, j]
        ones_array = np.ones([height, width])
        N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
        mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        q = mean_a * I + mean_b
        batch_q[:, :, j] = q

    return batch_q
