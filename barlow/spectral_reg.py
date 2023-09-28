import torch


def get_pca_variance(data):
    """calculate the eigenvalues of the covariance matrix"""
    data = data.view(data.size(0), -1).type(torch.float32)
    normalized_data = data - data.mean(0)

    # Finding the Eigen Values and Vectors for the data
    sigma = torch.matmul(torch.transpose(normalized_data, 0, 1), normalized_data)

    eigen_values, eigen_vectors = torch.linalg.eigh(sigma)

    # resort (from big to small) and normalize sum to 1
    return eigen_values.flip(0) / torch.sum(eigen_values)


def fit_offset(x2, y2, target_alpha):
    m2 = -target_alpha
    t2 = y[2] - m2 * x[2]


def get_alpha_mse(data, target_alpha=1, minx=2, maxx=100):
    eigen_values = get_pca_variance(data)

    y = torch.log10(eigen_values)
    x = torch.log10(torch.arange(1, eigen_values.shape[0] + 1, 1.0, device=y.device))

    m2 = -target_alpha
    t2 = y[minx] - m2 * x[minx]

    x2 = x[minx:maxx]
    y2 = y[minx:maxx]

    pred_y = m2 * x2 + t2

    mse = torch.mean((pred_y - y2) ** 2)
    return mse


def get_alpha(data, min_x=2, max_x=100, weighting=True):
    eigen_values = get_pca_variance(data)

    y = torch.log10(eigen_values)
    x = torch.log10(torch.arange(1, eigen_values.shape[0] + 1, 1.0, device=y.device))

    x2 = x[min_x:max_x]
    y2 = y[min_x:max_x]

    """
    if not weighting:  # no weighting
        weights = x2*0+1
    else:  # logarithmic weighting
        weights = x[1:] - x[:-1]
        weights = weights[min_x:max_x]
    """

    m2, t2 = linear_fit(x2, y2)

    pred_y = m2 * x2 + t2

    mse = torch.mean((pred_y - y2) ** 2)
    return -m2, mse
    plt.plot(x, y)
    plt.plot(x2, pred_y)
    plt.show()


def linear_fit(x_data, y_data):
    """calculate the linear regression fit for a list of xy points."""
    x_mean = torch.mean(x_data)
    y_mean = torch.mean(y_data)
    m = torch.sum((x_data - x_mean) * (y_data - y_mean)) / torch.sum(
        (x_data - x_mean) ** 2
    )
    t = y_mean - (m * x_mean)
    return m, t
