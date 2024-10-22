class TotalLoss:
    def __init__(self, supervised_branch, unsupervised_branch):
        self.supervised_branch = supervised_branch
        self.unsupervised_branch = unsupervised_branch

    def calculate_total_loss(self, x_p, y_p, x, y, fake_y, lambda_sup):
        l_supervised = self.supervised_branch.forward(x_p, y_p)
        l_unsupervised = self.unsupervised_branch.forward(x, fake_y)

        return l_unsupervised + lambda_sup * l_supervised
