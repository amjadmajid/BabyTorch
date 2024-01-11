class LambdaLR:
    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda

    def step(self, epoch):
        # Apply the lambda function to the epoch to get the learning rate factor
        lr_factor = self.lr_lambda(epoch)

        # Multiply the initial learning rate of each parameter group by the factor
        self.optimizer.learning_rate = lr_factor
