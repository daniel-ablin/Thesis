def adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon, learning_rate):
    m = beta_1 * m + (1 - beta_1) * grad
    u = beta_2 * u + (1 - beta_2) * grad ** 2
    m_hat = m / (1 - beta_1 ** (itr + 1))
    u_hat = u / (1 - beta_2 ** (itr + 1))
    adam = (learning_rate * m_hat) / (u_hat ** 0.5 + epsilon)

    return adam, m, u