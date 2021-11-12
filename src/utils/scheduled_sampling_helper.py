

def exponential_ss_prob(epoch, slope, min_value):
    return max(min_value, slope ** epoch)