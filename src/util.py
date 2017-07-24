import progressbar

import tensorflow as tf


def _create_progress_bar(dynamic_msg=None):
    
    widgets = [
        ' [batch ', progressbar.SimpleProgress(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') '
    ]

    if dynamic_msg is not None:
        widgets.append(progressbar.DynamicMessage(dynamic_msg))

    return progressbar.ProgressBar(widgets=widgets)




def non_zero_tokens(tokens):
    """Receives a vector of tokens (float) which are zero-padded. Returns a vector of the same size, which has the value
    1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))