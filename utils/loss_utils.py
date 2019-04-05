import tensorflow as tf


def cross_entropy(labels_tensor, logits_tensor):
    diff = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_tensor, labels=labels_tensor)
    loss = tf.reduce_mean(diff)
    return loss


def weighted_cross_entropy(y, logits, n_class):
    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(y, [-1, n_class])
    # your class weights
    class_weights = tf.constant([[9.,  1.49328859, 19.48905109, 30.68965517,  9.11262799, 87.06521739, 70.88495575]])
    # class_weights = tf.constant([[2*9.,  1.49328859, 2*19.48905109, 2*30.68965517,  2*9.11262799, 2*87.06521739, 2*70.88495575]])
    weighted_losses = tf.nn.weighted_cross_entropy_with_logits(targets=flat_labels, logits=flat_logits,
                                                               pos_weight=class_weights)
    loss = tf.reduce_mean(weighted_losses)
    return loss
