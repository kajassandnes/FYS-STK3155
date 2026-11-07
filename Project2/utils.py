# useful functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

def identity(z):
    return z

def identity_der(z):
    return 1

def mse(predict, target):
    return jnp.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2 * (predict - target) / predict.size

def cross_entropy(predict, target):
    return jnp.sum(-target * jnp.log(predict))

def binary_cross_entropy(predict, target):
    p = jnp.clip(predict, 1e-12, 1 - 1e-12)
    return -jnp.sum(target * jnp.log(p) + (1 - target) * jnp.log1p(-p))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def smce_der(predict, target):
    return predict - target 

def one_hot(y, n_classes=None):
    """Integer labels -> one-hot rows."""
    n_classes = int(np.max(y)) + 1 if n_classes is None else n_classes
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# classification plotting:
def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    """C[i, j] = number of samples of true class i predicted as class j."""
    Cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(Cm, (np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)), 1)
    return Cm

def per_class_from_confusion(Cm):
    """Precision and recall of every class from the K x K matrix."""
    tp = np.diag(Cm).astype(float)
    return tp / np.maximum(Cm.sum(axis=0), 1), tp / np.maximum(Cm.sum(axis=1), 1)

def plot_classification(y, p, threshold = 0.5, label = "your classifier"):
      y_test, p_test = y, p          # replace by your test labels and output probabilities
      y_pred = p_test > threshold

      cm = confusion_matrix(y_test, y_pred)   # [[TN, FP], [FN, TP]]
      print(cm)
      print(f"accuracy {accuracy_score(y_test, y_pred):.4f}  precision {precision_score(y_test, y_pred, zero_division=0):.4f}  "
            f"recall {recall_score(y_test, y_pred):.4f}  F1 {f1_score(y_test, y_pred):.4f}")
      print(classification_report(y_test, y_pred, target_names=["negative", "positive"], zero_division=0))

      fpr, tpr, thresholds = sk_roc_curve(y_test, p_test, drop_intermediate=False)
      print(f"AUC {roc_auc_score(y_test, p_test):.4f}")

      fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
      ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(ax=ax[0], colorbar=False)
      RocCurveDisplay.from_predictions(y_test, p_test, ax=ax[1], name = label)
      ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
      plt.tight_layout(); plt.show()