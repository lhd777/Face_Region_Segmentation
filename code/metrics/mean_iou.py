from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np


class EvaluationMetric():
    def evaluate(self, true_label, pred_label):
        f1 = f1_score(true_label, pred_label, average='macro')
        acc = accuracy_score(true_label, pred_label)
        cm = confusion_matrix(true_label, pred_label)
        IoU = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))
        # 只计算非背景类的meanIoU
        mIoU = np.mean(IoU[1:])

        return f1, acc, mIoU