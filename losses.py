import tensorflow.keras
import matplotlib.pyplot as plt
import sklearn.metrics as SKMmetrics
import seaborn as sn
import pandas as pd
from matplotlib import interactive
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from draw_confusion_matrix import plot_confusion_matrix_from_data


# Write a Loss History class to save the loss and acc of the training set
# Of course, I can not do this at all, I can directly use the history object returned by the model.fit() method to do it
class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.mse = {'batch': [], 'epoch': []}
        self.mae = {'batch': [], 'epoch': []}
        self.mape = {'batch': [], 'epoch': []}
        self.rmses = {'batch': [], 'epoch': []}
        self.cos_prox = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.val_mse = {'batch': [], 'epoch': []}
        self.val_mae = {'batch': [], 'epoch': []}
        self.val_mape= {'batch': [], 'epoch': []}
        self.val_rmse = {'batch': [], 'epoch': []}
        self.val_cos_prox = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.mse['batch'].append(logs.get('mean_squared_error'))
        self.mae['batch'].append(logs.get('mean_absolute_error'))
        self.mape['batch'].append(logs.get('mean_absolute_percentage_error'))
        self.rmses['batch'].append(logs.get('rmse'))
        self.cos_prox['batch'].append(logs.get('cosine_proximity'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))
        self.val_mse['batch'].append(logs.get('val_mean_squared_error'))
        self.val_mae['batch'].append(logs.get('val_mean_absolute_error'))
        self.val_mape['batch'].append(logs.get('val_mean_absolute_percentage_error'))
        self.val_rmse['batch'].append(logs.get('val_rmse'))
        self.val_cos_prox['batch'].append(logs.get('val_cosine_proximity'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.mse['epoch'].append(logs.get('mean_squared_error'))
        self.mae['epoch'].append(logs.get('mean_absolute_error'))
        self.mape['epoch'].append(logs.get('mean_absolute_percentage_error'))
        self.rmses['epoch'].append(logs.get('rmse'))
        self.cos_prox['epoch'].append(logs.get('cosine_proximity'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
        self.val_mse['epoch'].append(logs.get('val_mean_squared_error'))
        self.val_mae['epoch'].append(logs.get('val_mean_absolute_error'))
        self.val_mape['epoch'].append(logs.get('val_mean_absolute_percentage_error'))
        self.val_rmse['epoch'].append(logs.get('val_rmse'))
        self.val_cos_prox['epoch'].append(logs.get('val_cosine_proximity'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))

        plt.figure(1)
        plt.plot(iters, self.accuracy[loss_type], 'g', label='Training Accuracy')
        plt.plot(iters, self.cos_prox[loss_type], 'b', label='Cosine Proximity')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'r', label='Validation Accuracy')
            plt.plot(iters, self.val_cos_prox[loss_type], 'm', label='Validation Proximity')
        plt.grid(True);     plt.xlabel(loss_type);      plt.ylabel('Accuracy')
        plt.legend(loc="lower right");  interactive(True);  plt.show()

        plt.figure(2)
        plt.plot(iters, self.losses[loss_type], 'r', label='Train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'm', label='Validation loss')
        plt.grid(True); plt.xlabel(loss_type);  plt.ylabel('Loss')
        plt.legend(loc="upper right");      plt.show()

        plt.figure(3)
        plt.plot(iters, self.mae[loss_type], 'g', label='Mean Absolute Error')
        plt.plot(iters, self.mse[loss_type], 'b', label='Mean Square Error')
        plt.plot(iters, self.rmses[loss_type], 'r', label='Root Mean Square Error')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_mae[loss_type], 'c', label='Validation MAE')
            plt.plot(iters, self.val_mse[loss_type], 'k', label='Validation MSE')
            plt.plot(iters, self.val_rmse[loss_type], 'm', label='Validation RMSE')
        plt.grid(True);    plt.xlabel(loss_type);    plt.ylabel('Error')
        plt.legend(loc="upper right");    interactive(False);     plt.show()


def figPlot(y_true, y_pred, saveAs='foo.png', title=''):
    print(classification_report(y_true, y_pred))
    print("Accuracy: {0}".format(accuracy_score(y_true, y_pred)))

    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['PUSH', 'PULL', 'CLRT', 'CCRT', 'CRPH', 'CCRP']

    if (len(y_true) > 10):
        figsize = [14, 14];
    plot_confusion_matrix_from_data(y_true, y_pred, columns=categories, annot=True, cmap='Oranges', fmt='.2f', fz=18,
                                    lw=0.5, cbar=False, figsize=[9, 9], show_null_values=2, pred_val_axis='y',
                                    SaveFig = 1, saveAs=saveAs, title=title)

    #print(SKMmetrics.confusion_matrix(y_true, y_pred))

    # plot confusion matrix on heat map
    #labels = sorted(list(set(y_true)))
    #plt.figure(figsize=(10, 7))
    #cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    #df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)    
    #sn.heatmap(df_cmx, annot=True) 
    #plt.savefig(saveAs)
    #plt.draw()
    #plt.show(block=False)
    

def trainPlot(history, saveAs='foo.png'):
    if hasattr(history, 'history'):
        dAcc, dVaCC = history.history['acc'], history.history['val_acc']
        dloss, dVloss = history.history['loss'], history.history['val_loss']
        dRmse, dVrmse = history.history['rmse'], history.history['val_rmse']
        dMae, dVmae   = history.history['mae'], history.history['val_mae']
    else:
        if len(history)==10:
            dloss, dVloss = history[0][1], history[5][1]
            dAcc, dVaCC = history[1][1], history[6][1]
            dRmse, dVrmse = history[2][1], history[7][1]
            dMae, dVmae = history[3][1], history[8][1]
            dMse, dVmse = history[4][1], history[9][1]
        else:
            dloss, dVloss = history[0][1], history[4][1]
            dAcc, dVaCC = history[1][1], history[5][1]
            dRmse, dVrmse = history[2][1], history[6][1]
            dMae, dVmae = history[3][1], history[7][1]

    # plot history
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('Training and Validation Acuraccy')
    axs[0, 0].plot(dAcc, label='Training')
    axs[0, 0].plot(dVaCC, label='Validation')

    axs[0, 1].set_title('Training and Validation Loss')
    axs[0, 1].plot(dloss, label='Training')
    axs[0, 1].plot(dVloss, label='Validation')

    axs[1, 0].set_title('Training and Validation RMSE')
    axs[1, 0].plot(dRmse)
    axs[1, 0].plot(dVrmse)

    # axs[1, 0].set_title('Training and Validation Co-Proximity')
    # axs[1, 0].plot(history.history['cosine_proximity'], label='Training')
    # axs[1, 0].plot(history.history['val_cosine_proximity'], label='Validation')

    axs[1, 1].set_title('Training and Validation MAE')
    axs[1, 1].plot(dMae, label='Training')
    axs[1, 1].plot(dVmae, label='Validation')

    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Performace Measure')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()

        # cm_analysis = ConfusionMatrix(actual_vector=y_actl, predict_vector=y_pred)

    '''
    pyplot.imshow(cf_matrix, cmap=pyplot.cf_matrix.Blues)
    pyplot.xlabel("Predicted labels")
    pyplot.ylabel("True labels")
    pyplot.xticks([], [])
    pyplot.yticks([], [])
    pyplot.title('Confusion matrix ')
    pyplot.colorbar()
    pyplot.show()
    '''
    plt.legend()
    plt.savefig(saveAs, dpi=300)
    plt.show(block=True)
    
def trainPlot2(history, saveAs='foo.png'):
    if hasattr(history, 'history'):
        dAcc, dVaCC = history.history['acc'], history.history['val_acc']
        dloss, dVloss = history.history['loss'], history.history['val_loss']
        dRmse, dVrmse = history.history['rmse'], history.history['val_rmse']
        dMae, dVmae   = history.history['mae'], history.history['val_mae']
    else:
        dloss, dVloss = history[0][1], history[4][1]
        dAcc, dVaCC = history[1][1], history[5][1]
        dRmse, dVrmse = history[2][1], history[6][1]
        dMae, dVmae = history[3][1], history[7][1]

    # plot history
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('Training and Validation Acuraccy')
    axs[0, 0].plot(dAcc, label='Training')
    axs[0, 0].plot(dVaCC, label='Validation')

    axs[0, 1].set_title('Training and Validation Loss')
    axs[0, 1].plot(dloss, label='Training')
    axs[0, 1].plot(dVloss, label='Validation')

    axs[1, 0].set_title('Training and Validation RMSE')
    axs[1, 0].plot(dRmse)
    axs[1, 0].plot(dVrmse)

    # axs[1, 0].set_title('Training and Validation Co-Proximity')
    # axs[1, 0].plot(history.history['cosine_proximity'], label='Training')
    # axs[1, 0].plot(history.history['val_cosine_proximity'], label='Validation')

    axs[1, 1].set_title('Training and Validation MAE')
    axs[1, 1].plot(dMae, label='Training')
    axs[1, 1].plot(dVmae, label='Validation')

    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Performace Measure')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()

        # cm_analysis = ConfusionMatrix(actual_vector=y_actl, predict_vector=y_pred)

    '''
    pyplot.imshow(cf_matrix, cmap=pyplot.cf_matrix.Blues)
    pyplot.xlabel("Predicted labels")
    pyplot.ylabel("True labels")
    pyplot.xticks([], [])
    pyplot.yticks([], [])
    pyplot.title('Confusion matrix ')
    pyplot.colorbar()
    pyplot.show()
    '''
    plt.legend()
    plt.savefig(saveAs, dpi=300)
    plt.show(block=True)
