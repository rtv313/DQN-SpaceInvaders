from tensorflow.keras.models import save_model
import tensorflow as tf
import os


class SaveLoadModule:
    @staticmethod
    def save_nn_model(model, episode, epsilon):
        if not os.path.exists('models'):
            os.mkdir('models')
        save_model(model, 'models/' + 'ep-' + str(episode) + '-eps-' + str(epsilon) + '-.h5')

    @staticmethod
    def load_nn_model(model_name):
        return tf.keras.models.load_model('models/'+model_name)

    @staticmethod
    def get_most_trained_model():
        if not os.path.exists('models'):
            return None

        models_names = os.listdir(os.curdir + '/models')
        if len(models_names) == 0:
            return None

        models_names.sort(reverse=True)
        best_trained_model = models_names[0]
        return best_trained_model

    @staticmethod
    def get_epsilon(model_name):
        epsilon = model_name.split('-')[3]
        return float(epsilon)

    @staticmethod
    def get_episode(model_name):
        episode = model_name.split('-')[1]
        return int(episode)

    @staticmethod
    def get_epsilon_start_point():
        if not os.path.exists('models'):
            return 1
        most_trained_model = SaveLoadModule.get_most_trained_model()
        epsilon = SaveLoadModule.get_epsilon(most_trained_model)
        return epsilon

    @staticmethod
    def get_most_advanced_episode():
        if not os.path.exists('models'):
            return 0

        most_trained_model = SaveLoadModule.get_most_trained_model()
        actual_episode = SaveLoadModule.get_episode(most_trained_model)
        return actual_episode

    @staticmethod
    def get_models_sorted():
        models_names = os.listdir(os.curdir + '/models')
        models_names.sort()