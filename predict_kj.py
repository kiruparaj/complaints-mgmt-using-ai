from tensorflow.python.keras import models
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import build_model_kj

def predict_kj(input=None):
    #model = models.Sequential()
    model = load_model('./models/compliants-mgmt-mlp_07_Aug_2018_22_29_01.h5')
    print('==================== PREPARTING A PREDICTION KJ ================================ ')
    x = np.array([])
    x = np.append(x,'SOMEONE USE MY INFORMATION. THIS ISNT MY ACCOUNT.')
    print(model.predict(x))



def predict(input_string):
    print('==================== PREPARTING A PREDICTION ================================ ')
    x = np.array([])
    x = np.append(x,input_string)
    model = load_trained_model()
    print(model.predict(x))

def create_model():
    print('==================== CREATING A MODEL INSTANCE ================================ ')
    # Create model instance.
    model = build_model_kj.mlp_model(layers=2,
                                  units=64,
                                  dropout_rate=0.2,
                                  input_shape=(20000,),
                                  num_classes=11)

    return model

def load_trained_model():
   model = create_model()
   weights_path = './models/compliants-mgmt-mlp_07_Aug_2018_22_29_01.h5'
   model.load_weights(weights_path)
   return model

if __name__ == '__main__':
    print('Calling predict')
    predict('SOMEONE USE MY INFORMATION. THIS ISNT MY ACCOUNT.')
   # predict_kj('SOMEONE USE MY INFORMATION. THIS ISNT MY ACCOUNT.')
    print('Init Successful')

    # try this https://www.opencodez.com/python/text-classification-using-keras.htm