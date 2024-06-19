import pickle as pkl

def load_processor():
    with open('models/processor.pkl','rb') as f:
        processor=pkl.load(f)
    return processor

def load_model():
    with open('models/model.pkl','rb') as f:
        model=pkl.load(f)
    return  model

def prediction(data):
    # Apply processor
    processor=load_processor()
    transform_data=processor.transform(data)

    # Do the predicton
    model=load_model()
    y_pred=model.predict(transform_data)

    return y_pred
