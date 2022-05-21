import argparse as ap
import config 
import pandas as pd

class FraudInsuranceModel:
    def __init__(self, model_object_path):
        self.model = pd.read_pickle(model_object_path)

        
class PreprocessedUserInputData:
    def __init__(self, input_data_csv_path, output_data_csv_path):
        self.input_data_csv_path = input_data_csv_path
        self.output_data_csv_path = output_data_csv_path
        self.data_df = pd.read_csv(self.input_data_csv_path)
        self.pred_df = None
        
    def predict(self, model):
        self.pred_df = pd.DataFrame({
            'fraud_prob_score': model.predict_proba(self.data_df)[:, 1],
            'fraud_score': model.predict(self.data_df)})
        
    def load_data(self):
        return self.data_df
    
    def load_pred(self):
        return self.pred_df
    
    def save_data(self):
        df = pd.concat([self.data_df, self.pred_df], axis=1)
        df.to_csv(self.output_data_csv_path, index=False)

if __name__ == '__main__':
    
    print("Started running prediction of user input data.")
    fraud_insurance_model = FraudInsuranceModel(config.FRAUD_MODEL_PATH)
    preprocess_user_input_data = PreprocessedUserInputData(
        config.CHATBOT_OUTPUT_CSV_FILEPATH,
        config.CHATBOT_OUTPUT_CSV_FILEPATH)
    preprocess_user_input_data.predict(fraud_insurance_model.model)
    preprocess_user_input_data.save_data()
    
    pred_df = preprocess_user_input_data.load_pred()
    print("Processed a total of {} claim applications.".format(len(pred_df)))
    print("Done running prediction of user input data.")
    print("Appended predicted scores are stored in {}".format(config.CHATBOT_OUTPUT_CSV_FILEPATH))
    
    
    
    
    
        
        