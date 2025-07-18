# -*- coding: utf-8 -*-            
# @Author : Tongqing Shen
# @Email : tqshen95@163.com
# @Time : 2024/7/3 14:51

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import itertools
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import shap
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,  activation=nn.ReLU(), outfunction=nn.Softplus(), seed=0):
        super(NeuralNet, self).__init__()
        self.seed=seed
        torch.manual_seed(self.seed)
        self.hidden_layers = nn.ModuleList()
        if hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
            for i in range(len(hidden_sizes) - 1):
                self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        for layer in self.hidden_layers:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0.1)

        self.activation = activation
        self.outfunction = outfunction
        self.output_layer = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        out = self.output_layer(x)
        out = self.outfunction(out)
        return out

def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

def train_model(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze(1).tolist())
    return predictions

def standardize_data(train_data, test_data=None):
    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    if test_data is not None:
        test_data_normalized = scaler.transform(test_data)
        return train_data_normalized, test_data_normalized, scaler
    else:
        return train_data_normalized, scaler

def standardize_data_pre(train_data, scaler):
    train_data_normalized = scaler.transform(train_data)
    return train_data_normalized

def reverse_standardization(predictions, scaler):
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

def main(use_saved_model = False,CvGs = False, DefHp = True, CvEva = False, ReTrain = False, pre = True, explan = False):

    train_data = pd.read_csv('Input1_Train_NDVI_monT104.csv')  # Load training data
    train_data_shuffled = train_data.sample(frac=1, random_state=10).reset_index(drop=True)

    # Normalize features based on baseline data
    GYH_data = pd.read_csv('Input2_Prediction_baseline.csv', header=None)
    # Normalize the target variable using the measured log10(Q10)
    GYH_data_target = pd.read_csv('Input1_Train_NDVI_monT104.csv')

    train_features = train_data_shuffled.iloc[:, 1:5].values
    train_targets = train_data_shuffled.iloc[:, 11].values

    GYH_features = GYH_data.iloc[:, 1:5].values
    GYH_targets = GYH_data_target.iloc[:, 11].values

    # Data normalization
    testxxx, scaler_features = standardize_data(GYH_features)
    train_features_std = standardize_data_pre(train_features,scaler_features)

    testyyy, scaler_targets = standardize_data(GYH_targets.reshape(-1, 1))
    train_targets_std = standardize_data_pre(train_targets.reshape(-1, 1), scaler_targets)

    # Use grid search with cross-validation to find the optimal hyperparameters.
    if CvGs:
        hidden_sizes_grid = [[10, 10, 10], [15,15,15], [20, 20, 20],[10,15,10],[15,20,15]]
        learning_rates = [0.01, 0.001, 0.0001]
        batch_sizes =  [16, 32, 64]
        weight_decays = [0.00001,0.00003,0.00005]
        num_epochs = [300, 500,700,1000]

        param_grid = itertools.product(hidden_sizes_grid, learning_rates, batch_sizes,weight_decays, num_epochs)

        best_params = None
        best_val_loss = float('inf')

        num_runs = 20

        for params in param_grid:
            hidden_sizes, learning_rate, batch_size, weight_decay, num_epochs = params
            print(f'Trying params: hidden_sizes={hidden_sizes}, learning_rate={learning_rate}, batch_size={batch_size},weight_decay={weight_decay}, num_epochs={num_epochs}')

            all_val_losses = []
            all_r2_scores = []
            avg_val_loss = 0.0

            for run in range(num_runs):
                print(f'Running cross-validation {run + 1}/{num_runs} with seed {run}')

                random_seed = run
                model = NeuralNet(input_size=4, hidden_sizes=hidden_sizes, output_size=1, seed = random_seed)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
                fold_val_losses = []
                fold_r2_scores = []
                fold = 0

                for train_index, val_index in kf.split(train_features_std):
                    fold += 1
                    X_train, X_val = train_features_std[train_index], train_features_std[val_index]
                    y_train, y_val = train_targets_std[train_index], train_targets_std[val_index]

                    train_dataset = CustomDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                    val_dataset = CustomDataset(X_val)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

                    for epoch in range(num_epochs):
                        train_loss = train_model(model, criterion, optimizer, train_loader)

                    val_predictions = predict(model, val_loader)
                    val_predictions = reverse_standardization(np.array(val_predictions).reshape(-1, 1), scaler_targets)
                    val_targets =  reverse_standardization(np.array(y_val).reshape(-1, 1), scaler_targets)

                    val_loss = criterion(torch.tensor(val_predictions),
                                         torch.tensor(val_targets, dtype=torch.float32)).item()

                    fold_val_losses.append(val_loss)

                    r2 = r2_score(val_targets, val_predictions)

                    fold_r2_scores.append(r2)
                # Compute average performance for this run
                avg_fold_val_loss = np.mean(fold_val_losses)
                avg_fold_r2_score = np.mean(fold_r2_scores)

                all_val_losses.append(avg_fold_val_loss)
                all_r2_scores.append(avg_fold_r2_score)

            # After 20 runs, compute the average validation loss and R² score
            avg_val_loss = np.mean(all_val_losses)
            avg_r2_score = np.mean(all_r2_scores)

            print(f'Average Validation Loss for this set of params: {avg_val_loss:.4f}')
            print(f'Average R^2 Score for this set of params: {avg_r2_score:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params = params

        hidden_sizes, learning_rate, batch_size, weight_decay, num_epochs = best_params
        print(f'Best params found: hidden_sizes={hidden_sizes}, learning_rate={learning_rate}, batch_size={batch_size},weight_decay={weight_decay}, num_epochs={num_epochs}')

    # Define the optimal hyperparameters
    if DefHp:

        input_size = 4
        hidden_sizes = [20, 20, 20]
        output_size = 1
        learning_rate = 0.001
        batch_size = 32
        weight_decay = 0.00005
        num_epochs = 1000

        model = NeuralNet(input_size, hidden_sizes, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Assess model robustness by altering random seeds.
    if CvEva:

        num_iterations = 500
        all_results_total = []
        all_r2_scores_total = []
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")

            seed = iteration
            np.random.seed(seed)
            model = NeuralNet(input_size, hidden_sizes, output_size, seed=seed)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            kf = KFold(n_splits=10, shuffle=True, random_state=seed)
            fold = 0
            all_results = []
            all_r2_scores = []
            avg_val_loss2 = 0.0
            for train_index, val_index in kf.split(train_features_std):
                fold += 1
                print(f'Fold [{fold}/{kf.n_splits}]')

                X_train, X_val = train_features_std[train_index], train_features_std[val_index]
                y_train, y_val = train_targets_std[train_index], train_targets_std[val_index]

                train_dataset = CustomDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                val_dataset = CustomDataset(X_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                for epoch in range(num_epochs):
                    train_loss = train_model(model, criterion, optimizer, train_loader)

                val_predictions = predict(model, val_loader)

                val_predictions = reverse_standardization(np.array(val_predictions).reshape(-1, 1), scaler_targets)
                val_targets = reverse_standardization(np.array(y_val).reshape(-1, 1), scaler_targets)

                fold_results = pd.DataFrame({
                    'Prediction': val_predictions.squeeze(),
                    'Target': val_targets.squeeze()
                })
                all_results.append(fold_results)

                r2 = r2_score(val_targets, val_predictions)
                all_r2_scores.append(r2)

                val_loss = criterion(torch.tensor(val_predictions),
                                     torch.tensor(val_targets, dtype=torch.float32)).item()

                avg_val_loss2 += val_loss / kf.n_splits

            print(f'Average Validation Loss for Iteration {iteration + 1}: {avg_val_loss2:.4f}')
            print(f'Average R^2 Score for Iteration {iteration + 1}: {np.mean(all_r2_scores):.4f}')

            all_results_total.append(pd.concat(all_results, ignore_index=True))
            all_r2_scores_total.append(np.mean(all_r2_scores))

        all_results_df = pd.concat(all_results_total, ignore_index=True)
        all_results_df.to_csv('output1_cross_validation_results_500_iterations_4v_104.csv', index=False)


        r2_scores_df = pd.DataFrame({
            'Iteration': range(1, num_iterations + 1),
            'Average R^2 Score': all_r2_scores_total
        })
        r2_scores_df.to_csv('output2_r2_scores_500_iterations_4v_104.csv', index=False)

        print("Completed 100 iterations of cross-validation.")

    # Train the final model and save it.
    if ReTrain:

        num_runs = 500
        for run in range(num_runs):
            seed = 0 + run
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = NeuralNet(input_size, hidden_sizes, output_size, seed=seed)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            full_train_dataset = CustomDataset(train_features_std, train_targets_std)
            full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False)

            if use_saved_model and os.path.exists('saved_model.pth'):
                model.load_state_dict(torch.load('saved_model.pth'))
                print(f'Run {run + 1}: Using saved model parameters for prediction.')
            else:
                for epoch in range(num_epochs):
                    train_loss = train_model(model, criterion, optimizer, full_train_loader)
                    print(f'Run {run + 1}, Epoch [{epoch + 1}/{num_epochs}], Training loss on full dataset: {train_loss:.4f}')

                torch.save(model.state_dict(), f'saved_model_run_4v_104_{run+1}.pth')

    if pre:

        model.eval()
        Index_dataset = 'indexes_txt.csv'


        Main_folder = 'G:/input_Cmip_scenario'
        Output_folder = 'G:/output_Cmip_scenario'
        Temp_folder = 'G:/temporary'

        indexes = np.genfromtxt(Index_dataset, skip_header=1, delimiter=",")
        index_list = indexes[:, -1]

        for sub_folder in os.listdir(Main_folder):
            sub_folder_path = os.path.join(Main_folder, sub_folder)
            if os.path.isdir(sub_folder_path):
                print(f'Processing folder: {sub_folder}')

                for predict_file in os.listdir(sub_folder_path):
                    if predict_file.endswith('.csv'):
                        Predict_file = os.path.join(sub_folder_path, predict_file)
                        print(f'Processing file: {predict_file}')

                        test_data = pd.read_csv(Predict_file, header=None)
                        test_features = test_data.iloc[:, 1:5].values

                        test_features_std = standardize_data_pre(test_features,
                                                                 scaler_features)

                        test_dataset = CustomDataset(test_features_std)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        all_predictions = []

                        if os.path.exists(Temp_folder):
                            for file in os.listdir(Temp_folder):
                                file_path = os.path.join(Temp_folder, file)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                        else:
                            os.makedirs(Temp_folder, exist_ok=True)

                        # seed 0-100 and R2>0.7
                        specific_runs = [1, 8, 13, 16, 23, 26, 28, 30, 32, 33, 35, 51, 54, 59, 60, 71,
                                         82, 84, 85, 92, 94, 97]

                        for run in specific_runs:

                            model = NeuralNet(input_size, hidden_sizes, output_size, seed=0 + run)
                            model.load_state_dict(torch.load(f'saved_model_run_4v_104_{run}.pth'))

                            test_predictions = predict(model, test_loader)
                            test_predictions = reverse_standardization(np.array(test_predictions).reshape(-1, 1),
                                                                       scaler_targets)
                            test_predictions = np.power(10, test_predictions).flatten().tolist()

                            temp_csv_file = os.path.join(Temp_folder,
                                                         f'{os.path.splitext(predict_file)[0]}_run_{run}.csv')
                            pd.DataFrame(test_predictions).to_csv(temp_csv_file, index=False, header=False)

                        prediction_files = [os.path.join(Temp_folder, f) for f in os.listdir(Temp_folder) if
                                            f.startswith(os.path.splitext(predict_file)[0]) and f.endswith('.csv')]
                        predictions_list = [pd.read_csv(file, header=None).values.flatten() for file in
                                            prediction_files]
                        mean_predictions = np.mean(np.array(predictions_list), axis=0)

                        j = 0
                        # k = 0
                        ASCII_list = []
                        for i in index_list:
                            if i == 0:
                                ASCII_list.append(-9999)
                            else:
                                ASCII_list.append(test_predictions[j])
                                j += 1

                        ASCII_list_split = func(ASCII_list, 2821)

                        output_sub_folder = os.path.join(Output_folder, sub_folder)
                        os.makedirs(output_sub_folder, exist_ok=True)

                        output_file = os.path.join(output_sub_folder,
                                                   f'{os.path.splitext(predict_file)[0]}_mean_prediction.txt')
                        with open(output_file, 'w') as f:
                            f.write('ncols         2821' + '\n')
                            f.write('nrows         1729' + '\n')
                            f.write('xllcorner     -664833.32935843' + '\n')
                            f.write('yllcorner     2865493.7575816' + '\n')
                            f.write('cellsize      1000' + '\n')
                            f.write('NODATA_value  -9999' + '\n')
                            for line_list in ASCII_list_split:
                                f.write(' '.join(map(str, line_list)) + '\n')
                        print(f'{predict_file} processing completed.')

    else:
        print('No prediction was performed.')

    # SHAP is used to interpret the model and understand feature contributions.
    if explan:

        Predict_file = 'Exp_R1.csv' #Regions that require SHAP interpretation

        test_data = pd.read_csv(Predict_file, header=None)
        test_features = test_data.iloc[:, 1:5].values

        test_features_std = standardize_data_pre(test_features, scaler_features)

        def predict_fn(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(inputs).numpy()
            return outputs

        all_importances = []

        num_runs = 500

        for run in range(num_runs):

            model = NeuralNet(input_size, hidden_sizes, output_size, seed=0 + run)
            model.load_state_dict(torch.load(f'saved_model_run_4v_104_{run + 1}.pth'))

            background_data = shap.sample(train_features_std, 104)

            n_samples = 10000
            random_indices = np.random.choice(test_features_std.shape[0], n_samples, replace=False)

            explainer = shap.KernelExplainer(predict_fn, background_data)
            Need_expla_data =  test_features_std[random_indices]
            shap_values = explainer.shap_values(Need_expla_data)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.array(shap_values)

            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

            feature_names = [f'Feature {i}' for i in range(Need_expla_data.shape[1])]
            importance_df = pd.DataFrame({'Model': [f'Model {run + 1}'] * len(feature_names),
                                          'Feature': feature_names,
                                          'Importance': mean_abs_shap_values})

            all_importances.append(importance_df)

            print(f"Feature importance for model {run + 1}:")
            print(importance_df)

        all_importances_df = pd.concat(all_importances, ignore_index=True)

        output_file = 'output3_feature_importances_R1.csv'
        all_importances_df.to_csv(output_file, index=False)

        print(f'Feature importances from all models have been saved to {output_file}')
    else:
        print('No SHAP explanation was performed.')

if __name__ == "__main__":
    main()