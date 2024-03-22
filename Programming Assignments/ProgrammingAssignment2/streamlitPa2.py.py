#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer# for data transformation
from sklearn.pipeline import Pipeline#for data preprocessing pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
def data(path):
    return pd.read_csv(path, sep=';')#load data from csv file
def preprocess_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('quality', axis=1), data['quality'], test_size=0.3, random_state=50)  # Splitting the data into training and testing sets
    return X_train, X_test, y_train, y_test
def gen_preprocessor(floa_features, obj_features):
    preprocessor = ColumnTransformer(transformers=[('floa', Pipeline(steps=[('scaler', StandardScaler())]), floa_features),  # Preprocessing float features using StandardScaler
                                                ('obj', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]), obj_features)])  # Preprocessing object features using OneHotEncoder
    return preprocessor
def train_model(X_train, y_train, preprocessor, model_params):
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor())])  # Creating a pipeline with preprocessing and model
    grid_search = GridSearchCV(pipe, model_params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')  # Setting up grid search for hyperparameter tuning
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
def visualize_prediction_intervals(y_true, y_pred, percentile=95):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, color='blue', alpha=0.6)  # Scatter plot of true vs predicted values
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='green', linestyle='-.')  # Plotting the identity line
    ax.fill_between([y_true.min(), y_true.max()], 
                    np.percentile(y_pred, (100 - percentile) / 2), np.percentile(y_pred, 100 - (100 - percentile) / 2), color='gray', alpha=0.3)  # Filling the area between upper and lower prediction intervals
    ax.set_xlabel('True Quality')
    ax.set_ylabel('Predicted Quality')
    ax.set_title(f'{percentile}% Prediction Intervals')
    st.pyplot(fig)
def visualize_feature_importance(model, X_train, feature_names):
    importances = model.named_steps['model'].feature_importances_  # Extracting feature importances from the trained model
    indices = np.argsort(importances)[::-1]  # Sorting feature importances in descending order
    fig, ax = plt.subplots(figsize=(10, 6))  # Creating a subplot for visualization
    bars = ax.bar(range(X_train.shape[1]), importances[indices], align="center", color=plt.cm.viridis(np.linspace(0, 1, len(indices))))  # Creating bar plot of feature importances
    ax.set_title("Feature Importances")
    ax.set_xticks(range(X_train.shape[1]))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_xlim([-1, X_train.shape[1]])
    for i in range(len(indices) - 1):  # Loop over the sorted indices of feature importances, excluding the last index
        x1 = bars[i].get_x() + bars[i].get_width() / 2  # Calculate x-coordinate of the center of the current bar
        x2 = bars[i + 1].get_x() + bars[i + 1].get_width() / 2  # Calculate x-coordinate of the center of the next bar
        y1 = importances[indices[i]]  # Get feature importance of the current feature
        y2 = importances[indices[i + 1]]  # Get feature importance of the next feature
        ax.plot([x1, x2], [y1, y2], color='red', linestyle='-.', linewidth=1)  # Plot a line between the centers of the two bars
    st.pyplot(fig)
def visualize_residuals(y_true, y_pred):
    residuals = y_true - y_pred  # Calculating residuals by subtracting predicted values from true values
    fig, ax = plt.subplots(figsize=(10, 6))  # Creating a subplot for residual plot visualization
    sns.residplot(y=y_pred, x=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1}, ax=ax)  # Plotting residual plot with lowess smoothing and a red trend line
    ax.set_title('Residual Plot')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)  
def visualize_distribution(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(y_true, label='True Values', color='green', linestyle='-.', ax=ax)
    sns.kdeplot(y_pred, label='Predicted Values', color='orange', ax=ax)
    ax.set_title('Distribution of True and Predicted Values')
    ax.set_xlabel('Quality')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)
# Function to visualize feature distribution
def visualize_feature_distribution(data):
    fig, axes = plt.subplots(3, 4, figsize=(10, 6))
    for i, feature in enumerate(data.columns):  # Iterating over each column in the dataset
        sns.histplot(data[feature], kde=True, ax=axes[i//4, i%4])  # Plotting the histogram of the current feature with KDE
        axes[i//4, i%4].set_title(f'Distribution of {feature}')  # Setting the title of the subplot as the current feature's distribution
        axes[i//4, i%4].set_xlabel(feature)  # Setting the x-label of the subplot as the current feature
    plt.tight_layout()
    st.pyplot(fig)
# streamlit app
def main():
    wine_df=[data('H:\Sem6\EE769-ML\Programming Assignments\PA2\wine+quality\winequality-'+i+'.csv')for i in ['red','white']]#wine dataset
    features = [wine_df[0].drop('quality', axis=1).select_dtypes(include=[i]).columns for i in ['float64','object']]  # Extracting columns of specific data types (float64 and object) from the first DataFrame in wine_df
    (X_red_train, X_red_test, y_red_train, y_red_test), (X_white_train, X_white_test, y_white_train, y_white_test) = [preprocess_data(wine_df[i]) for i in range(2)]
    preprocessor = gen_preprocessor(features[0], features[1])  # Generating a preprocessor using the extracted float and object features
    params = {'model__n_estimators': [100, 150], 'model__max_depth': [10, 15]}  # Defining a dictionary of hyperparameters for tuning the RandomForestRegressor
    model=[train_model(X_red_train, y_red_train, preprocessor, params),train_model(X_white_train, y_white_train, preprocessor,params)]#red and white wine trained models
    st.title('Wine Quality Predictor')
    wine_type = st.selectbox("Select Wine Type:", ('Red','White'))#selecting wine type
    X_train, X_test, y_test, usemodel = (X_red_train, X_red_test, y_red_test, model[0]) if wine_type == 'Red' else (X_white_train, X_white_test, y_white_test, model[1])  # Assigning training and testing data based on selected wine type
    st.sidebar.title('Adjust Wine Features')
    features_sliders = {feature: st.sidebar.slider(f'Select {feature}', float(X_test[feature].min()), float(X_test[feature].max()), float(X_test[feature].mean())) for feature in X_test.columns}  # Creating sliders for selecting feature values on the sidebar
    if st.button('Predict the Quality'): 
        use_data = pd.DataFrame([features_sliders])  # Creating a DataFrame with selected feature values
        prediction = usemodel.predict(use_data)  # Making predictions using the selected model and feature values
        st.success(f'Predicted Quality: {prediction[0]:.4f}')  # Displaying the predicted quality with four decimal places
    y_pred = usemodel.predict(X_test)#predictions based on model
    mse,mae = mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred)
    st.write(f'Mean Squared Error on Test Set: {mse:.2f}"\n" Mean Absolute Error on Test Set: {mae:.2f}')
    visualize_prediction_intervals(y_test, y_pred); visualize_feature_importance(usemodel, X_train, X_train.columns)
    more_visualizations = st.checkbox('Do you want more Visualizations')
    st.write(f"Show additional visualizations = {more_visualizations}")
    if more_visualizations:
        visualize_residuals(y_test, y_pred);visualize_distribution(y_test, y_pred);visualize_feature_distribution(X_train)
if __name__ == "__main__":
    main()
