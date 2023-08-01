from django.shortcuts import render,HttpResponse
from .logistic_regression import LogisticRegression
import pickle
import numpy as np

# Create your views here.
# with open('D:/cancer/prediction/trained_logistic_regression_model.pkl', 'rb') as file:
#     logistic_regression_model = pickle.load(file)
with open('D:/cancer/prediction/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



import joblib

# Load the trained logistic regression model
with open('D:/cancer/prediction/trained_logistic_regression_model.joblib', 'rb') as file:
    logistic_regression_model = joblib.load(file)

# logistic_regression_model = LogisticRegression()
# logistic_regression_model.model.set_params(**model_params)



def predict_breast_cancer(request):
    if request.method == 'POST':
        # Get user input for each attribute from the form
        radius_mean = float(request.POST.get('radius_mean', ''))
        texture_mean = float(request.POST.get('texture_mean', ''))
        perimeter_mean = float(request.POST.get('perimeter_mean', ''))
        area_mean = float(request.POST.get('area_mean', ''))
        smoothness_mean = float(request.POST.get('smoothness_mean', ''))
        compactness_mean = float(request.POST.get('compactness_mean', ''))
        concavity_mean = float(request.POST.get('concavity_mean', ''))
        concave_points_mean = float(request.POST.get('concave_points_mean', ''))
        symmetry_mean = float(request.POST.get('symmetry_mean', ''))
        fractal_dimension_mean = float(request.POST.get('fractal_dimension_mean', ''))
        radius_se = float(request.POST.get('radius_se', ''))
        texture_se = float(request.POST.get('texture_se', ''))
        perimeter_se = float(request.POST.get('perimeter_se', ''))
        area_se = float(request.POST.get('area_se', ''))
        smoothness_se = float(request.POST.get('smoothness_se', ''))
        compactness_se = float(request.POST.get('compactness_se', ''))
        concavity_se = float(request.POST.get('concavity_se', ''))
        concave_points_se = float(request.POST.get('concave_points_se', ''))
        symmetry_se = float(request.POST.get('symmetry_se', ''))
        fractal_dimension_se = float(request.POST.get('fractal_dimension_se', ''))
        radius_worst = float(request.POST.get('radius_worst', ''))
        texture_worst = float(request.POST.get('texture_worst', ''))
        perimeter_worst = float(request.POST.get('perimeter_worst', ''))
        area_worst = float(request.POST.get('area_worst', ''))
        smoothness_worst = float(request.POST.get('smoothness_worst', ''))
        compactness_worst = float(request.POST.get('compactness_worst', ''))
        concavity_worst = float(request.POST.get('concavity_worst', ''))
        concave_points_worst = float(request.POST.get('concave_points_worst', ''))
        symmetry_worst = float(request.POST.get('symmetry_worst', ''))
        fractal_dimension_worst = float(request.POST.get('fractal_dimension_worst', ''))
        


        
        # Get other attributes similarly

        # Create a numpy array with the user input
        user_data = np.array([[radius_mean, texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,
                               perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,
                               smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])
        scaled_user_data = scaler.transform(user_data)
        # Use the trained model to make predictions
        context = {
            'prediction_result': prediction_result,
        }

        return render(request, 'prediction_form.html', context)


    return render(request, 'prediction_form.html')