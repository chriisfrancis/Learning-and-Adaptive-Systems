import math
import pandas as pd

# -----------------------------------
# Function to calculate Entropy
# -----------------------------------
def entropy(data, target_col):
    # Get unique values in target column
    values = data[target_col].unique()
    
    entropy_value = 0
    
    for value in values:
        probability = len(data[data[target_col] == value]) / len(data)
        entropy_value -= probability * math.log2(probability)
    
    return entropy_value


# -----------------------------------
# Function to calculate Information Gain
# -----------------------------------
def information_gain(data, feature, target_col):
    # Total entropy before split
    total_entropy = entropy(data, target_col)
    
    # Get unique values of the feature
    values = data[feature].unique()
    
    weighted_entropy = 0
    
    for value in values:
        subset = data[data[feature] == value]
        probability = len(subset) / len(data)
        weighted_entropy += probability * entropy(subset, target_col)
    
    # Information Gain formula
    info_gain = total_entropy - weighted_entropy
    
    return info_gain


# -----------------------------------
# Example Dataset (Play Tennis)
# -----------------------------------
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)


# -----------------------------------
# Calculate Entropy of dataset
# -----------------------------------
print("Entropy of dataset:", entropy(df, 'Play'))


# -----------------------------------
# Calculate Information Gain for each feature
# -----------------------------------
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

for feature in features:
    ig = information_gain(df, feature, 'Play')
    print(f"Information Gain of {feature}: {ig}")