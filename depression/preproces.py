import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import numpy as np

data = pd.read_csv('resources/Student Depression Dataset.csv')

preprocesed_data = data.drop(columns=['id', 'City', 'Profession', 'Work Pressure'])

values = preprocesed_data['Degree'].unique()
print(values)

def map_degree_level(degree):
    undergrad = ['B.Pharm', 'BSc', 'BA', 'BCA', 'B.Ed', 'LLB', 'BE', 'BHM', 'B.Com', 'B.Arch', 'B.Tech', 'BBA']
    postgrad = ['M.Tech', 'M.Ed', 'MSc', 'M.Pharm', 'MCA', 'MA', 'MBA', 'M.Com', 'ME', 'MHM', 'LLM', 'MD']
    doctorate = ['PhD']
    school = ['Class 12']
    if degree in undergrad:
        return 1
    elif degree in postgrad:
        return 2
    elif degree in doctorate:
        return 3
    elif degree in school:
        return 0
    else:
        return -1  # Otros/no clasificados


# Encode Gender
preprocesed_data.loc[:, ['Gender']] = preprocesed_data['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Encode Sleep Duration
order_sleep = ['Less than 5 hours', '5-6 hours', 'Others', '7-8 hours', 'More than 8 hours']
sleep_duration_encoder = OrdinalEncoder(categories=[order_sleep])
preprocesed_data.loc[:, ['Sleep Duration']] = sleep_duration_encoder.fit_transform(preprocesed_data[['Sleep Duration']])

# Encode Dietary Habits
preprocesed_data.loc[:, ['Dietary Habits']] = preprocesed_data[['Dietary Habits']].replace('Others', 'Moderate')
order_dietary = ['Unhealthy', 'Moderate', 'Healthy']
dietary_habits_encoder = OrdinalEncoder(categories=[order_dietary])
preprocesed_data.loc[:, ['Dietary Habits']] = dietary_habits_encoder.fit_transform(preprocesed_data[['Dietary Habits']])

# Encode Have you ever had suicidal thoughts?
preprocesed_data.loc[:, ['Have you ever had suicidal thoughts ?']] = preprocesed_data['Have you ever had suicidal thoughts ?'].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode Family History of Mental Illness
preprocesed_data.loc[:, ['Family History of Mental Illness']] = preprocesed_data['Family History of Mental Illness'].apply(lambda x: 1 if x == 'Yes' else 0)

## Encode Degree
preprocesed_data.loc[:, ['Degree']] = preprocesed_data['Degree'].apply(map_degree_level)

def fillna_random(df):
    for col in df.columns:
        if df[col].isnull().any():
            valores = df[col].dropna().values
            df[col] = df[col].apply(
                lambda x: np.random.choice(valores) if pd.isna(x) else x
            )
    return df

# 1) Rellenas todo el df
preprocesed_data = fillna_random(preprocesed_data)

preprocesed_data.to_csv('resources/preprocessed_student_depression_data.csv', index=False)


for col in preprocesed_data.columns:
    print(f"{col}: {preprocesed_data[col].min()}, {preprocesed_data[col].max()}")


