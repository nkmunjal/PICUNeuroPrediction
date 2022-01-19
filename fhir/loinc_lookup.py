import numpy as np
from functools import partial
import random

loinc_lookup_table = {
        "pH": "2744-1", "PCO2": "2019-8", "PaO2": "2703-7", "IonCalcium": "17863-2",
        "Sodium": "2951-2", "Potassium": "2823-3", "BUN": "3094-0", 
        "Creatinine": "2160-0", "Glucose": "2345-7", "CO2": "1963-8", "TotalCalcium": "17861-6",
        "WBC": "6690-2", "Hemoglobin": "718-7", "Platelets": "777-3", "PT": "5902-2",
        "PTT": "5901-4", "INR": "6301-6", "Temp": "8310-5", "RespRate": "9279-1",
        "HeartRate": "8867-4", "SBP": "8480-6", "DBP": "8462-4", "Age": "30525-0",
        "GCSMotor": "9268-4", "GCSEye": "9267-6", "GCSVerbal": "9270-0", "GCSTotal": "9269-2",
        "LOC": "80288-4", "RightPupilResponse": "79815-7",
        "LeftPupilResponse": "79899-1", "Hypothermia": "88668-9", "Sex": "72143-1",
        "Location": "80412-0","GCSIntub": "55285-1"
        }
reverse_loinc_lookup = {value:key for key,value in loinc_lookup_table.items()}

units = {
        "pH": "pH", "PCO2": "mmHg", "PaO2": "mmHg", "IonCalcium": "mg/dL",
        "Sodium": "mmol/L", "Potassium": "mmol/L", "BUN": "mg/dL",
        "Creatinine": "mg/dL", "Glucose": "mg/dL", "CO2": "mmol/dL", "TotalCalcium": "mg/dL",
        "WBC": "10*3/uL", "Hemoglobin": "g/dL", "Platelets": "10*3/uL", "PT": "s",
        "PTT": "s", "INR": "INR", "Temp": "deg C", "RespRate": "breaths/min",
        "HeartRate": "beats/min", "SBP": "mmHg", "DBP": "mmHg", "Age": "days",
        "GCSMotor": "", "GCSEye": "", "GCSVerbal": "", "GCSTotal": "",
        "LOC": "", "RightPupilResponse": "",
        "LeftPupilResponse": "", "Hypothermia": "", "Sex": "",
        "Location": ""
        }

model_data_points = ["pH", "PCO2", "PaO2", "IonCalcium", "Sodium", "Potassium", "BUN",
    "Creatinine", "Glucose", "CO2", "TotalCalcium", "WBC", "Hemoglobin",
    "Platelets", "PT", "PTT", "INR", "Temp", "RespRate", "HeartRate", "SBP", "DBP",
    "Age", "GCSMotor", "GCSEye", "GCSVerbal", "GCSTotal", "GCSIntub", "LOC",
    "RightPupilResponse", "LeftPupilResponse", "Hypothermia", "Location"]

model_vitals = ["Temp", "RespRate", "HeartRate", "SBP", "DBP"]
model_labs = ["pH", "PCO2", "PaO2", "IonCalcium", "Sodium", "Potassium", "BUN",
    "Creatinine", "Glucose", "CO2", "TotalCalcium", "WBC", "Hemoglobin",
    "Platelets", "PT", "PTT", "INR"]
model_exam = ["GCSMotor", "GCSEye", "GCSVerbal", "GCSTotal", "GCSIntub", "LOC", "RightPupilResponse", "LeftPupilResponse"]
model_other = ["Age","Hypothermia","Location"]

dataframe_all_columns = ['LowpH', 'HighpH', 'HighPCO2', 'LowPa02', 'LowIonCalcium',
       'HighIonCalcium', 'LowSodium', 'HighSodium', 'HighPotassium', 'HighBUN',
       'HighCreatinine', 'LowGlucose', 'HighGlucose', 'LowCO2', 'HighCO2',
       'LowTotalCalcium', 'HighTotalCalcium', 'LowWBC', 'HighWBC',
       'LowHemoglobin', 'HighHemoglobin', 'LowPlatelets', 'HighPT', 'HighPTT',
       'HighINR', 'HighTemp', 'LowTemp', 'HighRespRate', 'LowRespRate',
       'HighHeartRate', 'LowHeartRate', 'HighSBP', 'LowSBP', 'BIRTHDAY',
       'CNSInjury_No', 'CNSInjury_Yes', 'GCSWorstMotor_1', 'GCSWorstMotor_2',
       'GCSWorstMotor_3', 'GCSWorstMotor_4', 'GCSWorstMotor_5',
       'GCSWorstMotor_6', 'GCSWorstMotor_Unable to assess', 'GCSWorstTotal_10',
       'GCSWorstTotal_11', 'GCSWorstTotal_12', 'GCSWorstTotal_13',
       'GCSWorstTotal_14', 'GCSWorstTotal_15', 'GCSWorstTotal_3',
       'GCSWorstTotal_4', 'GCSWorstTotal_5', 'GCSWorstTotal_6',
       'GCSWorstTotal_7', 'GCSWorstTotal_8', 'GCSWorstTotal_9',
       'GCSWorstTotal_Unable to assess', 'GCSIntub_No', 'GCSIntub_Yes',
       'LOCWorst_Coma (unresponsive)', 'LOCWorst_No coma',
       'LOCWorst_Unable to assess', 'PupilWorst_Both non-reactive (> 3mm)',
       'PupilWorst_Both pupils < 3mm, cannot be scored',
       'PupilWorst_Both reactive', 'PupilWorst_One non-reactive (> 3mm)',
       'PupilHypo_No', 'PupilHypo_Yes', 'Sex_Female', 'Sex_Male']

dataframe_continuous_columns = {
        'LowpH': ['pH',0],
        'HighpH': ['pH',1],
        'HighPCO2': ['PCO2',1],
        'LowPa02': ['PaO2',0],
        'LowIonCalcium': ['IonCalcium',0],
        'HighIonCalcium': ['IonCalcium',1],
        'LowSodium': ['Sodium',0],
        'HighSodium': ['Sodium',1],
        'HighPotassium': ['Potassium',1],
        'HighBUN': ['BUN',1],
        'HighCreatinine': ['Creatinine',1],
        'LowGlucose': ['Glucose',0],
        'HighGlucose': ['Glucose',1],
        'LowCO2': ['CO2',0],
        'HighCO2': ['CO2',1],
        'LowTotalCalcium': ['TotalCalcium',0],
        'HighTotalCalcium': ['TotalCalcium',1],
        'LowWBC': ['WBC',0],
        'HighWBC': ['WBC',1],
        'LowHemoglobin': ['Hemoglobin',0],
        'HighHemoglobin': ['Hemoglobin',1],
        'LowPlatelets': ['Platelets',0],
        'HighPT': ['PT',1],
        'HighPTT': ['PTT',1],
        'HighINR': ['INR',1],
        'HighTemp': ['Temp',1],
        'LowTemp': ['Temp',0],
        'HighRespRate': ['RespRate',1],
        'LowRespRate': ['RespRate',0],
        'HighHeartRate': ['HeartRate',1],
        'LowHeartRate': ['HeartRate',0],
        'HighSBP': ['SBP',1],
        'LowSBP': ['SBP',0],
        'BIRTHDAY': ['Age',1],
}
dataframe_categorical_columns = {
        'CNSInjury_No': ['CNSInjury',1,'No'],
        'CNSInjury_Yes': ['CNSInjury',1,'Yes'],
        'GCSWorstMotor_1': ['GCSMotor',0,1],
        'GCSWorstMotor_2': ['GCSMotor',0,2],
        'GCSWorstMotor_3': ['GCSMotor',0,3],
        'GCSWorstMotor_4': ['GCSMotor',0,4],
        'GCSWorstMotor_5': ['GCSMotor',0,5],
        'GCSWorstMotor_6': ['GCSMotor',0,6],
        'GCSWorstMotor_Unable to assess': ['GCSMotor',1,-1],
        'GCSWorstTotal_10': ['GCSTotal',0,10],
        'GCSWorstTotal_11': ['GCSTotal',0,11],
        'GCSWorstTotal_12': ['GCSTotal',0,12],
        'GCSWorstTotal_13': ['GCSTotal',0,13],
        'GCSWorstTotal_14': ['GCSTotal',0,14],
        'GCSWorstTotal_15': ['GCSTotal',0,15],
        'GCSWorstTotal_3': ['GCSTotal',0,3],
        'GCSWorstTotal_4': ['GCSTotal',0,4],
        'GCSWorstTotal_5': ['GCSTotal',0,5],
        'GCSWorstTotal_6': ['GCSTotal',0,6],
        'GCSWorstTotal_7': ['GCSTotal',0,7],
        'GCSWorstTotal_8': ['GCSTotal',0,8],
        'GCSWorstTotal_9': ['GCSTotal',0,9],
        'GCSWorstTotal_Unable to assess': ['GCSTotal',1,-1],
        'GCSIntub_No': ['GCSIntub',0,'Initial GCS has legitimate values without interventions such as intubation and sedation'],
        'GCSIntub_Yes': ['GCSIntub',0,'Patient intubated'],
        'LOCWorst_Coma (unresponsive)': ['LOC',0,'Coma (unresponsive)'],
        'LOCWorst_No coma': ['LOC',0,'No coma'],
        'LOCWorst_Unable to assess': ['LOC',0,'Unable to assess'],
        'PupilHypo_No': ['Hypothermia',1,"No"],
        'PupilHypo_Yes': ['Hypothermia',1,"Yes"],
        'Sex_Female': ['Sex',0,"female"],
        'Sex_Male': ['Sex',0,"male"]
       }
dataframe_pupil_columns = {
        'PupilWorst_Both non-reactive (> 3mm)': ['PupilResponse',0,2,'Non-reactive (> 3mm)'],
        'PupilWorst_Both pupils < 3mm, cannot be scored': ['PupilResponse',0,2,'Non-reactive (< 3mm)'],
        'PupilWorst_Both reactive': ['PupilResponse',0,2,'Reactive'],
        'PupilWorst_One non-reactive (> 3mm)': ['PupilResponse',0,1,'Reactive'],
        }

model_choices = {
        "LOC": ['Coma (unresponsive)','No coma','Unable to assess'],
        "RightPupilResponse": ['Non-reactive (> 3mm)','Non-reactive (< 3mm)','Reactive'],
        "LeftPupilResponse": ['Non-reactive (> 3mm)','Non-reactive (< 3mm)','Reactive'],
        "Hypothermia": ['No','Yes'],
        "Sex": ["Male","Female"],
        "Location": ['CHOP','UPMC','UCLA','PHNX','CNMC','MICH','CHOM','CHLA'],
        "GCSIntub": ["Patient intubated","Initial GCS has legitimate values without interventions such as intubation and sedation"]
        }

model_generator = {
        "pH": [partial(np.random.normal,loc=7.4,scale=.05),2],
        "PCO2": [partial(np.random.normal,loc=40,scale=5),0],
        "PaO2": [partial(np.random.normal,loc=90,scale=20),0],
        "IonCalcium": [partial(np.random.normal,loc=1.1,scale=0.2),2],
        "Sodium": [partial(np.random.normal,loc=140,scale=5),0],
        "Potassium": [partial(np.random.normal,loc=4,scale=0.5),1],
        "BUN": [partial(np.random.normal,loc=20,scale=5),1],
        "Creatinine": [partial(np.random.normal,loc=1,scale=0.2),2],
        "Glucose": [partial(np.random.normal,loc=150,scale=20),0],
        "CO2": [partial(np.random.normal,loc=24,scale=3),1],
        "TotalCalcium": [partial(np.random.normal,loc=9,scale=1),1],
        "WBC": [partial(np.random.normal,loc=9,scale=2),1],
        "Hemoglobin": [partial(np.random.normal,loc=13,scale=3),1],
        "Platelets": [partial(np.random.normal,loc=400,scale=150),0],
        "PT": [partial(np.random.normal,loc=18,scale=5),1],
        "PTT": [partial(np.random.normal,loc=40,scale=5),1],
        "INR": [partial(np.random.normal,loc=1.3,scale=.2),1],
        "Temp": [partial(np.random.normal,loc=37,scale=1),1],
        "RespRate": [partial(np.random.normal,loc=18,scale=5),0],
        "HeartRate": [partial(np.random.normal,loc=90,scale=20),0],
        "SBP": [partial(np.random.normal,loc=110,scale=20),0],
        "DBP": [partial(np.random.normal,loc=80,scale=20),0],
        "Age": [partial(np.random.normal,loc=3000,scale=800),0],
        "GCSMotor": [partial(random.randint,1,6),0],
        "GCSEye": [partial(random.randint,1,4),0],
        "GCSVerbal": [partial(random.randint,1,5),0],
        "GCSTotal": [partial(random.randint,3,15),0],
        "GCSIntub": [partial(random.choice,model_choices["GCSIntub"]),-1],
        "LOC": [partial(random.choice,model_choices["LOC"]),-1],
        "RightPupilResponse": [partial(random.choice,model_choices["RightPupilResponse"]),-1],
        "LeftPupilResponse": [partial(random.choice,model_choices["LeftPupilResponse"]),-1],
        "Hypothermia": [partial(random.choice,model_choices["Hypothermia"]),-1],
        "Sex": [partial(random.choice,model_choices["Sex"]),-1],
        "Location": [partial(random.choice,model_choices["Location"]),-1]
        }

def name_to_loinc(name):
    return loinc_lookup_table[name]

def loinc_to_name(loinc):
    return reverse_loinc_lookup[loinc]

